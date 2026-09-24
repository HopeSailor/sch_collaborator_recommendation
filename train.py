#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch

from utils import (
    evaluate_ranking,
    load_config,
    load_graph,
    pair_neighbors,
    prepare_model,
    resolve_device,
    seed_everything,
)


def sample_negatives(
    pairs: list[tuple[str, str]],
    scholars: list[str],
    positives: dict[str, set[str]],
    generator: random.Random,
) -> list[str]:
    negatives = []
    for source, _ in pairs:
        forbidden = positives.get(source, set()) | {source}
        pool = [candidate for candidate in scholars if candidate not in forbidden]
        if not pool:
            raise RuntimeError(f"No negative candidate for scholar {source}")
        negatives.append(generator.choice(pool))
    return negatives


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sch.yaml"))
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    model_config = config["model"]
    seed = int(config["seed"])
    seed_everything(seed)
    device = resolve_device(str(model_config["device"]))
    graph = load_graph(config["data"]["graph"])
    model, scholars, scholar_index = prepare_model(graph, model_config, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(model_config["learning_rate"])
    )

    train_pairs = [tuple(map(str, pair)) for pair in graph["splits"]["train_pairs"]]
    directed_pairs = train_pairs + [(right, left) for left, right in train_pairs]
    all_pairs = (
        graph["splits"]["train_pairs"]
        + graph["splits"]["validation_pairs"]
        + graph["splits"]["test_pairs"]
    )
    positives = pair_neighbors(all_pairs)
    generator = random.Random(seed)
    cutoffs = [int(k) for k in model_config["evaluation_k"]]
    batch_size = int(model_config["batch_size"])
    margin = float(model_config["margin"])

    best_recall = -1.0
    best_epoch = 0
    stale_epochs = 0
    best_state = None
    history = []

    for epoch in range(1, int(model_config["max_epochs"]) + 1):
        model.train()
        generator.shuffle(directed_pairs)
        losses = []
        for start in range(0, len(directed_pairs), batch_size):
            batch = directed_pairs[start : start + batch_size]
            negatives = sample_negatives(batch, scholars, positives, generator)
            source = torch.tensor(
                [scholar_index[left] for left, _ in batch], device=device
            )
            positive = torch.tensor(
                [scholar_index[right] for _, right in batch], device=device
            )
            negative = torch.tensor(
                [scholar_index[target] for target in negatives], device=device
            )

            optimizer.zero_grad(set_to_none=True)
            embeddings = model.encode()
            positive_score = model.score(embeddings, source, positive)
            negative_score = model.score(embeddings, source, negative)
            loss = torch.relu(margin + negative_score - positive_score).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation = evaluate_ranking(
            model,
            scholars,
            scholar_index,
            graph["splits"]["validation_pairs"],
            graph["splits"]["train_pairs"],
            cutoffs,
        )
        recall = float(validation.get("recall@10", 0.0))
        history.append(
            {
                "epoch": epoch,
                "loss": sum(losses) / len(losses),
                **validation,
            }
        )
        print(f"epoch={epoch:03d} loss={history[-1]['loss']:.4f} recall@10={recall:.4f}")

        if recall > best_recall:
            best_recall = recall
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= int(model_config["early_stopping_patience"]):
                break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()

    checkpoint_path = Path(config["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "config": config,
            "best_epoch": best_epoch,
        },
        checkpoint_path,
    )

    test_metrics = evaluate_ranking(
        model,
        scholars,
        scholar_index,
        graph["splits"]["test_pairs"],
        graph["splits"]["train_pairs"]
        + graph["splits"]["validation_pairs"],
        cutoffs,
    )
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    result = {"best_epoch": best_epoch, "test": test_metrics}
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
