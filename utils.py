from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from model import SCHModel, sparse_adjacency


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = path.parent.resolve()
    config["data"] = {
        key: str((base / value).resolve()) for key, value in config["data"].items()
    }
    config["output_dir"] = str((base / config["output_dir"]).resolve())
    config["checkpoint"] = str((base / config["checkpoint"]).resolve())
    return config


def load_graph(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    return torch.device(
        "cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu"
    )


def pair_neighbors(pairs: list[list[str]]) -> dict[str, set[str]]:
    neighbors: dict[str, set[str]] = defaultdict(set)
    for left, right in pairs:
        neighbors[str(left)].add(str(right))
        neighbors[str(right)].add(str(left))
    return neighbors


def prepare_model(
    graph: dict, model_config: dict, device: torch.device
) -> tuple[SCHModel, list[str], dict[str, int]]:
    if int(model_config["layers"]) != 2:
        raise ValueError("SCH uses two hypergraph convolutional layers")
    node_index = {node["id"]: index for index, node in enumerate(graph["nodes"])}
    scholar_ids = sorted(
        str(node["source_id"])
        for node in graph["nodes"]
        if node["type"] == "scholar"
    )
    scholar_index = {
        scholar: node_index[f"S:{scholar}"] for scholar in scholar_ids
    }
    train_papers = set(graph["splits"]["train_papers"])
    hyperedges = [
        [node_index[node] for node in edge["nodes"]]
        for edge in graph["hyperedges"]
        if edge["type"] != "coauthorship"
        or edge.get("source", {}).get("paper_id") in train_papers
    ]
    adjacency = sparse_adjacency(len(node_index), hyperedges, device)
    model = SCHModel(
        len(node_index), int(model_config["embedding_dim"]), adjacency
    ).to(device)
    return model, scholar_ids, scholar_index


@torch.no_grad()
def evaluate_ranking(
    model: SCHModel,
    scholar_ids: list[str],
    scholar_index: dict[str, int],
    ground_truth_pairs: list[list[str]],
    excluded_pairs: list[list[str]],
    cutoffs: list[int],
    queries: set[str] | None = None,
) -> dict[str, float]:
    model.eval()
    embeddings = model.encode()
    ground_truth = pair_neighbors(ground_truth_pairs)
    excluded = pair_neighbors(excluded_pairs)
    values: dict[str, list[float]] = defaultdict(list)

    evaluated = 0
    for query in sorted(ground_truth):
        if queries is not None and query not in queries:
            continue
        candidates = [
            candidate
            for candidate in scholar_ids
            if candidate != query and candidate not in excluded.get(query, set())
        ]
        if not candidates:
            continue
        source = torch.full(
            (len(candidates),), scholar_index[query], device=embeddings.device
        )
        target = torch.tensor(
            [scholar_index[candidate] for candidate in candidates],
            device=embeddings.device,
        )
        scores = model.score(embeddings, source, target)
        order = torch.argsort(scores, descending=True).tolist()
        ranking = [candidates[index] for index in order]
        relevant = ground_truth[query]
        evaluated += 1

        for cutoff in cutoffs:
            hits = [
                float(candidate in relevant) for candidate in ranking[:cutoff]
            ]
            denominator = min(cutoff, len(relevant))
            recall = sum(hits) / denominator if denominator else 0.0
            dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
            ideal = sum(
                1.0 / math.log2(rank + 2)
                for rank in range(min(cutoff, len(relevant)))
            )
            values[f"recall@{cutoff}"].append(recall)
            values[f"ndcg@{cutoff}"].append(dcg / ideal if ideal else 0.0)

    metrics = {
        name: sum(items) / len(items) for name, items in sorted(values.items())
    }
    metrics["queries"] = evaluated
    return metrics
