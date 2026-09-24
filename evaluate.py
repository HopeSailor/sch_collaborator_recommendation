#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from utils import (
    evaluate_ranking,
    load_graph,
    pair_neighbors,
    prepare_model,
    resolve_device,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--topk", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--cold-start", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu")
    config = checkpoint["config"]
    model_config = config["model"]
    device = resolve_device(str(model_config["device"]))
    graph = load_graph(config["data"]["graph"])
    model, scholars, scholar_index = prepare_model(graph, model_config, device)
    model.load_state_dict(checkpoint["model_state"])

    queries = None
    if args.cold_start:
        neighbors = pair_neighbors(graph["splits"]["train_pairs"])
        queries = {
            scholar
            for scholar in scholars
            if len(neighbors.get(scholar, set())) < 5
        }

    metrics = evaluate_ranking(
        model,
        scholars,
        scholar_index,
        graph["splits"]["test_pairs"],
        graph["splits"]["train_pairs"]
        + graph["splits"]["validation_pairs"],
        sorted(set(args.topk)),
        queries=queries,
    )
    result = {
        "subset": "cold-start" if args.cold_start else "all",
        "metrics": metrics,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
