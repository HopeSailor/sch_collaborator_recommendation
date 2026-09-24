#!/usr/bin/env python3
"""Build a Scientific Collaboration Hypergraph from AMiner records."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import yaml


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = path.parent.resolve()
    for key, value in config["data"].items():
        config["data"][key] = str((base / value).resolve())
    config["output_dir"] = str((base / config["output_dir"]).resolve())
    config["checkpoint"] = str((base / config["checkpoint"]).resolve())
    return config


def normalized_label(value: object, fallback: str = "unknown") -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if text else fallback


def stable_id(prefix: str, label: str) -> str:
    digest = hashlib.sha1(label.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}:{digest}"


def file_fingerprint(path: Path) -> dict:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "sha256": digest.hexdigest(),
    }


def org_match_score(profile_org: object, candidate_org: object) -> tuple[float, float]:
    left = set(re.findall(r"[a-z0-9]+", str(profile_org or "").lower()))
    right = set(re.findall(r"[a-z0-9]+", str(candidate_org or "").lower()))
    if not left or not right:
        return (0.0, 0.0)
    jaccard = len(left & right) / len(left | right)
    sequence = SequenceMatcher(
        None,
        " ".join(sorted(left)),
        " ".join(sorted(right)),
    ).ratio()
    return (jaccard, sequence)


def profile_index(
    profiles_path: Path, resolution_strategy: str
) -> tuple[dict[str, dict], dict]:
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    by_aminer: dict[str, dict] = {}
    duplicate_aminer_ids = 0
    profiles_resolved = 0
    ambiguous_profiles_resolved = 0
    for profile in profiles:
        candidates = profile.get("aminer_candidates") or []
        if not candidates:
            continue
        if resolution_strategy == "unambiguous":
            if len(candidates) != 1:
                continue
            candidate = candidates[0]
            resolution_score = (1.0, 1.0, 0)
        elif resolution_strategy == "best_org":
            scored = [
                (org_match_score(profile.get("org"), candidate.get("org")), -position, candidate)
                for position, candidate in enumerate(candidates)
            ]
            best_score, negative_position, candidate = max(
                scored, key=lambda item: (item[0][0], item[0][1], item[1])
            )
            resolution_score = (*best_score, negative_position)
            if len(candidates) > 1:
                ambiguous_profiles_resolved += 1
        else:
            raise ValueError(f"Unsupported author_resolution: {resolution_strategy}")
        author_id = str(candidate.get("index", "")).strip()
        if not author_id:
            continue
        profiles_resolved += 1
        enriched = dict(profile)
        enriched["candidate"] = candidate
        enriched["resolution_score"] = resolution_score
        if author_id in by_aminer:
            duplicate_aminer_ids += 1
            if resolution_score > by_aminer[author_id]["resolution_score"]:
                by_aminer[author_id] = enriched
        else:
            by_aminer[author_id] = enriched
    audit = {
        "input_profiles": len(profiles),
        "resolution_strategy": resolution_strategy,
        "profiles_without_candidates": sum(
            1 for p in profiles if not (p.get("aminer_candidates") or [])
        ),
        "profiles_with_exactly_one_candidate": sum(
            1 for p in profiles if len(p.get("aminer_candidates") or []) == 1
        ),
        "profiles_resolved": profiles_resolved,
        "ambiguous_profiles_resolved": ambiguous_profiles_resolved,
        "unique_resolved_aminer_ids": len(by_aminer),
        "duplicate_resolved_aminer_ids": duplicate_aminer_ids,
    }
    return by_aminer, audit


def read_author_papers(
    author2paper_path: Path, eligible_authors: set[str]
) -> tuple[dict[int, set[str]], dict[str, set[int]]]:
    paper_authors: dict[int, set[str]] = defaultdict(set)
    author_papers: dict[str, set[int]] = defaultdict(set)
    with author2paper_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            author_id = parts[1]
            if author_id not in eligible_authors:
                continue
            try:
                paper_id = int(parts[2])
            except ValueError:
                continue
            paper_authors[paper_id].add(author_id)
            author_papers[author_id].add(paper_id)
    return paper_authors, author_papers


def collaboration_adjacency(
    paper_authors: dict[int, set[str]],
) -> tuple[dict[str, set[str]], dict[int, set[str]]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    collaborative: dict[int, set[str]] = {}
    for paper_id, authors in paper_authors.items():
        if len(authors) < 2:
            continue
        collaborative[paper_id] = authors
        for left, right in itertools.combinations(sorted(authors), 2):
            adjacency[left].add(right)
            adjacency[right].add(left)
    return adjacency, collaborative


def connected_components(adjacency: dict[str, set[str]]) -> list[set[str]]:
    components: list[set[str]] = []
    seen: set[str] = set()
    for start in sorted(adjacency):
        if start in seen:
            continue
        component: set[str] = set()
        queue = [start]
        seen.add(start)
        while queue:
            node = queue.pop()
            component.add(node)
            for neighbor in sorted(adjacency[node], reverse=True):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return sorted(components, key=lambda c: (-len(c), min(c)))


def select_authors(adjacency: dict[str, set[str]], maximum: int) -> list[str]:
    return sorted(adjacency, key=lambda author: (-len(adjacency[author]), author))[
        :maximum
    ]


def select_papers(
    collaborative: dict[int, set[str]], selected_authors: set[str], maximum: int
) -> dict[int, list[str]]:
    induced: list[tuple[int, list[str]]] = []
    for paper_id, authors in collaborative.items():
        retained = sorted(authors & selected_authors)
        if len(retained) >= 2:
            induced.append((paper_id, retained))
    induced.sort(key=lambda item: (-len(item[1]), item[0]))
    return dict(induced[:maximum])


def parse_selected_papers(papers_path: Path, wanted: set[int]) -> dict[int, dict]:
    if not wanted:
        return {}
    records: dict[int, dict] = {}
    current: dict | None = None
    maximum = max(wanted)

    def finish(record: dict | None) -> None:
        if record and record.get("id") in wanted:
            records[record["id"]] = record

    with papers_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if line.startswith("#index "):
                finish(current)
                try:
                    paper_id = int(line[7:].strip())
                except ValueError:
                    current = None
                    continue
                if paper_id > maximum:
                    current = None
                    break
                current = {
                    "id": paper_id,
                    "title": "",
                    "year": None,
                    "venue": "unknown",
                    "references": [],
                }
            elif current is not None and current["id"] in wanted:
                if line.startswith("#* "):
                    current["title"] = normalized_label(line[3:])
                elif line.startswith("#t "):
                    value = line[3:].strip()
                    current["year"] = int(value) if value.isdigit() else None
                elif line.startswith("#c "):
                    current["venue"] = normalized_label(line[3:])
                elif line.startswith("#% "):
                    value = line[3:].strip()
                    if value.isdigit():
                        current["references"].append(int(value))
        finish(current)
    for paper_id in wanted:
        records.setdefault(
            paper_id,
            {
                "id": paper_id,
                "title": f"AMiner paper {paper_id}",
                "year": None,
                "venue": "unknown",
                "references": [],
            },
        )
    return records


def pairs_for_papers(
    paper_ids: Iterable[int], paper_authors: dict[int, list[str]]
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for paper_id in paper_ids:
        for left, right in itertools.combinations(sorted(paper_authors[paper_id]), 2):
            pairs.add((left, right))
    return pairs


def split_papers(
    paper_ids: list[int], train_fraction: float, validation_fraction: float, seed: int
) -> tuple[list[int], list[int], list[int]]:
    shuffled = list(paper_ids)
    random.Random(seed).shuffle(shuffled)
    train_end = max(1, int(len(shuffled) * train_fraction))
    validation_end = max(train_end + 1, int(len(shuffled) * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, len(shuffled) - 1)
    return (
        sorted(shuffled[:train_end]),
        sorted(shuffled[train_end:validation_end]),
        sorted(shuffled[validation_end:]),
    )


def location_from_org(org: str) -> str:
    parts = [normalized_label(part) for part in re.split(r"[,;]", org) if part.strip()]
    if len(parts) < 2:
        return "unknown"
    return parts[-1]


class GraphBuilder:
    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.hyperedges: list[dict] = []
        self.edge_counter = 0

    def add_node(self, node_id: str, node_type: str, label: str, **extra: object) -> str:
        self.nodes.setdefault(
            node_id,
            {"id": node_id, "type": node_type, "label": label, **extra},
        )
        return node_id

    def add_category(self, node_type: str, label: str) -> str:
        prefixes = {
            "gender": "G",
            "career_stage": "C",
            "expertise": "E",
            "institution": "I",
            "productivity": "P",
            "venue": "V",
            "location": "L",
        }
        node_id = stable_id(prefixes[node_type], label)
        return self.add_node(node_id, node_type, label)

    def add_edge(
        self, edge_type: str, nodes: list[str], source: dict | None = None
    ) -> None:
        self.edge_counter += 1
        self.hyperedges.append(
            {
                "id": f"e{self.edge_counter:06d}",
                "type": edge_type,
                "nodes": nodes,
                "source": source or {},
            }
        )


def productivity_bins(author_ids: list[str], profiles: dict[str, dict]) -> dict[str, int]:
    ranked = sorted(
        author_ids,
        key=lambda a: (
            int(float(profiles[a]["candidate"].get("hi") or 0)),
            a,
        ),
    )
    count = len(ranked)
    return {author_id: min(4, (rank * 5) // count) for rank, author_id in enumerate(ranked)}


def build(config: dict) -> dict:
    build_cfg = config["build"]
    source_paths = {
        key: Path(config["data"][key])
        for key in ("profiles", "author2paper", "papers")
    }
    resolution_strategy = str(build_cfg.get("author_resolution", "unambiguous"))
    profiles, profile_audit = profile_index(
        source_paths["profiles"], resolution_strategy
    )
    paper_authors_all, _ = read_author_papers(
        source_paths["author2paper"], set(profiles)
    )
    adjacency, collaborative = collaboration_adjacency(paper_authors_all)
    components = connected_components(adjacency)
    if not components:
        raise RuntimeError("No collaborations found among resolved AMiner authors")
    selected_authors = select_authors(adjacency, int(build_cfg["max_scholars"]))
    selected_author_set = set(selected_authors)
    selected_paper_authors = select_papers(
        collaborative, selected_author_set, int(build_cfg["max_papers"])
    )
    if len(selected_paper_authors) < 3:
        raise RuntimeError("At least three collaborative papers are required")
    paper_metadata = parse_selected_papers(
        source_paths["papers"], set(selected_paper_authors)
    )

    train_papers, validation_papers, test_papers = split_papers(
        list(selected_paper_authors),
        float(build_cfg["train_fraction"]),
        float(build_cfg["validation_fraction"]),
        int(config["seed"]),
    )
    train_pairs = pairs_for_papers(train_papers, selected_paper_authors)
    validation_pairs = pairs_for_papers(validation_papers, selected_paper_authors) - train_pairs
    test_pairs = (
        pairs_for_papers(test_papers, selected_paper_authors)
        - train_pairs
        - validation_pairs
    )
    if not validation_pairs or not test_pairs:
        raise RuntimeError(
            "The deterministic paper split produced an empty held-out pair set; "
            "increase max_papers or change the seed."
        )

    graph = GraphBuilder()
    bins = productivity_bins(selected_authors, profiles)
    topics_per_scholar = int(build_cfg["topics_per_scholar"])

    for author_id in selected_authors:
        profile = profiles[author_id]
        candidate = profile["candidate"]
        scholar_node = graph.add_node(
            f"S:{author_id}",
            "scholar",
            normalized_label(profile.get("name")),
            source_id=author_id,
            profile_id=profile.get("id"),
            raw_h_index=int(float(candidate.get("hi") or 0)),
        )
        gender = graph.add_category("gender", normalized_label(profile.get("gender")))
        graph.add_edge("gender_association", [scholar_node, gender])
        career = graph.add_category("career_stage", normalized_label(profile.get("title")))
        graph.add_edge("career_progression", [scholar_node, career])
        institution_label = normalized_label(profile.get("org"))
        institution = graph.add_category("institution", institution_label)
        graph.add_edge("institutional_affiliation", [scholar_node, institution])
        location = graph.add_category("location", location_from_org(institution_label))
        graph.add_edge("geographic_proximity", [scholar_node, location])
        productivity_label = f"h_index_quintile_{bins[author_id] + 1}"
        productivity = graph.add_category("productivity", productivity_label)
        graph.add_edge("impact_link", [scholar_node, productivity])
        topics = [
            normalized_label(topic)
            for topic in str(candidate.get("topics") or "").split(";")
            if topic.strip()
        ][:topics_per_scholar]
        for topic in topics:
            expertise = graph.add_category("expertise", topic)
            graph.add_edge("research_interest", [scholar_node, expertise])

    selected_paper_set = set(selected_paper_authors)
    for paper_id, authors in sorted(selected_paper_authors.items()):
        metadata = paper_metadata[paper_id]
        publication = graph.add_node(
            f"A:{paper_id}",
            "publication",
            metadata["title"],
            source_id=paper_id,
            year=metadata["year"],
        )
        graph.add_edge(
            "coauthorship",
            [f"S:{author_id}" for author_id in authors] + [publication],
            {"paper_id": paper_id},
        )
        venue = graph.add_category("venue", normalized_label(metadata.get("venue")))
        graph.add_edge(
            "venue_association", [publication, venue], {"paper_id": paper_id}
        )
        for cited_id in metadata.get("references", []):
            if cited_id in selected_paper_set:
                graph.add_edge(
                    "citation_link",
                    [publication, f"A:{cited_id}"],
                    {"citing_paper_id": paper_id, "cited_paper_id": cited_id},
                )

    node_counts: dict[str, int] = defaultdict(int)
    for node in graph.nodes.values():
        node_counts[node["type"]] += 1
    edge_counts: dict[str, int] = defaultdict(int)
    for edge in graph.hyperedges:
        edge_counts[edge["type"]] += 1

    return {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "construction": {
            "description": "Scientific Collaboration Hypergraph",
            "author_resolution": resolution_strategy,
            "scholar_selection": "descending collaboration degree, then author ID",
            "location_approximation": "last comma/semicolon-delimited affiliation segment",
            "productivity_quantization": "within-sample h-index quintiles",
            "hypergraph_approximation_note": "Publication events are retained as multi-node coauthorship hyperedges",
            "config": build_cfg,
            "profile_audit": profile_audit,
            "component_count": len(components),
            "largest_component_size": len(components[0]),
            "source_files": {
                key: file_fingerprint(path) for key, path in source_paths.items()
            },
        },
        "statistics": {
            "nodes": len(graph.nodes),
            "hyperedges": len(graph.hyperedges),
            "node_counts": dict(sorted(node_counts.items())),
            "hyperedge_counts": dict(sorted(edge_counts.items())),
        },
        "nodes": list(graph.nodes.values()),
        "hyperedges": graph.hyperedges,
        "splits": {
            "train_papers": train_papers,
            "validation_papers": validation_papers,
            "test_papers": test_papers,
            "train_pairs": [list(pair) for pair in sorted(train_pairs)],
            "validation_pairs": [list(pair) for pair in sorted(validation_pairs)],
            "test_pairs": [list(pair) for pair in sorted(test_pairs)],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sch.yaml"))
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    graph = build(config)
    output_path = Path(config["data"]["graph"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "output": str(output_path),
        "statistics": graph["statistics"],
        "split_sizes": {
            key: len(value) for key, value in graph["splits"].items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
