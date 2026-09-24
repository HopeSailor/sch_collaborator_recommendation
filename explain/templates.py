from __future__ import annotations

from collections import defaultdict


TEMPLATES = {
    "institutional_affiliation": "Both scholars are affiliated with {value}.",
    "research_interest": "Both scholars work on {value}.",
    "venue_association": "Both scholars have published in {value}.",
}


def explain_pair(
    graph: dict, source_id: str, target_id: str, limit: int = 3
) -> list[str]:
    labels = {node["id"]: node["label"] for node in graph["nodes"]}
    scholars = {source_id: f"S:{source_id}", target_id: f"S:{target_id}"}
    attributes: dict[str, dict[str, set[str]]] = {
        scholar: defaultdict(set) for scholar in scholars
    }
    publications: dict[str, set[str]] = {scholar: set() for scholar in scholars}

    for edge in graph["hyperedges"]:
        nodes = set(edge["nodes"])
        for scholar, scholar_node in scholars.items():
            if scholar_node not in nodes:
                continue
            if edge["type"] in ("institutional_affiliation", "research_interest"):
                attributes[scholar][edge["type"]].update(nodes - {scholar_node})
            elif edge["type"] == "coauthorship":
                publications[scholar].update(
                    node for node in nodes if node.startswith("A:")
                )

    for edge in graph["hyperedges"]:
        if edge["type"] != "venue_association":
            continue
        nodes = set(edge["nodes"])
        venues = {node for node in nodes if node.startswith("V:")}
        for scholar in scholars:
            if publications[scholar] & nodes:
                attributes[scholar]["venue_association"].update(venues)

    explanations = []
    for relation in TEMPLATES:
        shared = (
            attributes[source_id][relation] & attributes[target_id][relation]
        )
        for node in sorted(shared, key=lambda item: labels[item]):
            explanations.append(TEMPLATES[relation].format(value=labels[node]))
            if len(explanations) == limit:
                return explanations
    return explanations
