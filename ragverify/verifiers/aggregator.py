"""Combine NLI and heuristic signals into a final sentence label."""


def aggregate(nli: dict, heur: dict, mode: str = "combined") -> str:
    if mode == "nli":
        return nli["label"]
    if mode == "heuristics":
        return "Supported" if heur["label"] == "Supported" else heur["label"]

    # Combined fusion precedence:
    # 1. NLI contradiction wins. Heuristic signals measure overlap/similarity,
    #    which stays HIGH for near-verbatim contradictions, so they must not
    #    veto a contradiction verdict.
    # 2. NLI entailment wins. A paraphrase may score low on lexical overlap.
    # 3. Strong heuristic support rescues a neutral NLI verdict (evidence
    #    sentence selected by retrieval but not surfaced by NLI).
    # 4. Otherwise the sentence is Unsupported.
    if nli["label"] == "Contradicted":
        return "Contradicted"
    if nli["label"] == "Supported":
        return "Supported"
    if heur["label"] == "Supported":
        return "Supported"
    return "Unsupported"


def hallucination_score(labels: list) -> float:
    """(2 * contradicted + unsupported) / (2 * total) * 100"""
    if not labels:
        return 0.0
    c = labels.count("Contradicted")
    u = labels.count("Unsupported")
    return round((2 * c + u) / (2 * len(labels)) * 100, 1)
