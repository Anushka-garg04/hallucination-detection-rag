"""Source credibility screening -> verification percentage."""


def verification_percentage(docs: list) -> tuple:
    """(credible sources / total sources) * 100 — e.g. 5 of 10 trusted -> 50%."""
    total = len(docs)
    credible = sum(1 for d in docs if d.is_credible)
    pct = round(100.0 * credible / total, 1) if total else 0.0
    sources = []
    for d in docs:
        sources.append({
            "id": d.id,
            "title": d.title,
            "authors": d.authors,
            "venue": d.venue,
            "year": d.year,
            "doi": d.doi,
            "scopus_indexed": d.scopus_indexed,
            "blacklisted_authors": d.blacklisted_authors,
            "credible": d.is_credible,
            "reasons": d.credibility_reasons,
        })
    return pct, credible, total, sources
