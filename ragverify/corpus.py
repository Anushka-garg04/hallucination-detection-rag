import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Document:
    id: str
    title: str
    authors: list
    venue: str
    year: int
    doi: str
    scopus_indexed: bool
    text: str
    blacklisted_authors: list = field(default_factory=list)

    @property
    def credibility_reasons(self):
        reasons = []
        if not self.scopus_indexed:
            reasons.append("Not Scopus-indexed")
        if self.blacklisted_authors:
            reasons.append("Blacklisted author: " + ", ".join(self.blacklisted_authors))
        return reasons

    @property
    def is_credible(self):
        return len(self.credibility_reasons) == 0


def load_blacklist(path: Path = DATA_DIR / "blacklist.csv") -> set:
    names = set()
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                names.add(row["author"].strip().lower())
    return names


def load_documents(path: Path = DATA_DIR / "corpus.json") -> list:
    blacklist = load_blacklist()
    raw = json.loads(path.read_text(encoding="utf-8"))["documents"]
    docs = []
    for d in raw:
        blacklisted = [a for a in d["authors"] if a.lower() in blacklist]
        docs.append(Document(
            id=d["id"], title=d["title"], authors=d["authors"], venue=d["venue"],
            year=d["year"], doi=d["doi"], scopus_indexed=d["scopus_indexed"],
            text=d["text"], blacklisted_authors=blacklisted,
        ))
    return docs
