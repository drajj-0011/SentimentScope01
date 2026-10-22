"""Deterministically build ``data/sentiment.csv``.

Design goals
------------
* **Learnable.** Each class owns a distinct sentiment lexicon, so a linear model
  can separate the classes instead of guessing.
* **Not trivially separable.** One third of the rows are *contrastive* — a
  positive and a negative clause joined by "but", where the same clause
  vocabulary appears under both labels. The documented labelling rule is
  **the concluding clause decides**, so real overlap exists and no classifier
  can reach a suspiciously perfect score by memorising a template.
* **Reproducible.** A fixed seed, so ``git clone`` gives byte-identical data.

Usage:
    python -m scripts.build_dataset

Author: Debraj
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "sentiment.csv"
PER_CLASS_PLAIN = 140
PER_CLASS_CONTRASTIVE = 60

SUBJECTS = [
    "this product", "the app", "my order", "the laptop", "the support team",
    "the restaurant", "the latest update", "the headphones", "the website",
    "the service", "the camera", "the software", "the phone", "the delivery",
    "the booking process", "the customer portal", "the new feature", "the speaker",
    "the smartwatch", "the online shop", "the charger", "the dashboard",
    "the mobile version", "the subscription",
]

POSITIVE = [
    "is absolutely fantastic in every way",
    "works flawlessly and feels premium",
    "exceeded my expectations completely",
    "is superb value for the money",
    "has been an absolute delight to use",
    "is outstanding and beautifully built",
    "is wonderfully fast and responsive",
    "is remarkably polished and reliable",
    "is genuinely brilliant and worth every penny",
    "left me thoroughly impressed and happy",
    "is a joy to use and looks gorgeous",
    "performs brilliantly under heavy load",
    "is the best purchase i have made this year",
    "deserves a glowing recommendation from me",
    "is excellent, sturdy and thoughtfully designed",
    "came perfectly packed and works like a dream",
    "is flawless, intuitive and a pleasure to own",
    "is a triumph of good engineering",
]
POSITIVE_TAIL = [
    "and i would happily buy it again",
    "so i recommend it to everyone",
    "and the staff were lovely about it",
    "which made my whole week better",
    "and setup took about two minutes",
    "so i am thrilled with the result",
    "and it saved me hours of work",
    "which is exactly what i hoped for",
]

NEGATIVE = [
    "is a complete waste of money",
    "broke after two days of light use",
    "crashes constantly and loses my data",
    "is painfully slow and laggy",
    "is appalling quality for the price",
    "never works when i actually need it",
    "is overpriced garbage and clearly untested",
    "arrived scratched, dirty and damaged",
    "keeps freezing whenever i open it",
    "is a total letdown and deeply frustrating",
    "refuses to charge after a single month",
    "is unusable and riddled with bugs",
    "was a horrible experience from start to finish",
    "overheats and shuts itself down randomly",
    "is the worst thing i have bought in years",
    "failed on the very first day",
    "is buggy, flimsy and infuriating to use",
    "is a miserable disappointment overall",
]
NEGATIVE_TAIL = [
    "and support never replied to my emails",
    "so i demanded a refund immediately",
    "and i regret buying it",
    "which ruined my evening entirely",
    "and the return process is a joke",
    "so i would not recommend it to anyone",
    "and nobody has apologised since",
    "which left me extremely annoyed",
]

NEUTRAL = [
    "arrived on tuesday morning",
    "comes in three colours",
    "costs about forty pounds",
    "ships from a warehouse in leeds",
    "has a two year warranty included",
    "was released last month",
    "weighs roughly two kilograms",
    "measures six inches across",
    "runs on windows and linux",
    "supports bluetooth and wifi",
    "contains a charger and a cable",
    "opens at nine in the morning",
    "includes a printed manual",
    "lists four available models",
    "renews every twelve months",
    "was ordered online last friday",
    "sits on the second floor",
    "requires an account to sign in",
]
NEUTRAL_TAIL = [
    "according to the catalogue",
    "as stated on the invoice",
    "per the release notes",
    "which the manual confirms",
    "according to the specification sheet",
    "as listed in the email",
    "per the delivery note",
    "which the website mentions",
]

CONNECTORS = ["", "honestly, ", "frankly, ", "overall, ", "in short, ", "for me, ", "so far, "]

# Short clauses reused by BOTH polarities in the contrastive set. This is what
# stops the task from collapsing into a template lookup.
POS_CLAUSE = [
    "the design is gorgeous",
    "delivery was quick",
    "the price is fair",
    "it looks premium on my desk",
    "setup took two minutes",
    "the packaging was lovely",
    "the app feels smooth",
    "the staff were polite",
]
NEG_CLAUSE = [
    "it broke after two days",
    "support never replied",
    "it drains in under an hour",
    "the app keeps crashing",
    "the screen arrived scratched",
    "the manual made no sense",
    "it overheats constantly",
    "the invoice was wrong",
]


def _unique(pool: list[str], seen: set[str], limit: int, label: str) -> list[tuple[str, str]]:
    """Take ``limit`` unique sentences from ``pool`` in order."""
    out: list[tuple[str, str]] = []
    for sentence in pool:
        key = sentence.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append((sentence, label))
        if len(out) == limit:
            break
    if len(out) < limit:  # pragma: no cover - guard against silent shrinkage
        raise RuntimeError(f"only generated {len(out)} unique rows for {label}")
    return out


def build() -> list[tuple[str, str]]:
    rng = random.Random(2026)
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()

    plain = (
        ("positive", POSITIVE, POSITIVE_TAIL),
        ("negative", NEGATIVE, NEGATIVE_TAIL),
        ("neutral", NEUTRAL, NEUTRAL_TAIL),
    )

    for label, heads, tails in plain:
        pool = [
            f"{conn}{subject} {head}, {tail}." if conn else f"{subject} {head} and {tail}."
            for conn in CONNECTORS
            for subject in SUBJECTS
            for head in heads
            for tail in tails
        ]
        rng.shuffle(pool)
        rows.extend(_unique(pool, seen, PER_CLASS_PLAIN, label))

    # Contrastive rows: "<opposite clause>, but <deciding clause>." -> deciding label.
    for label, deciding, opposing in (
        ("positive", POS_CLAUSE, NEG_CLAUSE),
        ("negative", NEG_CLAUSE, POS_CLAUSE),
    ):
        pool = [
            f"{first}, but {second}."
            for first in opposing
            for second in deciding
            if first != second
        ]
        pool += [
            f"{conn}{first}, but {second}."
            for conn in ("", "admittedly, ", "to be fair, ")
            for first in opposing
            for second in deciding
        ]
        rng.shuffle(pool)
        rows.extend(_unique(pool, seen, PER_CLASS_CONTRASTIVE, label))

    rng.shuffle(rows)
    return rows


def main() -> None:
    rows = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for _, label in rows:
        counts[label] = counts.get(label, 0) + 1

    print(f"wrote {len(rows)} rows to {OUT}")
    for label, count in sorted(counts.items()):
        print(f"  {label:<9}: {count}")


if __name__ == "__main__":
    main()
