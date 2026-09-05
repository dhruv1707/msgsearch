#!/usr/bin/env python3
"""The verification loop. Scores a retriever against eval/gold.jsonl.

Run this before every commit that touches retrieval. A retrieval change with no
movement here is an unverified guess, not an improvement.

    python3 eval/bench.py                          # default retriever
    python3 eval/bench.py -r search.lexical        # pick a retriever module
    python3 eval/bench.py --save baseline          # snapshot metrics
    python3 eval/bench.py --against baseline       # diff vs a snapshot

Metrics mirror the standard IR set so numbers are comparable to published work:
precision@10, recall@50, MRR, nDCG@10.
"""

import argparse
import importlib
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

GOLD = os.path.join(ROOT, "eval", "gold.jsonl")
RESULTS = os.path.join(ROOT, "eval", "results")


def load_gold():
    with open(GOLD) as f:
        return [json.loads(line) for line in f if line.strip()]


def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def score_one(ranked, relevant, k=10, recall_k=50):
    rel = set(relevant)
    if not rel:
        return None
    top_k = ranked[:k]
    p_at_k = sum(1 for m in top_k if m in rel) / k
    r_at_n = sum(1 for m in ranked[:recall_k] if m in rel) / len(rel)
    rr = next((1.0 / (i + 1) for i, m in enumerate(ranked) if m in rel), 0.0)
    ideal = dcg([1.0] * min(len(rel), k))
    ndcg = (dcg([1.0 if m in rel else 0.0 for m in top_k]) / ideal) if ideal else 0.0
    return {"p@10": p_at_k, "r@50": r_at_n, "mrr": rr, "ndcg@10": ndcg}


def run(retriever_path, k=50):
    mod = importlib.import_module(retriever_path)
    gold = load_gold()
    scored, unlabeled = [], []
    for case in gold:
        ranked = [mid for mid, _ in mod.search(case["query"], k)]
        s = score_one(ranked, case.get("relevant", []))
        if s is None:
            unlabeled.append(case["id"])
        else:
            scored.append((case["id"], case.get("type", "semantic"), s))
    return scored, unlabeled


METRICS = ["p@10", "r@50", "mrr", "ndcg@10"]


def aggregate(scored):
    if not scored:
        return {k: 0.0 for k in METRICS}
    return {k: sum(s[k] for _, _, s in scored) / len(scored) for k in METRICS}


def by_type(scored):
    """Per-type breakdown. Which *kind* of query fails is the actual roadmap:
    losing on `semantic` while winning on `exact` means the dense stage is the
    work, not the ranking."""
    groups = {}
    for _, qtype, s in scored:
        groups.setdefault(qtype, []).append((None, None, s))
    return {t: (len(g), aggregate(g)) for t, g in sorted(groups.items())}


def type_table(groups):
    head = f"{'type':<13} {'n':>3}  " + "  ".join(f"{m:>7}" for m in METRICS)
    lines = [head, "-" * len(head)]
    for t, (n, agg) in groups.items():
        lines.append(f"{t:<13} {n:>3}  " + "  ".join(f"{agg[m]:>7.4f}" for m in METRICS))
    return "\n".join(lines)


def table(agg, prev=None):
    lines = [f"{'metric':<10} {'value':>8}" + (f" {'delta':>9}" if prev else "")]
    lines.append("-" * len(lines[0]))
    for k in METRICS:
        v = agg[k]
        row = f"{k:<10} {v:>8.4f}"
        if prev:
            d = v - prev.get(k, 0.0)
            row += f" {d:>+9.4f}"
        lines.append(row)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--retriever", default="search.lexical")
    ap.add_argument("--save", metavar="NAME")
    ap.add_argument("--against", metavar="NAME")
    args = ap.parse_args()

    scored, unlabeled = run(args.retriever)
    agg = aggregate(scored)

    prev = None
    if args.against:
        p = os.path.join(RESULTS, args.against + ".json")
        if os.path.exists(p):
            prev = json.load(open(p))["metrics"]
        else:
            print(f"warn: no snapshot {args.against!r}", file=sys.stderr)

    print(f"retriever: {args.retriever}")
    print(f"labeled queries: {len(scored)}  unlabeled: {len(unlabeled)}\n")
    print(table(agg, prev))

    groups = by_type(scored)
    if len(groups) > 1:
        print()
        print(type_table(groups))

    if unlabeled:
        print(f"\nUNLABELED (not scored): {', '.join(unlabeled)}")
        print("Add relevant message ids to eval/gold.jsonl -- unlabeled queries "
              "measure nothing.")

    if args.save:
        os.makedirs(RESULTS, exist_ok=True)
        with open(os.path.join(RESULTS, args.save + ".json"), "w") as f:
            json.dump({"retriever": args.retriever, "metrics": agg,
                       "by_type": {t: a for t, (_, a) in by_type(scored).items()}},
                      f, indent=2)
        print(f"\nsaved snapshot: {args.save}")

    # Non-zero exit when nothing is measurable, so agents cannot claim success.
    return 1 if not scored else 0


if __name__ == "__main__":
    sys.exit(main())
