#!/usr/bin/env python3
"""Build the gold set by pooling retriever candidates and labeling them by hand.

Standard TREC-style pooling: union the top-N from every retriever, show them once,
mark which are relevant. Judging the pool (rather than the whole corpus) is what
makes labeling 271k messages tractable.

    python3 eval/label.py "the wifi password someone sent me"
    python3 eval/label.py --relabel q1

Prints message bodies to YOUR terminal only. Nothing is written to the repo except
message ids -- gold.jsonl never contains message text.
"""

import argparse
import json
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

GOLD = os.path.join(ROOT, "eval", "gold.jsonl")
INDEX = os.environ.get("MSGSEARCH_INDEX", os.path.join(ROOT, "index.sqlite"))

RETRIEVERS = ["search.lexical"]  # add dense/hybrid here as they land


def pool(query, per_retriever=15):
    import importlib
    seen, ordered = set(), []
    for path in RETRIEVERS:
        try:
            mod = importlib.import_module(path)
        except ImportError:
            continue
        for mid, _ in mod.search(query, per_retriever):
            if mid not in seen:
                seen.add(mid)
                ordered.append(mid)
    return ordered


def bodies(ids):
    db = sqlite3.connect(f"file:{INDEX}?mode=ro", uri=True)
    q = ",".join("?" * len(ids))
    rows = db.execute(
        f"SELECT id, ts, chat_label, is_from_me, body FROM messages WHERE id IN ({q})",
        ids).fetchall()
    db.close()
    return {r[0]: r for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--relabel", metavar="ID")
    ap.add_argument("--id", help="gold id to assign (default: next qN)")
    args = ap.parse_args()

    gold = []
    if os.path.exists(GOLD):
        with open(GOLD) as f:
            gold = [json.loads(l) for l in f if l.strip()]

    if args.relabel:
        case = next((c for c in gold if c["id"] == args.relabel), None)
        if not case:
            sys.exit(f"no gold case {args.relabel!r}")
        query, gid = case["query"], case["id"]
    else:
        if not args.query:
            sys.exit("give a query, or --relabel <id>")
        query = args.query
        gid = args.id or f"q{len(gold) + 1}"

    ids = pool(query)
    if not ids:
        sys.exit("no candidates -- is index.sqlite built? (python3 search/index.py)")
    meta = bodies(ids)

    print(f"\nquery: {query!r}   ({len(ids)} pooled candidates)\n")
    for n, mid in enumerate(ids, 1):
        row = meta.get(mid)
        if not row:
            continue
        _, ts, chat, from_me, body = row
        who = "me" if from_me else (chat or "?")
        snippet = body.replace("\n", " ")[:110]
        print(f"[{n:>2}] {mid:<8} {(ts or '')[:10]}  {who[:18]:<18}  {snippet}")

    print("\nEnter the numbers that actually answer the query (e.g. 1 4 7), or blank for none.")
    picks = input("> ").split()
    relevant = [ids[int(p) - 1] for p in picks if p.isdigit() and 1 <= int(p) <= len(ids)]

    case = {"id": gid, "query": query, "relevant": relevant}
    gold = [c for c in gold if c["id"] != gid] + [case]
    gold.sort(key=lambda c: (len(c["id"]), c["id"]))
    with open(GOLD, "w") as f:
        for c in gold:
            f.write(json.dumps(c) + "\n")
    print(f"\nsaved {gid}: {len(relevant)} relevant -> eval/gold.jsonl")


if __name__ == "__main__":
    main()
