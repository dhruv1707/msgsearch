#!/usr/bin/env python3
"""Build the gold set by pooling retriever candidates and labeling them by hand.

Standard TREC-style pooling: union the top-N from every retriever, show them once,
mark which are relevant. Judging the pool (rather than the whole corpus) is what
makes labeling 271k messages tractable.

    python3 eval/label.py "the wifi password someone sent me"
    python3 eval/label.py --relabel q1

Prints message bodies to YOUR terminal only. Nothing is written to the repo except
window ids -- gold.jsonl never contains message text.
"""

import argparse
import json
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

GOLD = os.path.join(ROOT, "eval", "gold.jsonl")
INDEX = os.path.join(
    os.path.expanduser(os.environ.get("MSGSEARCH_INDEX", "~/msgsearch/index")), "index.db")

# Pool from the two *retrieval* stages, deliberately not from the full reranked
# pipeline: pooling from the thing you are about to score biases the gold set
# toward whatever the current ranking already prefers. Diverse pools are the
# whole point of TREC-style pooling.
RETRIEVERS = ["eval.retriever_bm25", "eval.retriever_dense"]

# qmd's taxonomy, adapted to messages. Purely for grouping -- it does not change
# search behaviour, it tells you *which kind* of query a retriever fails on.
TYPES = {
    "1": ("exact", "you remember the actual words (a name, number, phrase)"),
    "2": ("semantic", "you remember the meaning, not the words"),
    "3": ("topical", "a broad subject; many messages are valid answers"),
    "4": ("cross-domain", "the answer uses different vocabulary than the query"),
    "5": ("alias", "a person or thing referred to by another name/nickname"),
}


def pool(query, per_retriever=15):
    """Union the top hits of each retrieval stage, keeping the text that matched."""
    from eval import retriever
    stages = [("bm25", dict(no_dense=True, no_rerank=True)),
              ("dense", dict(no_bm25=True, no_rerank=True))]
    seen, ordered, matched = set(), [], {}
    for _, flags in stages:
        for wid, text in retriever.search_detailed(query, per_retriever, **flags):
            if wid not in seen:
                seen.add(wid)
                ordered.append(wid)
                matched[wid] = text
    return ordered, matched


def bodies(ids):
    """Window summaries for the pooled candidates, keyed by window_id."""
    db = sqlite3.connect(f"file:{INDEX}?mode=ro", uri=True)
    q = ",".join("?" * len(ids))
    rows = db.execute(
        f"""SELECT window_id, start_ts, chat_label, n_messages, tags, search_text
            FROM windows WHERE window_id IN ({q})""", ids).fetchall()
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

    ids, matched = pool(query)
    if not ids:
        sys.exit("no candidates -- is index.sqlite built? (python3 search/index.py)")
    meta = bodies(ids)

    print(f"\nquery: {query!r}   ({len(ids)} pooled candidates)\n")
    for n, wid in enumerate(ids, 1):
        row = meta.get(wid)
        if not row:
            continue
        _, ts, chat, n_msgs, tags, _full = row
        tag = f"[{tags}]" if tags else ""
        # Show the text from the matched message onward, not the window's head.
        snippet = " ".join(matched.get(wid, "").split())[:170]
        print(f"[{n:>2}] {wid:<14} {(ts or '')[:10]}  {n_msgs:>3}msg {tag}")
        print(f"     {snippet}")

    print("\nEnter the numbers whose conversation actually answers the query "
          "(e.g. 1 4 7), or blank for none.")
    print("Type ?N (e.g. ?7) to read a whole conversation first.")
    while True:
        raw = input("> ").strip()
        if raw.startswith("?"):
            want = raw[1:].strip()
            if want.isdigit() and 1 <= int(want) <= len(ids):
                wid = ids[int(want) - 1]
                print(f"\n----- [{want}] {wid} -----")
                print(meta[wid][5])
                print("-" * (len(wid) + 16) + "\n")
            else:
                print(f"no candidate {want!r}")
            continue
        picks = raw.split()
        break
    relevant = [ids[int(p) - 1] for p in picks if p.isdigit() and 1 <= int(p) <= len(ids)]

    print("\nWhat kind of query is this?")
    for key, (name, desc) in TYPES.items():
        print(f"  {key}. {name:<13} {desc}")
    choice = input("> [2] ").strip() or "2"
    qtype = TYPES.get(choice, ("semantic", ""))[0]
    if choice not in TYPES and choice in {n for n, _ in TYPES.values()}:
        qtype = choice  # allow typing the name directly

    case = {"id": gid, "query": query, "type": qtype, "relevant": relevant}
    gold = [c for c in gold if c["id"] != gid] + [case]
    gold.sort(key=lambda c: (len(c["id"]), c["id"]))
    with open(GOLD, "w") as f:
        for c in gold:
            f.write(json.dumps(c) + "\n")
    print(f"\nsaved {gid} [{qtype}]: {len(relevant)} relevant -> eval/gold.jsonl")


if __name__ == "__main__":
    main()
