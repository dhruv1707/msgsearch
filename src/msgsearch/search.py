"""Hybrid search over the message index.

A query runs through four stages, and each exists because the stage before it
cannot do its job.

1. Keyword search (BM25 over FTS5) finds windows containing the literal words you
   typed. It is precise about names and identifiers and understands no meaning.
2. Vector search finds passages whose *meaning* is close to the query, which is
   how a request for a login reaches a message containing an email address and a
   password but neither of those words. It runs over three-message passages
   rather than whole windows, because a vector for thirty messages represents
   none of them well.
3. Fusion merges the two lists. Their scores are on unrelated scales and cannot be
   compared, so Reciprocal Rank Fusion ignores the scores and uses positions
   only. The purpose of this stage is recall: get the right answer somewhere into
   a shortlist of about fifty.
4. Reranking decides the final order. A cross-encoder reads the query and one
   passage together and scores the pair directly, which is far more accurate than
   comparing two independently-made vectors, and far too slow to run over the
   whole index. Running it over the shortlist is what makes it affordable.

Throughout, the passage is the unit that gets matched and the window is the unit
that gets shown.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config


@dataclass
class Result:
    window_row: int
    window_id: str
    chat_label: str
    start_ts: str
    end_ts: str
    speakers: list[str]
    tags: set[str]
    text: str
    offsets: list[list[int]] = field(default_factory=list)
    passage_text: str = ""
    matched_rowids: list[int] = field(default_factory=list)
    bm25_rank: int | None = None
    dense_rank: int | None = None
    fusion_score: float = 0.0
    rerank_score: float | None = None


def open_index(index_dir: Path | None = None) -> tuple[sqlite3.Connection, np.ndarray]:
    index_dir = Path(index_dir or config.INDEX_DIR).expanduser()
    db_path = index_dir / "index.db"
    vec_path = index_dir / "vectors.npy"
    if not db_path.exists() or not vec_path.exists():
        raise FileNotFoundError(
            f"No index in {index_dir}. Build one first with: python index.py"
        )
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True), np.load(vec_path)


def index_meta(db) -> dict[str, str]:
    return {key: value for key, value in db.execute("SELECT key, value FROM meta")}


def assert_model_matches(db, embedder) -> None:
    """Refuse to query an index built by a different embedding model.

    Two models' vectors are not comparable even when their dimensions agree, so
    this would otherwise degrade silently: the search returns confident-looking
    nonsense rather than failing. When the dimensions differ it crashes instead,
    which is better but still baffling. Neither is acceptable given how long a
    rebuild takes.
    """
    built_with = index_meta(db).get("embed_model")
    if built_with and built_with != embedder.model_name:
        raise SystemExit(
            f"Index/model mismatch.\n"
            f"  index was built with : {built_with}\n"
            f"  you are querying with: {embedder.model_name}\n\n"
            f"Vectors from different models are not comparable, so results would "
            f"be meaningless. Either point at the model the index was built with:\n\n"
            f"  MSGSEARCH_EMBED_MODEL={built_with}\n\n"
            f"or rebuild the index with the current model: python index.py"
        )


def _fts_expression(query: str) -> str:
    """Turn free text into an FTS5 expression.

    User input cannot be handed to FTS5 directly, because characters such as
    quotes and hyphens are operators there and would either raise an error or
    silently change the query. Each word is extracted and quoted, then combined
    with OR so a partial match still retrieves something.
    """
    terms = re.findall(r"\w+", query.lower())
    return " OR ".join(f'"{term}"' for term in terms)


def _filter_clause(args) -> tuple[str, list]:
    clauses, params = [], []
    if args.chat:
        clauses.append("w.chat_label LIKE ?")
        params.append(f"%{args.chat}%")
    if getattr(args, "from_", None):
        clauses.append("w.speakers LIKE ?")
        params.append(f"%{args.from_}%")
    if args.after:
        clauses.append("w.start_ts >= ?")
        params.append(args.after)
    if args.before:
        clauses.append("w.start_ts <= ?")
        params.append(args.before + "T23:59:59")
    if args.type:
        # Padded so the match is on a whole tag. A plain LIKE '%credential%'
        # would also match 'credential_talk', which is the opposite of what
        # someone hunting for an actual password wants.
        clauses.append("(' ' || w.tags || ' ') LIKE ?")
        params.append(f"% {args.type} %")
    return (" AND " + " AND ".join(clauses)) if clauses else "", params


def _allowed_windows(db, args) -> set[int] | None:
    where, params = _filter_clause(args)
    if not where:
        return None
    return {
        row[0]
        for row in db.execute(
            f"SELECT w.window_row FROM windows w WHERE 1=1 {where}", params
        )
    }


def bm25_candidates(db, query: str, args, limit: int) -> list[int]:
    expression = _fts_expression(query)
    if not expression:
        return []
    where, params = _filter_clause(args)
    sql = f"""
        SELECT f.rowid
        FROM windows_fts f
        JOIN windows w ON w.window_row = f.rowid
        WHERE windows_fts MATCH ? {where}
        ORDER BY bm25(windows_fts)
        LIMIT ?
    """
    return [row[0] for row in db.execute(sql, [expression, *params, limit])]


def dense_candidates(
    db, vectors, query_vector, args, limit: int
) -> tuple[list[int], dict[int, tuple[str, list[int]]]]:
    """Rank passages, then collapse them to the windows they came from.

    Several passages from the same conversation often score well together. Only
    the best one is kept per window, both so the shortlist holds distinct results
    and so the reranker is later shown the strongest slice of each candidate.
    """
    scores = vectors @ query_vector
    allowed = _allowed_windows(db, args)

    # Passages outnumber windows several times over, so scan deeper than the
    # window limit before collapsing.
    depth = min(len(scores), max(limit * 12, 200))
    top = np.argpartition(-scores, depth - 1)[:depth]
    top = top[np.argsort(-scores[top])]

    placeholders = ",".join("?" * len(top))
    mapping = {
        row[0]: (row[1], row[2], row[3])
        for row in db.execute(
            f"""SELECT vector_row, window_row, text, rowids
                FROM passages WHERE vector_row IN ({placeholders})""",
            [int(i) for i in top],
        )
    }

    ordered: list[int] = []
    best: dict[int, tuple[str, list[int]]] = {}
    for passage_row in top:
        entry = mapping.get(int(passage_row))
        if entry is None:
            continue
        window_row, text, rowids = entry
        if allowed is not None and window_row not in allowed:
            continue
        if window_row in best:
            continue
        best[window_row] = (text, json.loads(rowids))
        ordered.append(window_row)
        if len(ordered) >= limit:
            break

    return ordered, best


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]], k: int = config.RRF_K
) -> dict[int, float]:
    """Merge ranked lists using positions only.

    A window's contribution from one list is 1/(k + its rank). The constant k
    flattens the curve so the top result does not overwhelm everything beneath it,
    which is what lets agreement across two lists outweigh one strong placement.
    """
    scores: dict[int, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, row in enumerate(ranked, start=1):
            scores[row] += 1.0 / (k + rank)
    return scores


def load_windows(db, rows: list[int]) -> dict[int, Result]:
    if not rows:
        return {}
    placeholders = ",".join("?" * len(rows))
    sql = f"""SELECT window_row, window_id, chat_label, start_ts, end_ts,
                     speakers, tags, search_text, offsets
              FROM windows WHERE window_row IN ({placeholders})"""
    out = {}
    for row in db.execute(sql, rows):
        out[row[0]] = Result(
            window_row=row[0],
            window_id=row[1],
            chat_label=row[2],
            start_ts=row[3],
            end_ts=row[4],
            speakers=json.loads(row[5]),
            tags=set(row[6].split()) if row[6] else set(),
            text=row[7],
            offsets=json.loads(row[8]),
        )
    return out


def anchored_text(result: Result) -> str:
    """Return the window text starting at the message that matched.

    Showing the passage alone is not enough. A query like "what was the login"
    matches the message *asking* for it, while the answer arrives a moment later,
    so a view that stops at the match stops one message short of the thing you
    were looking for. Starting at the match and running forward shows both.
    """
    if not result.matched_rowids or not result.offsets:
        return result.passage_text or result.text

    positions = {row[0]: row[1] for row in result.offsets}
    starts = [positions[r] for r in result.matched_rowids if r in positions]
    if not starts:
        return result.passage_text or result.text

    line_start = result.text.rfind("\n", 0, min(starts)) + 1
    return result.text[line_start:]


def _fill_missing_passages(db, results: list[Result], query: str) -> None:
    """Give every candidate a passage to rerank on.

    Windows that arrived only through keyword search have no passage chosen for
    them yet, so pick the one containing the most query words. Reranking a short
    passage rather than a whole window matters for the same reason embedding one
    does: the signal is otherwise diluted.
    """
    terms = set(re.findall(r"\w+", query.lower()))
    missing = [r for r in results if not r.passage_text]
    if not missing:
        return

    placeholders = ",".join("?" * len(missing))
    rows = db.execute(
        f"""SELECT window_row, text, rowids FROM passages
            WHERE window_row IN ({placeholders})""",
        [r.window_row for r in missing],
    ).fetchall()

    by_window: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for window_row, text, rowids in rows:
        by_window[window_row].append((text, rowids))

    for result in missing:
        options = by_window.get(result.window_row)
        if not options:
            result.passage_text = result.text[:1500]
            continue
        text, rowids = max(
            options,
            key=lambda o: len(terms & set(re.findall(r"\w+", o[0].lower()))),
        )
        result.passage_text = text
        result.matched_rowids = json.loads(rowids)


def search(query: str, args, db=None, vectors=None, embedder=None, reranker=None):
    if db is None or vectors is None:
        db, vectors = open_index(args.index_dir)

    bm25 = [] if args.no_bm25 else bm25_candidates(db, query, args, config.BM25_CANDIDATES)

    dense: list[int] = []
    best_passages: dict[int, tuple[str, list[int]]] = {}
    if not args.no_dense:
        if embedder is None:
            from embedder import Embedder

            embedder = Embedder()
        assert_model_matches(db, embedder)
        dense, best_passages = dense_candidates(
            db, vectors, embedder.embed_query(query), args, config.DENSE_CANDIDATES
        )

    lists = [lst for lst in (bm25, dense) if lst]
    if not lists:
        return []

    fused = reciprocal_rank_fusion(lists)
    shortlist = sorted(fused, key=fused.get, reverse=True)[: config.RRF_CANDIDATES]

    results = load_windows(db, shortlist)
    bm25_pos = {row: i + 1 for i, row in enumerate(bm25)}
    dense_pos = {row: i + 1 for i, row in enumerate(dense)}
    for row, result in results.items():
        result.bm25_rank = bm25_pos.get(row)
        result.dense_rank = dense_pos.get(row)
        result.fusion_score = fused[row]
        if row in best_passages:
            result.passage_text, result.matched_rowids = best_passages[row]

    ordered = [results[r] for r in shortlist if r in results]

    # Done regardless of reranking: windows found only by keyword search still
    # need a passage chosen so the display can show what matched.
    if ordered:
        _fill_missing_passages(db, ordered, query)

    if config.RERANK_ENABLED and not args.no_rerank and ordered:
        if reranker is None:
            from embedder import Reranker

            reranker = Reranker()
        scores = reranker.score(query, [r.passage_text for r in ordered])
        for result, score in zip(ordered, scores):
            result.rerank_score = float(score)
        ordered.sort(key=lambda r: r.rerank_score, reverse=True)

    return ordered[: args.limit]


def format_result(
    index: int, result: Result, snippet_chars: int, full: bool = False
) -> str:
    day = result.start_ts[:10]
    start_time = result.start_ts[11:16]
    end_time = result.end_ts[11:16]
    who = ", ".join(result.speakers)

    ranks = []
    if result.bm25_rank:
        ranks.append(f"bm25 #{result.bm25_rank}")
    if result.dense_rank:
        ranks.append(f"vector #{result.dense_rank}")
    if result.rerank_score is not None:
        ranks.append(f"rerank {result.rerank_score:+.2f}")
    tags = f"  [{' '.join(sorted(result.tags))}]" if result.tags else ""

    # Start at the message that matched and run forward. Showing the head of the
    # window instead gives the reader whatever the conversation happened to open
    # with, which is usually unrelated to why the window was retrieved.
    text = result.text if full else anchored_text(result)
    if len(text) > snippet_chars:
        text = text[:snippet_chars].rstrip() + "\n  ..."
    body = "\n".join("  " + line for line in text.splitlines())

    return (
        f"\n{index}. {day} {start_time}-{end_time}  {result.chat_label}  "
        f"({who}){tags}\n"
        f"   {' · '.join(ranks)}\n{body}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Search your messages by meaning.")
    parser.add_argument("query", nargs="+", help="what you are looking for")
    parser.add_argument("--limit", type=int, default=config.DEFAULT_LIMIT)
    parser.add_argument("--chat", help="restrict to conversations matching this text")
    parser.add_argument("--from", dest="from_", help="restrict to a speaker")
    parser.add_argument("--after", help="YYYY-MM-DD")
    parser.add_argument("--before", help="YYYY-MM-DD")
    parser.add_argument("--type", help="shape tag: credential, email, phone, url, address")
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--snippet", type=int, default=1200, help="max characters shown")
    parser.add_argument(
        "--full", action="store_true", help="show the whole window, not just the match"
    )
    parser.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder")
    parser.add_argument("--no-dense", action="store_true", help="keyword search only")
    parser.add_argument("--no-bm25", action="store_true", help="vector search only")
    args = parser.parse_args()

    query = " ".join(args.query)
    try:
        results = search(query, args)
    except FileNotFoundError as error:
        sys.exit(str(error))

    if not results:
        print("no results")
        return

    print(f"{len(results)} result(s) for {query!r}")
    for i, result in enumerate(results, start=1):
        print(format_result(i, result, args.snippet, full=args.full))


if __name__ == "__main__":
    main()
