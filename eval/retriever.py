"""Adapter that lets the eval call the real search pipeline.

`bench.py` and `label.py` speak one contract:

    search(query, k) -> [(id, score)]   # best first

`search.py` speaks a richer one built for a CLI. This module bridges the two,
and settles the unit: the pipeline matches passages but *returns windows*, and a
window is what a person is shown, so a window is what gets judged. Gold labels
are therefore window ids.

Variants score parts of the pipeline separately, which is how you find out
whether a stage earns its place:

    -r eval.retriever           full pipeline
    -r eval.retriever_bm25      keyword only
    -r eval.retriever_dense     vectors only
    -r eval.retriever_norerank  fusion without the cross-encoder
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_state = {}


def _args(limit, **overrides):
    a = argparse.Namespace(
        limit=limit, chat=None, from_=None, after=None, before=None, type=None,
        index_dir=None, snippet=1200, full=False,
        no_rerank=False, no_dense=False, no_bm25=False,
    )
    for key, value in overrides.items():
        setattr(a, key, value)
    return a


def _pipeline():
    """Load index, vectors and the embedding model once, then reuse them."""
    if not _state:
        import search as S
        from embedder import Embedder
        db, vectors = S.open_index()
        _state.update(S=S, db=db, vectors=vectors, embedder=Embedder())
    return _state


def search_detailed(query, k=50, **overrides):
    """Like search(), but also returns the window text starting at the message
    that matched. The head of a 95-message window says nothing about why it was
    retrieved, so the labeler shows this instead."""
    p = _pipeline()
    results = p["S"].search(
        query, _args(k, **overrides),
        db=p["db"], vectors=p["vectors"], embedder=p["embedder"],
    )
    return [(r.window_id, p["S"].anchored_text(r)) for r in results]


def search(query, k=50, **overrides):
    p = _pipeline()
    results = p["S"].search(
        query, _args(k, **overrides),
        db=p["db"], vectors=p["vectors"], embedder=p["embedder"],
    )
    # Position is the real signal; the score is for display and some stages
    # leave it unset, so fall back to descending rank.
    out = []
    for position, r in enumerate(results):
        score = r.rerank_score if r.rerank_score is not None else r.fusion_score
        out.append((r.window_id, float(score if score is not None else -position)))
    return out
