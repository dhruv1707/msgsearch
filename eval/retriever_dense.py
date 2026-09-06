"""Vector search only. See eval/retriever.py."""

from eval.retriever import search as _search


def search(query, k=50):
    return _search(query, k, no_bm25=True, no_rerank=True)
