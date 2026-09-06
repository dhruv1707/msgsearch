"""BM25 + dense + RRF, no cross-encoder. See eval/retriever.py."""

from eval.retriever import search as _search


def search(query, k=50):
    return _search(query, k, no_rerank=True)
