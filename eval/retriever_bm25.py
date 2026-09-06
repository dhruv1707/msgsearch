"""Keyword (BM25) only. See eval/retriever.py."""

from eval.retriever import search as _search


def search(query, k=50):
    return _search(query, k, no_dense=True, no_rerank=True)
