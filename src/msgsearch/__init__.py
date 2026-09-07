"""Search your iMessage history by meaning, entirely on your own machine.

The pipeline, in the order data flows through it:

    extract   ->  chat.db rows become readable messages, decoding the
                  `attributedBody` blob that holds ~86% of them
    chunk     ->  messages become conversation windows, then the smaller
                  passages that actually get embedded
    index     ->  passages become vectors; windows become a keyword index
    search    ->  a query becomes ranked windows, via keyword search, vector
                  search, rank fusion and optional reranking

Nothing here sends message text anywhere. Both models run locally.
"""

__version__ = "0.2.5"

__all__ = ["__version__"]
