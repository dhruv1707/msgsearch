"""Local embedding and reranking models.

Nothing in this module sends text anywhere. Both models are downloaded once and
then run on your own machine, which matters here because the corpus being indexed
contains passwords, addresses and private conversations.

One detail is easy to get wrong and expensive to miss. EmbeddingGemma was trained
with task-specific prompt prefixes: a search query is presented to the model
differently from a document being searched. That asymmetry is what teaches the
model to place a question near its *answer* rather than near other questions,
which is exactly what this project needs, because the message you are looking for
usually shares no words with the query. Calling the generic `encode()` skips the
prefixes and quietly degrades results, so this module always goes through
`encode_query()` and `encode_document()`, which attach the correct prefix.
"""

from __future__ import annotations

import numpy as np

import config


def _load_failure_hint(model_name: str, error: Exception) -> str:
    """Turn an opaque model-download failure into something actionable.

    A gated model fails with a bare 401 that says "please log in" even when a
    token is present but expired, which is a confusing place to land.
    """
    text = str(error)
    gated = "gated" in text.lower() or "401" in text
    if not gated:
        return f"Could not load {model_name}: {error}"

    return (
        f"Could not download {model_name} — it is a gated model and this machine "
        f"is not authorised.\n\n"
        f"  1. Accept the licence at https://huggingface.co/{model_name}\n"
        f"  2. Create a read token at https://huggingface.co/settings/tokens\n"
        f"  3. Run: hf auth login\n\n"
        f"Note that an *expired* token produces this same error, so re-run step 3 "
        f"even if you have logged in before. To use an ungated model instead:\n\n"
        f"  MSGSEARCH_EMBED_MODEL=BAAI/bge-small-en-v1.5\n\n"
        f"Original error: {error}"
    )


def select_device() -> str:
    """Prefer Apple's GPU, then CUDA, then fall back to the CPU."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Embedder:
    """Turns text into vectors, with the right prompt prefix for each role."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or config.EMBED_MODEL
        self.device = device or select_device()
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as error:
            raise RuntimeError(_load_failure_hint(self.model_name, error)) from error

    @property
    def dimension(self) -> int:
        # Renamed in sentence-transformers 6; the old name still works but warns.
        getter = getattr(self.model, "get_embedding_dimension", None)
        return getter() if getter else self.model.get_sentence_embedding_dimension()

    def embed_documents(
        self, texts: list[str], batch_size: int = 32, show_progress: bool = False
    ) -> np.ndarray:
        """Embed windows for storage in the index.

        Vectors are normalised to unit length, which makes a dot product identical
        to cosine similarity and lets search be a single matrix multiply.
        """
        return self.model.encode_document(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        ).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a search query. Note this is a different prefix from documents."""
        return self.model.encode_query(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)


class Reranker:
    """Scores a query against candidate windows by reading them together.

    An embedding model turns the query and a window into two separate vectors and
    compares them, so it never actually looks at the pair. A cross-encoder reads
    both at once and returns a single relevance score, which is markedly more
    accurate and markedly slower. It is therefore only ever run over the short
    candidate list that fusion produces, never over the whole index.
    """

    def __init__(self, model_name: str | None = None, device: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name or config.RERANK_MODEL
        self.device = device or select_device()
        self.model = CrossEncoder(self.model_name, device=self.device)

    def score(self, query: str, texts: list[str], batch_size: int = 16) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)
        pairs = [(query, text) for text in texts]
        return np.asarray(
            self.model.predict(pairs, batch_size=batch_size), dtype=np.float32
        )
