"""Embedding backends for the L1 semantic cache (contract §7 stage 8).

One interface, two implementations, chosen once at startup by `get_embedder()`:

  SentenceTransformerEmbedder   all-MiniLM-L6-v2, 384-d. Preferred: it is trained for
                                sentence similarity, so "screen wont respond" and
                                "touch input is delayed" land near each other even with
                                no shared vocabulary.
  TfidfSvdEmbedder              TF-IDF + TruncatedSVD over the seeded corpus. The offline
                                fallback when the model cannot be downloaded. It is
                                LEXICAL underneath, so it recognises rewording far less
                                well; the report always names which one produced its
                                numbers because they are not comparable.

Whichever runs, the store is a dense numpy matrix and similarity is one matmul over at
most a few thousand rows. No FAISS: exhaustive cosine at this scale is microseconds, and
an ANN index would be a dependency and a Docker layer bought for nothing (Phase 0 §7).
"""
from __future__ import annotations

import abc
import logging
import os
from typing import List, Optional, Sequence

import numpy as np

from .. import config

log = logging.getLogger("prism.embeddings")


class Embedder(abc.ABC):
    """Returns L2-NORMALISED row vectors, so cosine similarity is a plain dot product."""

    name: str = "base"
    dim: int = 0

    #: Does a vector depend on what else is in the corpus?
    #: False for a transformer: encoding one string gives the same vector whatever else is
    #: stored, so new keys can be APPENDED to the cache matrix. True for TF-IDF, whose
    #: vocabulary and IDF weights ARE the corpus, so adding a key invalidates every
    #: existing row and the matrix has to be rebuilt.
    corpus_dependent: bool = False

    @abc.abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        ...

    @staticmethod
    def _normalise(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # A zero vector has no direction; leave it zero so it can never be similar to
        # anything rather than dividing by zero into NaN.
        norms[norms == 0.0] = 1.0
        return matrix / norms


class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model = SentenceTransformer(model_name or config.CACHE_EMBED_MODEL)
        # Renamed in sentence-transformers 6.x; support both rather than pin a version.
        getter = (getattr(self._model, "get_embedding_dimension", None)
                  or self._model.get_sentence_embedding_dimension)
        self.dim = int(getter())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(list(texts), convert_to_numpy=True,
                                     show_progress_bar=False, normalize_embeddings=False)
        return self._normalise(vectors)


class TfidfSvdEmbedder(Embedder):  # noqa: D101 - see below
    """TF-IDF + SVD. Fitted lazily on the first corpus it is given and refitted when the
    corpus has grown enough to change the vocabulary materially.

    This is a fallback, and its weakness is worth stating plainly: it can only relate two
    phrasings that share vocabulary (directly, or through co-occurrence captured by the
    SVD). A paraphrase with genuinely disjoint wording scores near zero.
    """

    name = "tfidf+svd"
    corpus_dependent = True   # IDF weights shift as the corpus grows; rows cannot be reused

    def __init__(self, n_components: int | None = None):
        from sklearn.decomposition import TruncatedSVD  # type: ignore
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

        self._TfidfVectorizer = TfidfVectorizer
        self._TruncatedSVD = TruncatedSVD
        self._target_components = n_components or config.CACHE_TFIDF_COMPONENTS
        self._vectorizer = None
        self._svd = None
        self.dim = 0

    def fit(self, corpus: Sequence[str]) -> None:
        corpus = [c for c in corpus if c and c.strip()]
        if not corpus:
            return
        vectorizer = self._TfidfVectorizer(
            sublinear_tf=True, analyzer="word", ngram_range=(1, 2), min_df=1,
        )
        matrix = vectorizer.fit_transform(corpus)
        # SVD needs strictly fewer components than features and than samples.
        n_components = max(1, min(self._target_components,
                                  matrix.shape[1] - 1, matrix.shape[0] - 1))
        if n_components < 1 or matrix.shape[0] < 2:
            self._vectorizer, self._svd, self.dim = vectorizer, None, matrix.shape[1]
            return
        svd = self._TruncatedSVD(n_components=n_components, random_state=0)
        svd.fit(matrix)
        self._vectorizer, self._svd = vectorizer, svd
        self.dim = n_components

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, max(self.dim, 1)), dtype=np.float32)
        if self._vectorizer is None:
            self.fit(list(texts))
        if self._vectorizer is None:  # still nothing usable
            return np.zeros((len(texts), 1), dtype=np.float32)
        matrix = self._vectorizer.transform(list(texts))
        dense = self._svd.transform(matrix) if self._svd is not None else matrix.toarray()
        return self._normalise(dense)

    @property
    def is_fitted(self) -> bool:
        return self._vectorizer is not None


_ACTIVE: Optional[Embedder] = None


def get_embedder(force: str | None = None) -> Embedder:
    """Pick the backend once. `force` is for tests and for the eval harness.

    Selection order is deliberate: the better model first, the offline one only if it is
    genuinely unavailable. A silent downgrade would make two runs' numbers look
    comparable when they are not, so the choice is logged and recorded in the report.
    """
    global _ACTIVE
    if force is None and _ACTIVE is not None:
        return _ACTIVE

    choice = force or os.getenv("PRISM_CACHE_EMBEDDER", "auto")
    errors: List[str] = []

    if choice in ("auto", "sentence-transformers"):
        try:
            embedder: Embedder = SentenceTransformerEmbedder()
            log.info("cache embedder: %s (%d-d)", embedder.name, embedder.dim)
            if force is None:
                _ACTIVE = embedder
            return embedder
        except Exception as exc:  # not installed, or the model cannot be downloaded
            errors.append(f"sentence-transformers unavailable: {exc}")
            if choice != "auto":
                raise

    if choice in ("auto", "tfidf"):
        try:
            embedder = TfidfSvdEmbedder()
            log.warning("cache embedder falling back to %s (%s)", embedder.name,
                        "; ".join(errors) or "requested")
            if force is None:
                _ACTIVE = embedder
            return embedder
        except Exception as exc:
            errors.append(f"tfidf unavailable: {exc}")

    raise RuntimeError("no embedding backend available: " + "; ".join(errors))


def reset_embedder() -> None:
    """Drop the process-wide choice. Tests only."""
    global _ACTIVE
    _ACTIVE = None
