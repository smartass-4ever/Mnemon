"""
Mnemon Embedder — auto-upgrading embedding backend.

Priority order (automatic, zero config):
  1. sentence-transformers  — 384-dim, ~85% retrieval precision (pip install mnemon-ai[full])
  2. OpenAI embeddings       — 1536-dim, ~90% retrieval precision (requires OPENAI_API_KEY)
  3. HashProjectionEmbedder — 64-dim fallback, System 1 cache only (always available)

The best available backend is selected on first embed() call.
No code changes needed when you upgrade.
"""

import hashlib
import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_ST_MODEL_CACHE = None  # loads only once per process


def _try_load_sentence_transformers():
    global _ST_MODEL_CACHE
    if _ST_MODEL_CACHE is not None:
        return _ST_MODEL_CACHE
    try:
        import os, warnings
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2", tokenizer_kwargs={"clean_up_tokenization_spaces": True})
        _ST_MODEL_CACHE = model
        logger.info("Mnemon embedder: sentence-transformers loaded (all-MiniLM-L6-v2, 384-dim)")
        return model
    except ImportError:
        return None
    except Exception as e:
        logger.warning(f"sentence-transformers found but failed to load: {e}")
        return None


def _try_load_openai_embedder() -> Optional["OpenAIEmbedder"]:
    """Return an OpenAIEmbedder if openai is installed and OPENAI_API_KEY is set."""
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        import openai  # noqa: F401 — just checking availability
        return OpenAIEmbedder()
    except ImportError:
        return None
    except Exception as e:
        logger.debug(f"Mnemon: OpenAI embedder unavailable — {e}")
        return None


class HashProjectionEmbedder:
    """
    Lightweight hash-projection fallback. 64-dim activation, 384-dim full.
    Always available — zero dependencies beyond numpy.
    Retrieval precision: ~56% on eval suite — sufficient for System 1 (exact cache).
    System 2 semantic recall requires a real embedder (see upgrade paths below).
    """
    DIM_ACTIVATION = 64
    DIM_FULL       = 384

    def embed(self, text: str) -> List[float]:
        tokens = text.lower().split()
        vec = np.zeros(self.DIM_ACTIVATION)
        for token in tokens:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = h % self.DIM_ACTIVATION
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_full(self, text: str) -> List[float]:
        tokens = text.lower().split()
        vec = np.zeros(self.DIM_FULL)
        for token in tokens:
            h = int(hashlib.sha256(token.encode()).hexdigest(), 16)
            idx = h % self.DIM_FULL
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


class SentenceTransformerEmbedder:
    """
    Production embedder using sentence-transformers.
    384-dim real semantic embeddings. ~85%+ retrieval precision.
    Install: pip install mnemon-ai[full]
    """
    DIM_ACTIVATION = 384
    DIM_FULL       = 384

    def __init__(self, model):
        self._model = model

    def embed(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self.DIM_ACTIVATION
        vec = self._model.encode(text[:512], normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        truncated = [t[:512] if t and t.strip() else "" for t in texts]
        vecs = self._model.encode(truncated, normalize_embeddings=True, batch_size=256)
        return vecs.tolist()

    def embed_full(self, text: str) -> List[float]:
        return self.embed(text)


class OpenAIEmbedder:
    """
    OpenAI text-embedding-3-small backend.
    1536-dim, ~90% retrieval precision. Requires OPENAI_API_KEY.
    Embeddings are cached in-memory to minimise API calls.
    Cost: ~$0.02 per million tokens — negligible for agent workloads.
    """
    DIM_ACTIVATION = 1536
    DIM_FULL       = 1536
    MODEL          = "text-embedding-3-small"

    def __init__(self):
        self._cache: dict = {}  # text → embedding, in-process cache

    def _call(self, text: str) -> List[float]:
        key = text[:500]
        if key in self._cache:
            return self._cache[key]
        from openai import OpenAI
        client = OpenAI()
        response = client.embeddings.create(model=self.MODEL, input=text[:8000])
        vec = response.data[0].embedding
        self._cache[key] = vec
        return vec

    def embed(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self.DIM_ACTIVATION
        try:
            return self._call(text)
        except Exception as e:
            logger.warning(f"Mnemon: OpenAI embed failed — {e}")
            return [0.0] * self.DIM_ACTIVATION

    def embed_full(self, text: str) -> List[float]:
        return self.embed(text)


class SimpleEmbedder:
    """
    Public embedder interface. Auto-selects best available backend on first use.

    Priority (automatic, zero config):
      1. sentence-transformers — offline, best quality (pip install mnemon-ai[full])
      2. OpenAI embeddings     — requires OPENAI_API_KEY, excellent quality
      3. hash-projection       — always works; System 1 cache only, no System 2

    System 2 semantic recall is active with backends 1 or 2.
    Backend 3 supports exact-match caching (System 1) only.
    """

    def __init__(self):
        self._backend = None  # loaded on first use
        self.dim = 384        # reported dim before load; updated after
        self.backend_name = "pending"

    def _load(self) -> None:
        if self._backend is not None:
            return

        st_model = _try_load_sentence_transformers()
        if st_model:
            self._backend = SentenceTransformerEmbedder(st_model)
            self.dim = 384
            self.backend_name = "sentence-transformers"
            return

        oai = _try_load_openai_embedder()
        if oai:
            self._backend = oai
            self.dim = 1536
            self.backend_name = "openai"
            logger.info("Mnemon embedder: OpenAI text-embedding-3-small (1536-dim, System 2 active)")
            return

        self._backend = HashProjectionEmbedder()
        self.dim = 64
        self.backend_name = "hash-projection"
        import sys as _sys
        print(
            "Mnemon: System 2 semantic recall is inactive (hash-projection fallback).\n"
            "  Enable it with one of:\n"
            "    pip install mnemon-ai[full]   # offline, no API key needed\n"
            "    export OPENAI_API_KEY=...     # uses OpenAI embeddings",
            file=_sys.stderr, flush=True,
        )

    def embed(self, text: str) -> List[float]:
        self._load()
        return self._backend.embed(text)

    def embed_full(self, text: str) -> List[float]:
        self._load()
        return self._backend.embed_full(text)

    @property
    def system2_active(self) -> bool:
        """True when semantic recall is available (not hash-projection)."""
        return self.backend_name in ("sentence-transformers", "openai")

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        va, vb = np.array(a), np.array(b)
        if va.shape != vb.shape:
            return 0.0
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)
