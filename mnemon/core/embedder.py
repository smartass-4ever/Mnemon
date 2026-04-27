"""
Mnemon Embedder — auto-upgrading embedding backend.

Priority order (automatic, zero config):
  1. sentence-transformers  — 384-dim real semantic embeddings
  2. HashProjectionEmbedder — 64-dim fallback, always available

The moment sentence-transformers is installed, Mnemon upgrades silently.
No code changes needed.
"""

import hashlib
import logging
from typing import List

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


class HashProjectionEmbedder:
    """
    Lightweight hash-projection fallback. 64-dim activation, 384-dim full.
    Always available — zero dependencies beyond numpy.
    Retrieval precision: ~56% on eval suite.
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


class SimpleEmbedder:
    """
    Public embedder interface. Auto-selects best available backend.

    With sentence-transformers: 384-dim, ~85% retrieval precision.
    Without: hash-projection 64-dim fallback, ~56% retrieval precision.

    Upgrade: pip install mnemon-ai[full]
    """

    def __init__(self):
        st_model = _try_load_sentence_transformers()
        if st_model:
            self._backend = SentenceTransformerEmbedder(st_model)
            self.dim = 384
            self.backend_name = "sentence-transformers"
        else:
            self._backend = HashProjectionEmbedder()
            self.dim = 64
            self.backend_name = "hash-projection"
            logger.warning(
                "Mnemon embedder: using hash-projection fallback (64-dim, ~56%% recall). "
                "Upgrade for production quality: pip install mnemon-ai[full]"
            )

    def embed(self, text: str) -> List[float]:
        return self._backend.embed(text)

    def embed_full(self, text: str) -> List[float]:
        return self._backend.embed_full(text)

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
