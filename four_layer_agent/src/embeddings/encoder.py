"""
Embedding encoder using sentence-transformers pretrained models.

This module provides semantic similarity computation using pretrained
sentence-transformers models. No training required - models are
downloaded automatically on first use.

Recommended models:
- paraphrase-multilingual-MiniLM-L12-v2 (470MB, 100+ languages)
- distiluse-base-multilingual-cased-v2 (500MB, higher quality)
- all-MiniLM-L6-v2 (80MB, English only, fast)
"""

import numpy as np
from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from four_layer_agent.src.core.data_models import SearchResult


class EmbeddingEncoder:
    """
    Embedding encoder using sentence-transformers pretrained models.

    No training required - uses pretrained models for semantic similarity.
    Models are automatically downloaded to ~/.cache/torch/sentence_transformers/
    on first use.
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        cache_size: int = 1000
    ):
        """
        Initialize the embedding encoder.

        Args:
            model_name: Pretrained model name from sentence-transformers
            cache_size: Maximum number of embeddings to cache
        """
        self._available = False
        self._model = None

        try:
            from sentence_transformers import SentenceTransformer
            # Try to load with timeout
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("Model loading timed out")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)  # 10 second timeout

            try:
                self._model = SentenceTransformer(model_name)
                self._available = True
            except (TimeoutError, Exception) as e:
                print(f"Warning: Could not load sentence-transformers model: {e}")
                print("Using fallback encoder (word overlap based).")
                self._available = False
            finally:
                signal.alarm(0)

        except ImportError:
            print("Warning: sentence-transformers not available. Using fallback encoder.")
            print("Install with: pip install sentence-transformers")

        self.cache: Dict[str, np.ndarray] = {}
        self.cache_size = cache_size
        self._cache_keys: List[str] = []

    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to embedding vector.

        Args:
            text: Input text to encode

        Returns:
            numpy array of shape (embedding_dim,)
        """
        if not self._available:
            return self._fallback_encode(text)

        # Check cache
        if text in self.cache:
            return self.cache[text]

        # Encode
        if self._available and self._model:
            embedding = self._model.encode(text, convert_to_numpy=True)
        else:
            embedding = self._fallback_encode(text)

        # Update cache (LRU strategy)
        self.cache[text] = embedding
        self._cache_keys.append(text)
        if len(self._cache_keys) > self.cache_size:
            oldest = self._cache_keys.pop(0)
            del self.cache[oldest]

        return embedding

    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.

        Args:
            text1, text2: Texts to compare

        Returns:
            Similarity score in range [0, 1], higher means more similar
        """
        if not self._available:
            return self._fallback_similarity(text1, text2)

        emb1 = self.encode(text1)
        emb2 = self.encode(text2)

        # Cosine similarity
        return float(np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
        ))

    def score_results(
        self,
        query: str,
        results: List[Dict]
    ) -> List['SearchResult']:
        """
        Score and rank search results by relevance.

        Args:
            query: Search query
            results: List of raw search results, each with url, title, snippet

        Returns:
            List of SearchResult sorted by relevance (descending)
        """
        from four_layer_agent.src.core.data_models import SearchResult

        if not self._available:
            return self._fallback_score_results(query, results)

        query_emb = self.encode(query)
        scored = []

        for r in results:
            # Combine title and snippet for encoding
            full_text = r.get('title', '') + " " + r.get('snippet', '')

            # Calculate similarity
            result_emb = self.encode(full_text)

            # Compute cosine similarity
            score = float(np.dot(query_emb, result_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(result_emb) + 1e-8
            ))

            scored.append(SearchResult(
                url=r.get('url', ''),
                title=r.get('title', ''),
                snippet=r.get('snippet', ''),
                relevance_score=score
            ))

        # Sort by relevance (descending)
        scored.sort(key=lambda x: x.relevance_score, reverse=True)
        return scored

    def batch_encode(self, texts: List[str]) -> np.ndarray:
        """
        Batch encode texts (faster than individual encoding).

        Args:
            texts: List of texts to encode

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not self._available:
            return np.array([self._fallback_encode(t) for t in texts])

        # Filter out cached texts
        uncached = [t for t in texts if t not in self.cache]

        if uncached:
            # Batch encode uncached texts
            if self._available and self._model:
                embeddings = self._model.encode(uncached, convert_to_numpy=True)
            else:
                embeddings = [self._fallback_encode(t) for t in uncached]

            for text, emb in zip(uncached, embeddings):
                self.cache[text] = emb
                self._cache_keys.append(text)
                if len(self._cache_keys) > self.cache_size:
                    oldest = self._cache_keys.pop(0)
                    del self.cache[oldest]

        # Return results (maintain original order)
        return np.array([self.cache[t] for t in texts])

    def _fallback_encode(self, text: str) -> np.ndarray:
        """Fallback encoding using simple hash (when sentence-transformers unavailable)"""
        words = set(text.lower().split())
        return np.array([hash(w) % 1000 for w in words], dtype=float)

    def _fallback_similarity(self, text1: str, text2: str) -> float:
        """Fallback similarity based on word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _fallback_score_results(
        self, query: str, results: List[Dict]
    ) -> List['SearchResult']:
        """Fallback scoring using simple word overlap"""
        from four_layer_agent.src.core.data_models import SearchResult

        scored = []
        query_words = set(query.lower().split())

        for r in results:
            full_text = r.get('title', '') + " " + r.get('snippet', '')
            text_words = set(full_text.lower().split())

            if not query_words or not text_words:
                score = 0.0
            else:
                intersection = len(query_words & text_words)
                score = intersection / len(query_words)

            scored.append(SearchResult(
                url=r.get('url', ''),
                title=r.get('title', ''),
                snippet=r.get('snippet', ''),
                relevance_score=score
            ))

        scored.sort(key=lambda x: x.relevance_score, reverse=True)
        return scored


class FallbackEncoder:
    """
    Standalone fallback encoder using simple text overlap.
    Does not require sentence-transformers.
    """

    def __init__(self):
        """Initialize the fallback encoder"""
        pass

    def encode(self, text: str) -> np.ndarray:
        """Return simple word frequency vector"""
        words = set(text.lower().split())
        return np.array([hash(w) % 1000 for w in words], dtype=float)

    def similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity based on word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def score_results(
        self, query: str, results: List[Dict]
    ) -> List['SearchResult']:
        """Score results using word overlap"""
        from four_layer_agent.src.core.data_models import SearchResult

        scored = []
        query_words = set(query.lower().split())

        for r in results:
            full_text = r.get('title', '') + " " + r.get('snippet', '')
            text_words = set(full_text.lower().split())

            if not query_words:
                score = 0.0
            else:
                intersection = len(query_words & text_words)
                score = intersection / len(query_words)

            # Bonus for exact match
            if query.lower() in full_text.lower():
                score = min(score + 0.3, 1.0)

            scored.append(SearchResult(
                url=r.get('url', ''),
                title=r.get('title', ''),
                snippet=r.get('snippet', ''),
                relevance_score=score
            ))

        scored.sort(key=lambda x: x.relevance_score, reverse=True)
        return scored
