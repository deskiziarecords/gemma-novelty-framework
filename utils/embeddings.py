"""
Embedding utilities for concept representation and similarity calculations.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer

class EmbeddingHelper:
    """
    Helper class for embedding operations.
    """
    text
    def __init__(self, config: Dict):
        self.config = config

        # Initialize embedding model if specified
        self.model = None
        if config.get('embeddings', {}).get('pretrained_model'):
            try:
                self.model = SentenceTransformer(
                    config['embeddings']['pretrained_model']
                )
            except:
                print(f"Warning: Could not load model {config['embeddings']['pretrained_model']}")

        # Embedding parameters
        self.embedding_dim = config.get('model', {}).get('embedding_dim', 512)
        self.normalize = config.get('embeddings', {}).get('normalization', True)

    def encode(self, text: str) -> torch.Tensor:
        """Encode text to embedding."""
        if self.model is not None:
            # Use pre-trained model
            embedding = self.model.encode(text, convert_to_tensor=True)
        else:
            # Fallback: random embedding (for testing)
            embedding = torch.randn(self.embedding_dim)

        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding

    def batch_encode(self, texts: List[str]) -> torch.Tensor:
        """Encode multiple texts."""
        if self.model is not None:
            embeddings = self.model.encode(texts, convert_to_tensor=True)
        else:
            embeddings = torch.randn(len(texts), self.embedding_dim)

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def cosine_similarity(self,
                         vec1: torch.Tensor,
                         vec2: torch.Tensor) -> float:
        """Compute cosine similarity between two vectors."""
        if self.normalize:
            vec1 = F.normalize(vec1, p=2, dim=-1)
            vec2 = F.normalize(vec2, p=2, dim=-1)

        similarity = F.cosine_similarity(vec1.unsqueeze(0),
                                       vec2.unsqueeze(0)).item()
        return similarity

    def cosine_distance(self,
                       vec1: torch.Tensor,
                       vec2: torch.Tensor) -> float:
        """Compute cosine distance (1 - similarity)."""
        similarity = self.cosine_similarity(vec1, vec2)
        return 1.0 - similarity

    def compute_centroid(self, vectors: torch.Tensor) -> torch.Tensor:
        """Compute centroid of a set of vectors."""
        centroid = vectors.mean(dim=0)
        if self.normalize:
            centroid = F.normalize(centroid, p=2, dim=-1)
        return centroid

    def find_nearest_neighbors(self,
                              query: torch.Tensor,
                              candidates: torch.Tensor,
                              k: int = 5) -> List[Tuple[int, float]]:
        """Find k nearest neighbors to query vector."""
        if self.normalize:
            query = F.normalize(query, p=2, dim=-1)
            candidates = F.normalize(candidates, p=2, dim=-1)

        # Compute similarities
        similarities = F.cosine_similarity(
            query.unsqueeze(0),
            candidates
        )

        # Get top-k
        top_k = torch.topk(similarities, min(k, len(candidates)))

        results = []
        for idx, sim in zip(top_k.indices.tolist(), top_k.values.tolist()):
            results.append((idx, sim))

        return results

    def compute_diversity(self, vectors: torch.Tensor) -> float:
        """
        Compute diversity of a set of vectors.

        Higher diversity means vectors are more spread out.
        """
        if len(vectors) < 2:
            return 0.0

        if self.normalize:
            vectors = F.normalize(vectors, p=2, dim=-1)

        # Compute pairwise distances
        n = len(vectors)
        total_distance = 0.0
        count = 0

        for i in range(n):
            for j in range(i + 1, n):
                distance = self.cosine_distance(vectors[i], vectors[j])
                total_distance += distance
                count += 1

        avg_distance = total_distance / count if count > 0 else 0.0
        return avg_distance
