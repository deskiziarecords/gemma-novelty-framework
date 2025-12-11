"""
Memory system for canonical concept representations and retrieval.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict
import heapq

class ConceptMemory:
    """
    Manages canonical memory representations of concepts using EMA.
    text
    Memory(v_i) = α_mem · Memory(v_i) + (1-α_mem) · v_new
    """
    def __init__(self, config: Dict):
        self.config = config

        # Memory storage
        self.memory_vectors = {}  # concept_id -> canonical vector
        self.memory_counts = {}   # concept_id -> observation count
        self.temporal_contexts = defaultdict(list)  # concept_id -> temporal patterns
        self.relations = defaultdict(dict)  # concept_id -> {related_id: strength}

        # EMA parameters
        self.decay_factor = config.get('memory', {}).get('decay_factor', 0.999)
        self.memory_size = config.get('memory', {}).get('memory_size', 10000)

        # Cache for quick retrieval
        self.retrieval_cache = {}
        self.cache_ttl = config.get('memory_config', {}).get('cache_ttl', 3600)

        # Normalization
        self.normalize = config.get('memory', {}).get('normalization', True)

    def update_memory(self,
                     concept_id: str,
                     vector: torch.Tensor,
                     context: Optional[Dict] = None):
        """
        Update canonical memory for a concept using EMA.

        Args:
            concept_id: Unique identifier for the concept
            vector: Current embedding of the concept
            context: Optional temporal/contextual information
        """
        # Normalize vector
        if self.normalize:
            vector = F.normalize(vector, p=2, dim=-1)

        # Update canonical representation
        if concept_id in self.memory_vectors:
            # EMA update
            old_vector = self.memory_vectors[concept_id]
            new_vector = (self.decay_factor * old_vector +
                         (1 - self.decay_factor) * vector)
            self.memory_vectors[concept_id] = new_vector
            self.memory_counts[concept_id] += 1
        else:
            # First observation
            self.memory_vectors[concept_id] = vector
            self.memory_counts[concept_id] = 1

            # Prune if memory is full
            if len(self.memory_vectors) > self.memory_size:
                self._prune_memory()

        # Update temporal context
        if context is not None:
            self.temporal_contexts[concept_id].append(context)
            # Keep only recent history
            max_history = self.config.get('memory_config', {}).get('max_temporal_history', 100)
            if len(self.temporal_contexts[concept_id]) > max_history:
                self.temporal_contexts[concept_id] = self.temporal_contexts[concept_id][-max_history:]

        # Clear cache for this concept
        if concept_id in self.retrieval_cache:
            del self.retrieval_cache[concept_id]

    def get_canonical(self,
                     vector: torch.Tensor,
                     threshold: float = 0.8) -> Tuple[Optional[str], torch.Tensor]:
        """
        Retrieve or create canonical memory for a vector.

        Args:
            vector: Input embedding
            threshold: Similarity threshold for matching existing concepts

        Returns:
            concept_id: ID of matched or new concept
            canonical_vector: Canonical memory representation
        """
        if self.normalize:
            vector = F.normalize(vector, p=2, dim=-1)

        # Try to find matching concept
        best_match_id = None
        best_similarity = -1.0

        for concept_id, mem_vector in self.memory_vectors.items():
            similarity = F.cosine_similarity(vector.unsqueeze(0),
                                           mem_vector.unsqueeze(0)).item()
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = concept_id

        # Check if match is above threshold
        if best_match_id and best_similarity >= threshold:
            canonical_vector = self.memory_vectors[best_match_id]
            return best_match_id, canonical_vector
        else:
            # Create new concept
            new_id = f"concept_{len(self.memory_vectors)}"
            self.memory_vectors[new_id] = vector.clone()
            self.memory_counts[new_id] = 1
            return new_id, vector

    def retrieve_context(self,
                        query_vector: torch.Tensor,
                        top_k: int = 5) -> List[Tuple[str, torch.Tensor, float]]:
        """
        Retrieve top-k most similar concepts from memory.

        Args:
            query_vector: Query embedding
            top_k: Number of results to return

        Returns:
            List of (concept_id, vector, similarity) tuples
        """
        if self.normalize:
            query_vector = F.normalize(query_vector, p=2, dim=-1)

        # Check cache
        cache_key = (query_vector.cpu().numpy().tobytes(), top_k)
        if cache_key in self.retrieval_cache:
            return self.retrieval_cache[cache_key]

        # Compute similarities
        results = []
        for concept_id, mem_vector in self.memory_vectors.items():
            similarity = F.cosine_similarity(query_vector.unsqueeze(0),
                                           mem_vector.unsqueeze(0)).item()
            results.append((concept_id, mem_vector, similarity))

        # Sort by similarity
        results.sort(key=lambda x: x[2], reverse=True)
        results = results[\:top_k]

        # Update cache
        self.retrieval_cache[cache_key] = results

        return results

    def update_relations(self,
                        concept_id1: str,
                        concept_id2: str,
                        strength: float = 1.0):
        """
        Update relation strength between two concepts.

        Args:
            concept_id1: First concept ID
            concept_id2: Second concept ID
            strength: Relation strength to add
        """
        if concept_id1 not in self.relations:
            self.relations[concept_id1] = {}
        if concept_id2 not in self.relations:
            self.relations[concept_id2] = {}

        # Update bidirectional relation
        current_strength1 = self.relations[concept_id1].get(concept_id2, 0.0)
        self.relations[concept_id1][concept_id2] = current_strength1 + strength

        current_strength2 = self.relations[concept_id2].get(concept_id1, 0.0)
        self.relations[concept_id2][concept_id1] = current_strength2 + strength

    def get_related_concepts(self,
                           concept_id: str,
                           threshold: float = 0.1) -> List[Tuple[str, float]]:
        """
        Get concepts related to a given concept.

        Args:
            concept_id: Concept ID to query
            threshold: Minimum relation strength

        Returns:
            List of (related_concept_id, strength) tuples
        """
        if concept_id not in self.relations:
            return []

        relations = self.relations[concept_id]
        results = [(rel_id, strength) for rel_id, strength in relations.items()
                  if strength >= threshold]
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def compute_memory_score(self, concept_id: str) -> float:
        """
        Compute Memory_Score = α|E(v_i)| + β∑T(v_i) + γdeg(R(v_i))

        Returns:
            Memory score indicating strength/establishedness
        """
        if concept_id not in self.memory_vectors:
            return 0.0

        # Get config weights
        alpha = self.config.get('memory_config', {}).get('alpha_M', 0.4)
        beta = self.config.get('memory_config', {}).get('beta_M', 0.3)
        gamma = self.config.get('memory_config', {}).get('gamma_M', 0.3)

        # Embedding magnitude term
        vector = self.memory_vectors[concept_id]
        embedding_term = alpha * vector.norm().item()

        # Temporal context term
        temporal_contexts = self.temporal_contexts.get(concept_id, [])
        temporal_term = beta * len(temporal_contexts)

        # Relation degree term
        relations = self.relations.get(concept_id, {})
        degree_term = gamma * len(relations)

        memory_score = embedding_term + temporal_term + degree_term

        return memory_score

    def _prune_memory(self):
        """Prune least used concepts from memory."""
        if len(self.memory_vectors) <= self.memory_size:
            return

        # Score concepts by usage and recency
        scores = []
        for concept_id, count in self.memory_counts.items():
            # Simple scoring: count * recency_factor
            # In practice, you might want a more sophisticated scoring
            score = count * (1.0 / (1 + len(self.memory_vectors) - self.memory_counts[concept_id]))
            scores.append((score, concept_id))

        # Sort by score (ascending)
        scores.sort(key=lambda x: x[0])

        # Remove lowest scoring concepts
        num_to_remove = len(self.memory_vectors) - self.memory_size
        for _, concept_id in scores[\:num_to_remove]:
            if concept_id in self.memory_vectors:
                del self.memory_vectors[concept_id]
            if concept_id in self.memory_counts:
                del self.memory_counts[concept_id]
            if concept_id in self.temporal_contexts:
                del self.temporal_contexts[concept_id]
            if concept_id in self.relations:
                del self.relations[concept_id]

        # Clear cache
        self.retrieval_cache.clear()

    def get_statistics(self) -> Dict:
        """Get memory system statistics."""
        return {
            'num_concepts': len(self.memory_vectors),
            'total_observations': sum(self.memory_counts.values()),
            'avg_observations': np.mean(list(self.memory_counts.values())) if self.memory_counts else 0,
            'cache_size': len(self.retrieval_cache),
            'memory_usage_mb': self._estimate_memory_usage()
        }

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        total_bytes = 0

        # Memory vectors
        for vector in self.memory_vectors.values():
            total_bytes += vector.element_size() * vector.nelement()

        # Counts and other data structures
        total_bytes += len(self.memory_counts) * (8 + 8)  # key + value
        total_bytes += len(self.temporal_contexts) * 100  # Approximate

        return total_bytes / (1024 * 1024)  # Convert to MB
