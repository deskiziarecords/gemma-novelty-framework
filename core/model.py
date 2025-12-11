"""
Main recursive model implementing the B_t equation and cognitive update.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np

class BeliefUpdateModel(nn.Module):
    """
    Implements the recursive belief update equation:
    text
    B_t = k_t · (I_t - R_t) + η_t · C_total(t) + ε_t
    Where:
    - I_t: Input information
    - R_t: Retrieved content
    - k_t, η_t: Learnable sensitivities
    - C_total(t): Synthesized Cognitive Complexity
    - ε_t: Noise/uncertainty
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Learnable parameters
        self.k_t = nn.Parameter(torch.tensor(config.get('k_t_initial', 0.5)))
        self.eta_t = nn.Parameter(torch.tensor(config.get('eta_t_initial', 0.5)))

        # Disparity amplification exponent
        self.alpha_D = nn.Parameter(torch.tensor(config.get('alpha_D_initial', 1.0)))

        # Semantic shift weights
        self.gamma = nn.Parameter(torch.tensor(config.get('gamma_initial', 1.0)))
        self.delta = nn.Parameter(torch.tensor(config.get('delta_initial', 1.0)))

        # Noise level
        self.register_buffer('epsilon', torch.tensor(config.get('noise_epsilon', 0.01)))

        # Normalization layers
        self.norm_input = nn.LayerNorm(config.get('embedding_dim', 512))
        self.norm_memory = nn.LayerNorm(config.get('embedding_dim', 512))

    def forward(self,
                input_embeddings: torch.Tensor,
                retrieved_embeddings: torch.Tensor,
                memory_system,
                active_concepts: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Compute B_t for current time step.

        Args:
            input_embeddings: Input information I_t
            retrieved_embeddings: Retrieved content R_t
            memory_system: <followup encodedFollowup="%7B%22snippet%22%3A%22Memory%20system%22%2C%22question%22%3A%22How%20does%20the%20memory%20system%20manage%20canonical%20concept%20representations%3F%22%2C%22id%22%3A%2296e9b7d9-83c3-4fa8-aa4b-3997e0722af8%22%7D" /> for concept weights
            active_concepts: Dictionary of active concepts and their states

        Returns:
            B_t: Novelty/belief update score
            metrics: Dictionary of intermediate computations
        """
        # Normalize inputs
        I_t = self.norm_input(input_embeddings)
        R_t = self.norm_memory(retrieved_embeddings)

        # Compute input novelty term
        input_novelty = self.k_t * (I_t - R_t).norm(dim=-1).mean()

        # Compute synthesized cognitive complexity
        C_total = self._compute_c_total(active_concepts, memory_system)

        # Add noise
        noise = torch.randn_like(input_novelty) * self.epsilon

        # Final belief update
        B_t = input_novelty + self.eta_t * C_total + noise

        # Prepare metrics
        metrics = {
            'input_novelty': input_novelty.item(),
            'C_total': C_total.item(),
            'k_t': self.k_t.item(),
            'eta_t': self.eta_t.item(),
            'alpha_D': self.alpha_D.item(),
            'gamma': self.gamma.item(),
            'delta': self.delta.item()
        }

        return B_t, metrics

    def _compute_c_total(self,
                          active_concepts: Dict,
                          memory_system) -> torch.Tensor:
        """
        Compute C_total(t) = Σ_i [w_i(t)·(D_i(t))^α_D + γ·Δv_i(t) + δ·Δf_ri(t)]
        """
        # Extract active concept vectors
        concept_vectors = active_concepts.get('vectors', [])
        if len(concept_vectors) == 0:
            return torch.tensor(0.0)

        # Convert to tensor
        vectors = torch.stack(concept_vectors)

        # Compute concept weights w_i(t)
        weights = self._compute_concept_weights(vectors, memory_system)

        # Compute concept disparities D_i(t)
        disparities = self._compute_concept_disparities(vectors)

        # Compute semantic shifts
        semantic_shifts = self._compute_semantic_shifts(vectors, active_concepts)

        # Combine terms
        combinatorial_term = (weights * disparities.pow(self.alpha_D)).sum()
        semantic_term = self.gamma * semantic_shifts['vector_shift']
        relational_term = self.delta * semantic_shifts['relational_shift']

        C_total = combinatorial_term + semantic_term + relational_term

        return C_total

    def _compute_concept_weights(self,
                                   vectors: torch.Tensor,
                                   memory_system) -> torch.Tensor:
        """Compute w_i(t) = A_i(t) · ||v_i(t) - Memory(v_i)||"""
        # Get activations (simplified - could be attention weights)
        activations = F.softmax(vectors.mean(dim=-1), dim=0)

        # Get memory distances
        memory_distances = []
        for i, vec in enumerate(vectors):
            memory_vec = memory_system.get_canonical(vec)
            distance = (vec - memory_vec).norm()
            memory_distances.append(distance)

        distances = torch.stack(memory_distances)

        # Combine
        weights = activations * distances

        return weights

    def _compute_concept_disparities(self, vectors: torch.Tensor) -> torch.Tensor:
        """Compute D_i(t) = 1 - cos(v_i(t), centroid of other concepts)"""
        n_concepts = vectors.shape[0]
        disparities = []

        for i in range(n_concepts):
            # Get other vectors
            mask = torch.ones(n_concepts, dtype=torch.bool)
            mask[i] = False
            other_vectors = vectors[mask]

            # Compute centroid
            centroid = other_vectors.mean(dim=0)

            # Compute cosine similarity
            similarity = F.cosine_similarity(vectors[i].unsqueeze(0),
                                           centroid.unsqueeze(0))

            # Convert to disparity (1 - similarity)
            disparity = 1 - similarity.clamp(-1, 1)
            disparities.append(disparity)

        return torch.stack(disparities)

    def _compute_semantic_shifts(self, vectors: torch.Tensor, active_concepts: Dict) -> Dict:
        """Compute semantic shift components"""
        # Vector shift: ||v_i(t) - v_i(t-1)||
        previous_vectors = active_concepts.get('previous_vectors', vectors)
        vector_shifts = (vectors - previous_vectors).norm(dim=-1).mean()

        # Relational shift (simplified)
        relational_shifts = torch.tensor(0.1)  # Placeholder

        return {
            'vector_shift': vector_shifts,
            'relational_shift': relational_shifts
        }

    def update_parameters(self,
                         loss: torch.Tensor,
                         learning_rates: Dict):
        """Update learnable parameters via gradient descent"""
        # Compute gradients
        loss.backward()

        # Apply gradient updates
        with torch.no_grad():
            self.k_t -= learning_rates.get('k_t', 0.01) * self.k_t.grad
            self.eta_t -= learning_rates.get('eta_t', 0.01) * self.eta_t.grad
            self.alpha_D -= learning_rates.get('alpha_D', 0.001) * self.alpha_D.grad
            self.gamma -= learning_rates.get('gamma', 0.001) * self.gamma.grad
            self.delta -= learning_rates.get('delta', 0.001) * self.delta.grad

            # Zero gradients
            self.k_t.grad.zero_()
            self.eta_t.grad.zero_()
            self.alpha_D.grad.zero_()
            self.gamma.grad.zero_()
            self.delta.grad.zero_()
