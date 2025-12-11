"""
<followup encodedFollowup="%7B%22snippet%22%3A%22Loss%20functions%22%2C%22question%22%3A%22What%20are%20the%20components%20of%20the%20composite%20loss%20function%20used%20in%20the%20framework%3F%22%2C%22id%22%3A%22dd98c4ac-3ec1-441e-b078-49d791abd8f4%22%7D" /> for novelty learning including entropy penalty and coherence constraint.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np

class NoveltyLoss(nn.Module):
    """
    Implements the composite loss function:
    text
    L = w1·(B̂(t) - B(t))² + w2·H(B(t)) + w3·C(state_t)
    Where:
    - B̂(t): Target novelty
    - B(t): Actual novelty
    - H(B(t)): Entropy penalty with temporal smoothing
    - C(state_t): Coherence constraint
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Loss weights (initial)
        self.w1 = nn.Parameter(torch.tensor(config.get('loss', {}).get('w1', 0.6)))
        self.w2 = nn.Parameter(torch.tensor(config.get('loss', {}).get('w2', 0.2)))
        self.w3 = nn.Parameter(torch.tensor(config.get('loss', {}).get('w3', 0.2)))

        # Temporal smoothing factor
        self.mu = config.get('loss', {}).get('temporal_smoothing', 0.1)

        # Entropy calculation parameters
        self.entropy_bins = config.get('loss_config', {}).get('entropy_bins', 20)
        self.entropy_window = config.get('loss_config', {}).get('entropy_window', 100)

        # History for entropy calculation
        self.B_history = []
        self.state_history = []

        # Coherence model (simplified placeholder)
        self.coherence_model = self._build_coherence_model(config)

    def forward(self,
                B_t: torch.Tensor,
                B_target: torch.Tensor,
                state_t: Dict,
                previous_state: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute composite loss.

        Args:
            B_t: Actual novelty score
            B_target: Target novelty score
            state_t: Current cognitive state
            previous_state: Previous state for temporal calculations

        Returns:
            total_loss: Composite loss
            loss_components: Dictionary of individual loss components
        """
        # Update history
        self.B_history.append(B_t.detach().cpu().item())
        if len(self.B_history) > self.entropy_window:
            self.B_history.pop(0)

        # 1. Mean Squared Error term
        mse_loss = F.mse_loss(B_t, B_target)

        # 2. Entropy penalty
        entropy_loss = self._compute_entropy_penalty(B_t, previous_state)

        # 3. Coherence constraint
        coherence_loss = self._compute_coherence_constraint(state_t)

        # Combine losses
        total_loss = (self.w1 * mse_loss +
                     self.w2 * entropy_loss +
                     self.w3 * coherence_loss)

        # Store loss components
        loss_components = {
            'mse': mse_loss.item(),
            'entropy': entropy_loss.item(),
            'coherence': coherence_loss.item(),
            'w1': self.w1.item(),
            'w2': self.w2.item(),
            'w3': self.w3.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_components

    def _compute_entropy_penalty(self,
                                B_t: torch.Tensor,
                                previous_state: Optional[Dict] = None) -> torch.Tensor:
        """
        Compute H(B(t)) = -Σ p(B_k(t))·log(p(B_k(t))) + μ·|B(t) - B(t-1)|
        """
        # Convert B_t to scalar for history
        B_scalar = B_t.detach().cpu().item()

        if len(self.B_history) < 2:
            # Not enough history for meaningful entropy
            return torch.tensor(0.0, device=B_t.device)

        # Discretize B values for entropy calculation
        B_values = np.array(self.B_history)

        # Create bins
        min_val, max_val = B_values.min(), B_values.max()
        if max_val - min_val < 1e-6:
            # All values are the same
            return torch.tensor(0.0, device=B_t.device)

        bins = np.linspace(min_val - 1e-6, max_val + 1e-6, self.entropy_bins + 1)

        # Compute histogram
        hist, _ = np.histogram(B_values, bins=bins)
        probs = hist / len(B_values)

        # Compute Shannon entropy (avoid log(0))
        mask = probs > 0
        if mask.sum() == 0:
            entropy = 0.0
        else:
            entropy = -np.sum(probs[mask] * np.log(probs[mask]))

        # Convert to tensor
        entropy_tensor = torch.tensor(entropy, device=B_t.device)

        # Add temporal smoothing term if previous state available
        temporal_term = torch.tensor(0.0, device=B_t.device)
        if previous_state is not None and 'B' in previous_state:
            B_prev = previous_state['B']
            temporal_term = self.mu * torch.abs(B_t - B_prev)

        # Combine
        entropy_loss = entropy_tensor + temporal_term

        return entropy_loss

    def _compute_coherence_constraint(self, state_t: Dict) -> torch.Tensor:
        """
        Compute C(state_t) = Σ_i Σ_j | actual_relation - expected_relation |

        Simplified implementation.
        """
        if 'concept_vectors' not in state_t:
            return torch.tensor(0.0)

        concept_vectors = state_t['concept_vectors']
        n_concepts = len(concept_vectors)

        if n_concepts < 2:
            return torch.tensor(0.0)

        # Convert to tensor if not already
        if not isinstance(concept_vectors, torch.Tensor):
            concept_vectors = torch.stack(concept_vectors)

        # Compute actual relations (cosine similarities)
        actual_relations = torch.zeros(n_concepts, n_concepts)
        for i in range(n_concepts):
            for j in range(n_concepts):
                if i != j:
                    sim = F.cosine_similarity(concept_vectors[i].unsqueeze(0),
                                            concept_vectors[j].unsqueeze(0))
                    actual_relations[i, j] = sim

        # Get expected relations (simplified: from coherence model or memory)
        expected_relations = self._get_expected_relations(concept_vectors, state_t)

        # Compute coherence loss
        coherence_loss = torch.abs(actual_relations - expected_relations).mean()

        return coherence_loss

    def _get_expected_relations(self,
                               concept_vectors: torch.Tensor,
                               state_t: Dict) -> torch.Tensor:
        """
        Get expected relations between concepts.

        This is a placeholder - in practice, this would use a learned model
        or retrieve from memory based on concept identities.
        """
        n_concepts = concept_vectors.shape[0]

        # Simple baseline: assume concepts are independent (expected similarity = 0)
        # Or use a learned model if available
        if hasattr(self, 'coherence_model') and self.coherence_model is not None:
            # Use learned coherence model
            with torch.no_grad():
                expected = self.coherence_model(concept_vectors)
                return expected
        else:
            # Default: zero matrix (no expected relations)
            return torch.zeros(n_concepts, n_concepts, device=concept_vectors.device)

    def _build_coherence_model(self, config: Dict):
        """
        Build a model for predicting expected relations between concepts.

        This is a placeholder for a more sophisticated implementation.
        """
        relation_function = config.get('loss_config', {}).get('relation_function', 'dot_product')

        if relation_function == 'mlp':
            # Simple MLP for relation prediction
            embedding_dim = config.get('model', {}).get('embedding_dim', 512)
            model = nn.Sequential(
                nn.Linear(embedding_dim * 2, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
            return model
        else:
            # Return None for dot product or other simple methods
            return None

    def update_weights(self,
                      meta_gradients: Dict,
                      learning_rate: float = 0.001):
        """
        Update loss weights based on meta-gradients.

        Args:
            meta_gradients: Gradients from meta-optimization
            learning_rate: Learning rate for weight updates
        """
        with torch.no_grad():
            if 'w1' in meta_gradients:
                self.w1 -= learning_rate * meta_gradients['w1']
            if 'w2' in meta_gradients:
                self.w2 -= learning_rate * meta_gradients['w2']
            if 'w3' in meta_gradients:
                self.w3 -= learning_rate * meta_gradients['w3']

            # Ensure weights are positive and sum to approximately 1
            self._normalize_weights()

    def _normalize_weights(self):
        """Normalize weights to sum to 1 and ensure positivity."""
        with torch.no_grad():
            # Ensure positivity
            self.w1.data = torch.clamp(self.w1, min=0.01)
            self.w2.data = torch.clamp(self.w2, min=0.01)
            self.w3.data = torch.clamp(self.w3, min=0.01)

            # Normalize sum to 1
            total = self.w1 + self.w2 + self.w3
            self.w1.data = self.w1 / total
            self.w2.data = self.w2 / total
            self.w3.data = self.w3 / total

    def get_weights(self) -> Dict:
        """Get current loss weights."""
        return {
            'w1': self.w1.item(),
            'w2': self.w2.item(),
            'w3': self.w3.item()
        }

    def reset_history(self):
        """Reset history buffers."""
        self.B_history = []
        self.state_history = []
