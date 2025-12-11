"""
Meta-optimizer for updating loss weights based on meta-loss.
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple
import numpy as np

class MetaOptimizer:
    """
    Optimizes loss weights based on meta-loss from MetaObjective.
    """
    text
    def __init__(self, config: Dict):
        self.config = config

        # Meta-learning rate
        self.meta_lr = config.get('learning_rates', {}).get('meta_learning_rate', 0.0005)

        # Optimization method
        self.method = config.get('meta_learning', {}).get('pareto_algorithm', 'gradient')

        # Gradient clipping
        self.grad_clip = config.get('training', {}).get('gradient_clip', 1.0)

        # Momentum
        self.momentum = 0.9
        self.velocity = None

    def compute_gradients(self,
                         loss_function,
                         current_weights: Dict,
                         optimal_weights: Tuple[float, float, float]) -> Dict:
        """
        Compute gradients for loss weights based on meta-loss.

        Args:
            loss_function: The NoveltyLoss instance
            current_weights: Current loss weights
            optimal_weights: Optimal weights from MetaObjective

        Returns:
            Gradients for w1, w2, w3
        """
        # Convert to tensors
        w1 = torch.tensor(current_weights['w1'], requires_grad=True)
        w2 = torch.tensor(current_weights['w2'], requires_grad=True)
        w3 = torch.tensor(current_weights['w3'], requires_grad=True)

        # Optimal weights as targets
        w1_opt, w2_opt, w3_opt = optimal_weights
        w1_opt_t = torch.tensor(w1_opt)
        w2_opt_t = torch.tensor(w2_opt)
        w3_opt_t = torch.tensor(w3_opt)

        # Compute meta-loss components
        # 1. Direct optimization error
        direct_loss = (torch.abs(w1 - w1_opt_t) +
                      torch.abs(w2 - w2_opt_t) +
                      torch.abs(w3 - w3_opt_t)) / 3

        # 2. Temporal consistency (simplified)
        if hasattr(self, '_prev_w1'):
            temporal_loss = (torch.abs(w1 - self._prev_w1) +
                           torch.abs(w2 - self._prev_w2) +
                           torch.abs(w3 - self._prev_w3)) / 3
        else:
            temporal_loss = torch.tensor(0.0)

        # Store current for next step
        self._prev_w1 = w1.detach().clone()
        self._prev_w2 = w2.detach().clone()
        self._prev_w3 = w3.detach().clone()

        # 3. Diversity bonus (entropy)
        # Convert to probability distribution
        weights_probs = torch.stack([w1, w2, w3])
        weights_probs = torch.softmax(weights_probs, dim=0)

        # Compute entropy
        entropy = -torch.sum(weights_probs * torch.log(weights_probs + 1e-10))

        # Get meta parameters
        alpha = self.config.get('meta', {}).get('meta_alpha', 0.6)
        beta = self.config.get('meta', {}).get('meta_beta', 0.3)
        gamma = self.config.get('meta', {}).get('meta_gamma', 0.1)

        # Combine into meta-loss
        meta_loss = (alpha * direct_loss +
                    beta * temporal_loss -
                    gamma * entropy)

        # Compute gradients
        meta_loss.backward()

        # Extract gradients
        gradients = {
            'w1': w1.grad.item(),
            'w2': w2.grad.item(),
            'w3': w3.grad.item()
        }

        # Apply gradient clipping
        grad_norm = np.sqrt(sum(g**2 for g in gradients.values()))
        if grad_norm > self.grad_clip:
            scale = self.grad_clip / grad_norm
            for key in gradients:
                gradients[key] *= scale

        return gradients

    def update_weights(self,
                      loss_function,
                      gradients: Dict):
        """
        Update loss weights using computed gradients.

        Args:
            loss_function: The NoveltyLoss instance
            gradients: Gradients for w1, w2, w3
        """
        # Initialize momentum if not exists
        if self.velocity is None:
            self.velocity = {
                'w1': 0.0,
                'w2': 0.0,
                'w3': 0.0
            }

        # Update with momentum
        for key in ['w1', 'w2', 'w3']:
            if key in gradients:
                # Momentum update
                self.velocity[key] = (self.momentum * self.velocity[key] +
                                     self.meta_lr * gradients[key])

                # Apply update to loss function
                if key == 'w1':
                    loss_function.w1.data -= self.velocity[key]
                elif key == 'w2':
                    loss_function.w2.data -= self.velocity[key]
                elif key == 'w3':
                    loss_function.w3.data -= self.velocity[key]

        # Ensure weights are valid
        loss_function._normalize_weights()

    def step(self,
            loss_function,
            meta_objective,
            current_performance: Dict) -> Dict:
        """
        Complete meta-optimization step.

        Args:
            loss_function: NoveltyLoss instance
            meta_objective: MetaObjective instance
            current_performance: Current performance metrics

        Returns:
            Meta-optimization statistics
        """
        # Get current weights
        current_weights = loss_function.get_weights()

        # Update meta-objective and get optimal weights
        optimal_weights = meta_objective.update(
            utility=current_performance.get('utility', 0.0),
            verifiability=current_performance.get('verifiability', 0.0),
            relevance=current_performance.get('relevance', 0.0),
            current_weights=(current_weights['w1'],
                           current_weights['w2'],
                           current_weights['w3']),
            step=current_performance.get('step', 0)
        )

        # Compute meta-loss
        meta_loss = meta_objective.compute_meta_loss(
            current_weights=(current_weights['w1'],
                           current_weights['w2'],
                           current_weights['w3']),
            optimal_weights=optimal_weights
        )

        # Compute gradients
        gradients = self.compute_gradients(
            loss_function,
            current_weights,
            optimal_weights
        )

        # Update weights
        self.update_weights(loss_function, gradients)

        # Get updated weights
        updated_weights = loss_function.get_weights()

        # Prepare statistics
        stats = {
            'meta_loss': meta_loss['total'],
            'meta_loss_components': meta_loss,
            'optimal_weights': optimal_weights,
            'updated_weights': updated_weights,
            'gradients': gradients
        }

        return stats

    def reset(self):
        """Reset optimizer state."""
        self.velocity = None
        if hasattr(self, '_prev_w1'):
            del self._prev_w1
            del self._prev_w2
            del self._prev_w3
