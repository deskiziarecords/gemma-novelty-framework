"""
Meta-objective for <followup encodedFollowup="%7B%22snippet%22%3A%22Pareto-based%20optimization%22%2C%22question%22%3A%22How%20does%20Pareto%20optimization%20help%20in%20balancing%20utility%2C%20verifiability%2C%20and%20relevance%3F%22%2C%22id%22%3A%22d1737dee-031b-469c-96f8-7e1a169bff06%22%7D" /> of loss weights.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import heapq

@dataclass
class PerformancePoint:
    """Represents a point in the performance space."""
    utility: float
    verifiability: float
    relevance: float
    weights: Tuple[float, float, float] # w1, w2, w3
    step: int

class MetaObjective:
    """
    Implements Pareto-based optimization for determining optimal loss weights.
    text
    Uses multi-objective optimization to find efficient trade-offs between
    utility, verifiability, and relevance objectives.
    """
    def __init__(self, config: Dict):
        self.config = config

        # Pareto optimization parameters
        self.pareto_window = config.get('meta', {}).get('pareto_window', 100)
        self.adaptation_rate = config.get('meta', {}).get('adaptation_rate', 0.1)
        self.stagnation_window = config.get('meta', {}).get('stagnation_window', 10)
        self.min_entropy = config.get('meta', {}).get('min_entropy', 0.5)

        # Performance history
        self.performance_history: List[PerformancePoint] = []
        self.pareto_front: List[PerformancePoint] = []

        # Learning dynamics tracking
        self.learning_rates_history = defaultdict(list)
        self.stagnation_counters = defaultdict(int)

    def update(self,
               utility: float,
               verifiability: float,
               relevance: float,
               current_weights: Tuple[float, float, float],
               step: int) -> Tuple[float, float, float]:
        """
        Update meta-objective and compute optimal weights.

        Args:
            utility: Current utility score
            verifiability: Current verifiability score
            relevance: Current relevance score
            current_weights: Current loss weights (w1, w2, w3)
            step: Current training step

        Returns:
            Optimal weights (w1_opt, w2_opt, w3_opt)
        """
        # Create performance point
        point = PerformancePoint(
            utility=utility,
            verifiability=verifiability,
            relevance=relevance,
            weights=current_weights,
            step=step
        )

        # Add to history
        self.performance_history.append(point)

        # Keep history bounded
        if len(self.performance_history) > self.pareto_window:
            self.performance_history.pop(0)

        # Update Pareto front
        self._update_pareto_front()

        # Compute optimal weights from Pareto front
        if len(self.pareto_front) > 0:
            optimal_weights = self._weights_from_pareto(current_weights)
        else:
            optimal_weights = current_weights

        # Apply constraints
        optimal_weights = self._apply_constraints(optimal_weights)

        return optimal_weights

    def _update_pareto_front(self):
        """Update the Pareto front from performance history."""
        if len(self.performance_history) == 0:
            self.pareto_front = []
            return

        # Extract objectives
        points = self.performance_history

        # Find non-dominated points (Pareto front)
        front = []
        for i, p1 in enumerate(points):
            dominated = False
            for j, p2 in enumerate(points):
                if i != j:
                    # Check if p2 dominates p1
                    if (p2.utility >= p1.utility and
                        p2.verifiability >= p1.verifiability and
                        p2.relevance >= p1.relevance and
                        (p2.utility > p1.utility or
                         p2.verifiability > p1.verifiability or
                         p2.relevance > p1.relevance)):
                        dominated = True
                        break
            if not dominated:
                front.append(p1)

        self.pareto_front = front

    def _weights_from_pareto(self,
                           current_weights: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Compute optimal weights from Pareto front.

        Args:
            current_weights: Current weights for adaptation

        Returns:
            Optimal weights considering Pareto front and learning dynamics
        """
        if len(self.pareto_front) == 0:
            return current_weights

        # Analyze learning dynamics
        learning_rates = self._compute_learning_rates()
        stagnation = self._detect_stagnation(learning_rates)

        # Get weights from Pareto front
        # Simple strategy: average weights of Pareto-optimal points
        pareto_weights = []
        for point in self.pareto_front:
            pareto_weights.append(point.weights)

        avg_pareto_weights = np.mean(pareto_weights, axis=0)

        # Adapt based on stagnation
        # More stagnation → more adaptation towards Pareto front
        adaptation = np.exp(-stagnation * self.adaptation_rate)

        # Interpolate between current and Pareto-optimal weights
        optimal_weights = (
            (1 - adaptation) * np.array(current_weights) +
            adaptation * avg_pareto_weights
        )

        # Normalize to sum to 1
        optimal_weights = optimal_weights / optimal_weights.sum()

        return tuple(optimal_weights.tolist())

    def _compute_learning_rates(self) -> np.ndarray:
        """Compute learning rates for each objective."""
        if len(self.performance_history) < 2:
            return np.zeros(3)

        # Extract recent performance
        window = min(self.stagnation_window, len(self.performance_history))
        recent_points = self.performance_history[-window:]

        # Compute improvements
        utilities = [p.utility for p in recent_points]
        verifiabilities = [p.verifiability for p in recent_points]
        relevances = [p.relevance for p in recent_points]

        # Compute rates of change
        utility_rate = np.mean(np.diff(utilities)) if len(utilities) > 1 else 0
        verifiability_rate = np.mean(np.diff(verifiabilities)) if len(verifiabilities) > 1 else 0
        relevance_rate = np.mean(np.diff(relevances)) if len(relevances) > 1 else 0

        rates = np.array([utility_rate, verifiability_rate, relevance_rate])

        # Store for history
        self.learning_rates_history['utility'].append(utility_rate)
        self.learning_rates_history['verifiability'].append(verifiability_rate)
        self.learning_rates_history['relevance'].append(relevance_rate)

        return rates

    def _detect_stagnation(self, learning_rates: np.ndarray) -> np.ndarray:
        """Detect stagnation in learning progress."""
        if len(self.learning_rates_history['utility']) < 2:
            return np.zeros(3)

        # Check if rates are near zero
        stagnation = np.exp(-np.abs(learning_rates))

        # Update stagnation counters
        for i, rate in enumerate(learning_rates):
            if abs(rate) < 0.001:  # Threshold for stagnation
                self.stagnation_counters[i] += 1
            else:
                self.stagnation_counters[i] = max(0, self.stagnation_counters[i] - 1)

        # Adjust stagnation based on counter
        max_counter = 10
        for i in range(3):
            stagnation[i] = min(1.0, stagnation[i] * (1 + self.stagnation_counters[i] / max_counter))

        return stagnation

    def _apply_constraints(self, weights: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Apply constraints to weights."""
        weights = np.array(weights)

        # 1. Non-negativity
        weights = np.maximum(weights, 0.1)  # Minimum weight

        # 2. Sum to 1
        weights = weights / weights.sum()

        # 3. Minimum diversity (entropy constraint)
        entropy = -np.sum(weights * np.log(weights + 1e-10))
        if entropy < self.min_entropy:
            weights = self._increase_diversity(weights)

        return tuple(weights.tolist())

    def _increase_diversity(self, weights: np.ndarray) -> np.ndarray:
        """Increase diversity of weights."""
        # Move towards more uniform distribution
        uniform = np.ones_like(weights) / len(weights)

        # Interpolate towards uniform
        alpha = 0.3  # Mixing parameter
        diversified = (1 - alpha) * weights + alpha * uniform

        # Renormalize
        diversified = diversified / diversified.sum()

        return diversified

    def compute_meta_loss(self,
                         current_weights: Tuple[float, float, float],
                         optimal_weights: Tuple[float, float, float]) -> Dict:
        """
        Compute meta-loss for weight optimization.

        L_meta = α·|w - w_optimal| + β·|Δw| - γ·H(w)
        """
        current = np.array(current_weights)
        optimal = np.array(optimal_weights)

        # Get meta parameters
        alpha = self.config.get('meta', {}).get('meta_alpha', 0.6)
        beta = self.config.get('meta', {}).get('meta_beta', 0.3)
        gamma = self.config.get('meta', {}).get('meta_gamma', 0.1)

        # 1. Direct optimization error
        direct_loss = np.abs(current - optimal).mean()

        # 2. Temporal consistency penalty
        if hasattr(self, '_previous_weights'):
            weight_change = np.abs(current - self._previous_weights).mean()
        else:
            weight_change = 0.0
        self._previous_weights = current.copy()

        temporal_loss = weight_change

        # 3. Diversity bonus (negative because we subtract it)
        entropy = -np.sum(current * np.log(current + 1e-10))
        diversity_bonus = entropy

        # Combined meta-loss
        meta_loss = (alpha * direct_loss +
                    beta * temporal_loss -
                    gamma * diversity_bonus)

        # Store for statistics
        self._last_meta_loss = {
            'direct': direct_loss,
            'temporal': temporal_loss,
            'diversity': diversity_bonus,
            'total': meta_loss
        }

        return self._last_meta_loss

    def get_statistics(self) -> Dict:
        """Get meta-objective statistics."""
        stats = {
            'performance_history_size': len(self.performance_history),
            'pareto_front_size': len(self.pareto_front),
            'stagnation_counters': dict(self.stagnation_counters)
        }

        if hasattr(self, '_last_meta_loss'):
            stats.update(self._last_meta_loss)

        # Add Pareto front statistics if available
        if len(self.pareto_front) > 0:
            utilities = [p.utility for p in self.pareto_front]
            verifiabilities = [p.verifiability for p in self.pareto_front]
            relevances = [p.relevance for p in self.pareto_front]

            stats.update({
                'pareto_avg_utility': np.mean(utilities),
                'pareto_avg_verifiability': np.mean(verifiabilities),
                'pareto_avg_relevance': np.mean(relevances),
                'pareto_std_utility': np.std(utilities),
                'pareto_std_verifiability': np.std(verifiabilities),
                'pareto_std_relevance': np.std(relevances)
            })

        return stats

    def reset(self):
        """Reset meta-objective state."""
        self.performance_history = []
        self.pareto_front = []
        self.learning_rates_history = defaultdict(list)
        self.stagnation_counters = defaultdict(int)
        if hasattr(self, '_previous_weights'):
            del self._previous_weights
        if hasattr(self, '_last_meta_loss'):
            del self._last_meta_loss
