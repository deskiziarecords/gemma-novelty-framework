"""
<followup encodedFollowup="%7B%22snippet%22%3A%22Satisfaction%20dynamics%22%2C%22question%22%3A%22How%20do%20satisfaction%20dynamics%20control%20the%20exploration%20vs.%20exploitation%20trade-off%3F%22%2C%22id%22%3A%22e5a1e1f5-a034-4c69-89d5-53d2b0717076%22%7D" /> and cognitive control for exploration vs exploitation.
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple
import numpy as np

class SatisfactionDynamics:
    """
    Implements satisfaction dynamics for cognitive control:
    text
    S(t) = (1-λ)·S(t-1) + λ·[ρ·U(t) + κ·V(t) + τ·R(t)]
    Control rules:
    - If S(t) > 0: dx/dt = +1, dS/dt = -1 (Explore more)
    - If S(t) = 0: dx/dt = -1, dS/dt = +1 (Focus/re-evaluate)
    """
    def __init__(self, config: Dict):
        self.config = config

        # Satisfaction parameters
        self.lambda_S = config.get('satisfaction', {}).get('lambda_S', 0.1)
        self.rho = config.get('satisfaction', {}).get('rho', 0.4)
        self.kappa = config.get('satisfaction', {}).get('kappa', 0.3)
        self.tau = config.get('satisfaction', {}).get('tau', 0.3)

        # Bounds
        self.S_min = config.get('satisfaction', {}).get('S_min', 0.15)
        self.S_max = config.get('satisfaction', {}).get('S_max', 1.0)

        # Exploration parameters
        self.depth_increment = config.get('exploration', {}).get('depth_increment', 1.0)
        self.depth_decrement = config.get('exploration', {}).get('depth_decrement', 1.0)
        self.satisfaction_increment = config.get('exploration', {}).get('satisfaction_increment', 1.0)
        self.satisfaction_decrement = config.get('exploration', {}).get('satisfaction_decrement', 1.0)
        self.depth_to_n_scale = config.get('exploration', {}).get('depth_to_n_scale', 0.1)

        # State variables
        self.S = config.get('initialization', {}).get('S0_initial', 0.7)
        self.x = 0.0  # Depth/exploration level
        self.N_min = config.get('dynamics', {}).get('N_min', 5)
        self.N_max = config.get('dynamics', {}).get('N_max', 100)

        # History
        self.history = {
            'S': [],
            'x': [],
            'N': [],
            'U': [],
            'V': [],
            'R': []
        }

    def update(self,
               U_t: float,
               V_t: float,
               R_t: float,
               time_step: int = 1) -> Tuple[float, float, int]:
        """
        Update satisfaction and exploration dynamics.

        Args:
            U_t: Utility score at time t
            V_t: Verifiability score at time t
            R_t: Relevance score at time t
            time_step: Discrete time step (for discrete approximation)

        Returns:
            S_t: Updated satisfaction
            x_t: Updated depth/exploration level
            N_t: Number of active concepts
        """
        # Store metrics
        self.history['U'].append(U_t)
        self.history['V'].append(V_t)
        self.history['R'].append(R_t)

        # Update satisfaction using EMA
        novelty_score = self.rho * U_t + self.kappa * V_t + self.tau * R_t
        S_new = (1 - self.lambda_S) * self.S + self.lambda_S * novelty_score

        # Apply bounds
        S_new = np.clip(S_new, self.S_min, self.S_max)

        # Update exploration depth based on satisfaction
        if S_new > 0:
            # Satisfied: Explore more (increase depth)
            x_new = self.x + self.depth_increment * time_step
            # S decreases as we explore
            S_new = max(self.S_min, S_new - self.satisfaction_decrement * time_step)
        else:
            # Not satisfied: Focus/re-evaluate (decrease depth)
            x_new = max(0, self.x - self.depth_decrement * time_step)
            # S increases as we focus
            S_new = min(self.S_max, S_new + self.satisfaction_increment * time_step)

        # Map depth to number of active concepts
        N_t = self._depth_to_N(x_new)

        # Update state
        self.S = S_new
        self.x = x_new

        # Store history
        self.history['S'].append(S_new)
        self.history['x'].append(x_new)
        self.history['N'].append(N_t)

        return S_new, x_new, N_t

    def _depth_to_N(self, x: float) -> int:
        """
        Map exploration depth x to number of active concepts N.

        N = f(x) where f is an increasing function bounded by [N_min, N_max]
        """
        mapping_method = self.config.get('dynamics', {}).get('depth_mapping', 'exponential')
        depth_scale = self.config.get('dynamics', {}).get('depth_scale', 10.0)

        if mapping_method == 'linear':
            N = int(self.N_min + (self.N_max - self.N_min) * (x / depth_scale))
        elif mapping_method == 'exponential':
            # Exponential mapping
            N = int(self.N_min + (self.N_max - self.N_min) *
                   (1 - np.exp(-x / depth_scale)))
        elif mapping_method == 'step':
            # Step function
            steps = [5, 10, 20, 35, 50, 70, 100]
            x_idx = min(int(x), len(steps) - 1)
            N = steps[x_idx]
        else:
            # Default: sigmoid
            N = int(self.N_min + (self.N_max - self.N_min) *
                   (1 / (1 + np.exp(-(x - depth_scale/2) / (depth_scale/10)))))

        # Ensure bounds
        N = max(self.N_min, min(self.N_max, N))

        return N

    def get_active_concepts_threshold(self, concept_scores: np.ndarray) -> float:
        """
        Get activation threshold based on current depth x.

        Higher x (more exploration) → lower threshold → more concepts active.
        """
        # Normalize x to [0, 1]
        x_norm = self.x / self.config.get('dynamics', {}).get('depth_scale', 10.0)
        x_norm = np.clip(x_norm, 0, 1)

        # Threshold decreases with x (more concepts active when exploring)
        # Base threshold at max x (fully exploring) is near 0
        # Base threshold at min x (fully focused) is near 1
        max_threshold = 0.9
        min_threshold = 0.1

        threshold = max_threshold - (max_threshold - min_threshold) * x_norm

        return threshold

    def should_explore_more(self) -> bool:
        """
        Decision rule for whether to explore more based on satisfaction.
        """
        return self.S > 0

    def get_exploration_bias(self) -> float:
        """
        Get current exploration bias based on satisfaction.

        Returns value in [0, 1] where 1 means maximum exploration.
        """
        # Map satisfaction to exploration bias
        # High satisfaction → high exploration bias
        exploration_bias = (self.S - self.S_min) / (self.S_max - self.S_min)
        exploration_bias = np.clip(exploration_bias, 0, 1)

        return exploration_bias

    def reset(self,
              S0: Optional[float] = None,
              x0: Optional[float] = None):
        """
        Reset dynamics to initial state.

        Args:
            S0: Initial satisfaction (if None, use config default)
            x0: Initial depth (if None, use 0)
        """
        if S0 is None:
            S0 = self.config.get('initialization', {}).get('S0_initial', 0.7)

        self.S = np.clip(S0, self.S_min, self.S_max)
        self.x = x0 if x0 is not None else 0.0

        # Clear history
        for key in self.history:
            self.history[key] = []

    def get_statistics(self) -> Dict:
        """Get dynamics statistics."""
        if len(self.history['S']) == 0:
            return {}

        stats = {
            'current_S': self.S,
            'current_x': self.x,
            'current_N': self._depth_to_N(self.x),
            'avg_S': np.mean(self.history['S'][-100:]) if len(self.history['S']) >= 100 else np.mean(self.history['S']),
            'avg_exploration': self.get_exploration_bias(),
            'exploration_phase': self.should_explore_more(),
            'history_length': len(self.history['S'])
        }

        return stats
