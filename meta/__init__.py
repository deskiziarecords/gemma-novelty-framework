"""
Meta-learning module for dynamic optimization of loss weights and novelty preferences.
"""
from .meta_objective import MetaObjective
from .meta_optimizer import MetaOptimizer
__all__ = [
    'MetaObjective',
    'MetaOptimizer'
]
