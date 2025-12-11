"""
Basic tests for the GEMMA Novelty Framework.
"""
import pytest
import torch
import numpy as np
import yaml
from pathlib import Path

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.model import BeliefUpdateModel
from core.memory import ConceptMemory
from core.dynamics import SatisfactionDynamics

def test_config_loading():
    """Test that configuration files can be loaded."""
    config_path = Path(__file__).parent.parent / "config" / "hyperparameters.yaml"
    assert config_path.exists(), "Config file not found"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    assert 'model' in config
    assert 'memory' in config
    assert 'learning_rates' in config

def test_model_initialization():
    """Test that the main model can be initialized."""
    config = {
        'model': {'embedding_dim': 128},
        'memory': {'decay_factor': 0.999, 'memory_size': 1000},
        'initialization': {'k_t_initial': 0.5, 'eta_t_initial': 0.5}
    }

    model = BeliefUpdateModel(config)
    assert isinstance(model.k_t, torch.nn.Parameter)
    assert isinstance(model.eta_t, torch.nn.Parameter)
    assert model.k_t.item() == pytest.approx(0.5)
    assert model.eta_t.item() == pytest.approx(0.5)

def test_memory_system():
    """Test basic memory operations."""
    config = {
        'memory': {'decay_factor': 0.9, 'memory_size': 100},
        'memory_config': {'cache_ttl': 3600}
    }

    memory = ConceptMemory(config)
    # Test adding concepts
    vec1 = torch.randn(128)
    vec2 = torch.randn(128)
    memory.update_memory('concept1', vec1)
    memory.update_memory('concept2', vec2)
    assert len(memory.memory_vectors) == 2
    assert 'concept1' in memory.memory_vectors
    assert 'concept2' in memory.memory_vectors

def test_satisfaction_dynamics():
    """Test satisfaction dynamics."""
    config = {
        'satisfaction': {
            'lambda_S': 0.1,
            'rho': 0.4,
            'kappa': 0.3,
            'tau': 0.3,
            'S_min': 0.1,
            'S_max': 1.0
        },
        'exploration': {
            'depth_increment': 1.0,
            'depth_decrement': 1.0
        },
        'dynamics': {
            'N_min': 5,
            'N_max': 50
        },
        'initialization': {
            'S0_initial': 0.7
        }
    }

    dynamics = SatisfactionDynamics(config)
    # Test initial state
    assert dynamics.S == pytest.approx(0.7)
    assert dynamics.x == 0.0
    # Test update
    S_new, x_new, N_new = dynamics.update(0.8, 0.7, 0.9)
    assert S_min <= S_new <= S_max
    assert x_new >= 0
    assert N_min <= N_new <= N_max

def test_model_forward():
    """Test model forward pass."""
    config = {
        'model': {'embedding_dim': 64},
        'memory': {'decay_factor': 0.999},
        'initialization': {'k_t_initial': 0.5}
    }

    model = BeliefUpdateModel(config)
    # Create dummy inputs
    batch_size = 4
    embedding_dim = 64
    input_embeddings = torch.randn(batch_size, embedding_dim)
    retrieved_embeddings = torch.randn(batch_size, embedding_dim)
    # Mock memory system
    class MockMemory:
        def get_canonical(self, vec):
            return torch.zeros_like(vec)
    memory_system = MockMemory()
    # Mock active concepts
    active_concepts = {
        'vectors': [torch.randn(embedding_dim) for _ in range(10)],
        'previous_vectors': [torch.randn(embedding_dim) for _ in range(10)]
    }
    # Forward pass
    B_t, metrics = model(input_embeddings, retrieved_embeddings,
                         memory_system, active_concepts)
    assert B_t.shape == ()
    assert isinstance(B_t, torch.Tensor)
    assert 'input_novelty' in metrics
    assert 'C_total' in metrics

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
