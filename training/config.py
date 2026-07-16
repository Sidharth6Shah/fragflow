"""
Training hyperparameters

Two configs:
- Milestone: tiny vocab, dummy reward, sanity check
- Full: full vocab, real reward, production training
"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Model
    d: int = 128  # embedding dimension
    num_gnn_layers: int = 3
    max_ap: int = 4

    # Vocab
    vocab_size: int = 200  # top-K fragments
    max_frags: int = 8  # N_max

    # Training
    lr: float = 1e-3
    batch_size: int = 32
    num_iterations: int = 10000

    # Reward
    beta: float = 1.0  # temperature
    reward_mode: str = "dummy"  # "dummy" or "full"

    # Logging
    log_every: int = 100
    wandb_project: str = "fragflow"
    wandb_enabled: bool = False

    # Checkpoint
    save_every: int = 1000
    checkpoint_dir: str = "checkpoints"


# Milestone config: sanity check
MILESTONE_CONFIG = TrainingConfig(
    vocab_size=20,
    max_frags=4,
    batch_size=16,
    num_iterations=5000,
    reward_mode="dummy",
    wandb_enabled=False
)

# Full training config
FULL_CONFIG = TrainingConfig(
    vocab_size=200,
    max_frags=8,
    batch_size=32,
    num_iterations=50000,
    reward_mode="full",
    wandb_enabled=True
)
