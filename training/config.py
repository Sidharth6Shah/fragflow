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
    reward_mode: str = "dummy"  # "dummy", "full", or "scaffold"
    scaffold_smarts: str = None  # SMARTS pattern for scaffold constraint
    scaffold_weight: float = 0.3  # Weight for scaffold bonus

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

# Full training config (v2 with gradient clipping)
FULL_CONFIG = TrainingConfig(
    vocab_size=200,
    max_frags=8,
    batch_size=16,  # Reduced from 32 for faster iteration
    num_iterations=10000,  # Reduced from 50000 for initial training
    reward_mode="full",
    beta=4.0,  # Increased from 1.0 for better exploration
    save_every=500,  # More frequent checkpoints for shorter training
    wandb_enabled=False  # Enable if wandb configured
)

# Scaffold-constrained config: Generate molecules containing benzene ring
SCAFFOLD_CONFIG = TrainingConfig(
    vocab_size=200,
    max_frags=8,
    batch_size=32,
    num_iterations=20000,
    reward_mode="scaffold",
    beta=4.0,
    scaffold_smarts="c1ccccc1",  # Benzene ring
    scaffold_weight=0.4,  # 40% weight on scaffold presence
    wandb_enabled=False
)
