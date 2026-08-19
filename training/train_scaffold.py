"""
Train GFlowNet with scaffold constraints.

This script trains a model to generate molecules containing a specific scaffold.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from training.train import GFlowNetTrainer
from training.config import SCAFFOLD_CONFIG


if __name__ == "__main__":
    print("=" * 80)
    print("Training FragFlow with Scaffold Constraints")
    print("=" * 80)
    print(f"\nScaffold: {SCAFFOLD_CONFIG.scaffold_smarts}")
    print(f"Scaffold weight: {SCAFFOLD_CONFIG.scaffold_weight}")
    print(f"Beta: {SCAFFOLD_CONFIG.beta}")
    print(f"Iterations: {SCAFFOLD_CONFIG.num_iterations}")
    print("\n" + "=" * 80 + "\n")

    trainer = GFlowNetTrainer(SCAFFOLD_CONFIG)
    trainer.train()
