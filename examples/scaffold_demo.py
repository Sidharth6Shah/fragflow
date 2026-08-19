"""
Demo: Scaffold-constrained molecule generation.

Shows how to train and sample molecules containing a specific scaffold.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from reward.reward_fn import compute_reward
from data.prepare_data import prepare_vocabulary
from env.molecule_state import create_initial_state, add_fragment


def demo_scaffold_reward():
    """Demo scaffold-constrained reward function."""

    print("=" * 80)
    print("Scaffold-Constrained Generation Demo")
    print("=" * 80)

    # Load vocab
    project_root = Path(__file__).parent.parent
    vocab = prepare_vocabulary(
        project_root / "data" / "raw",
        project_root / "data" / "fragments",
        top_k=200
    )

    # Create a simple test state (benzene derivative)
    state = create_initial_state()

    # Find benzene fragment in vocab
    benzene_idx = None
    for i, smiles in enumerate(vocab.base_fragments):
        if "c1ccccc1" in smiles and smiles.count("[*]") == 1:
            benzene_idx = i
            break

    if benzene_idx is not None:
        state = add_fragment(state, base_frag_id=benzene_idx)

        # Test different scaffold patterns
        print("\nTest molecule: benzene with one attachment point")
        print(f"Fragment SMILES: {vocab.base_fragments[benzene_idx]}")

        scaffolds = [
            ("Benzene ring", "c1ccccc1"),
            ("Pyridine ring", "c1ccncc1"),
            ("Any aromatic", "c1ccccc1,c1ccncc1"),
        ]

        print("\nReward with different scaffold constraints:")
        print("-" * 80)

        for name, smarts in scaffolds:
            reward = compute_reward(
                state, vocab,
                beta=4.0,
                mode="scaffold",
                scaffold_smarts=smarts,
                scaffold_weight=0.4
            )
            print(f"  {name:20s}: {reward:7.3f}")

        # Compare with base reward (no scaffold)
        base_reward = compute_reward(state, vocab, beta=4.0, mode="full")
        print(f"\n  Base reward (no scaffold): {base_reward:7.3f}")
    else:
        print("No benzene fragment found in vocab")

    print("\n" + "=" * 80)
    print("Training Configs Available:")
    print("=" * 80)
    print("\n1. FULL_CONFIG: Generic drug-like molecules")
    print("   - Optimizes QED, SA, LogP")
    print("   - No structural constraints")

    print("\n2. SCAFFOLD_CONFIG: Benzene-containing molecules")
    print("   - Requires benzene ring (c1ccccc1)")
    print("   - 40% weight on scaffold presence")
    print("   - Still optimizes drug-likeness")

    print("\nTo train with scaffold constraints:")
    print("  python training/train_scaffold.py")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    demo_scaffold_reward()
