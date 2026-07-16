"""
Reward function: finished molecule → QED/SA/LogP → scalar reward

Two modes:
- Dummy: constant reward for testing
- Full: QED + SA + LogP combination
"""

import torch
import numpy as np
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from rdkit.Contrib.SA_Score import sascorer

sys.path.append(str(Path(__file__).parent.parent))
from env.molecule_state import FragState, state_to_molecule
from data.prepare_data import FragmentVocab


def compute_reward(
    state: FragState,
    vocab: FragmentVocab,
    beta: float = 1.0,
    mode: str = "dummy"
) -> float:
    """
    Compute reward for terminal state.

    Args:
        state: Terminal fragment state
        vocab: Fragment vocabulary
        beta: Temperature parameter
        mode: "dummy" or "full"

    Returns:
        Log-space reward for TB
    """
    # Assemble molecule
    mol = state_to_molecule(state, vocab)

    # Invalid molecule
    if mol is None:
        return beta * np.log(1e-4)

    # Dummy mode: constant reward
    if mode == "dummy":
        return 1.0  # Simple constant for testing

    # Full mode: QED + SA + LogP
    return compute_full_reward(mol, beta)


def compute_full_reward(mol: Chem.Mol, beta: float = 1.0) -> float:
    """
    Full reward: QED + SA + LogP combination.

    Args:
        mol: RDKit molecule
        beta: Temperature parameter

    Returns:
        Log-space reward
    """
    try:
        # QED: [0, 1], higher better
        q = QED.qed(mol)

        # SA Score: [1, 10], lower better → invert to [0, 1]
        sa_score = sascorer.calculateScore(mol)
        s = np.clip((10 - sa_score) / 9.0, 0.0, 1.0)

        # LogP: Gaussian window around target 2.5
        logp = Descriptors.MolLogP(mol)
        l = np.exp(-0.5 * ((logp - 2.5) / 2.0) ** 2)

        # Weighted combination
        R_raw = 0.5 * q + 0.3 * s + 0.2 * l

        # Log-space reward
        logR = beta * np.log(R_raw + 1e-4)

        return logR

    except Exception:
        # Error in reward computation
        return beta * np.log(1e-4)


if __name__ == "__main__":
    from data.prepare_data import prepare_vocabulary
    from env.molecule_state import create_initial_state, add_fragment

    project_root = Path(__file__).parent.parent
    vocab = prepare_vocabulary(
        project_root / "data" / "raw",
        project_root / "data" / "fragments",
        top_k=20
    )

    # Test dummy reward
    state = create_initial_state()
    if len(vocab.base_fragments) > 0:
        state = add_fragment(state, base_frag_id=0)

        reward = compute_reward(state, vocab, beta=1.0, mode="dummy")
        print(f"Dummy reward: {reward}")

        # Test full reward (may fail if state_to_molecule is incomplete)
        try:
            reward = compute_reward(state, vocab, beta=1.0, mode="full")
            print(f"Full reward: {reward}")
        except Exception as e:
            print(f"Full reward failed (expected): {e}")
