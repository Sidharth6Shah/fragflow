"""
Evaluation metrics for generated molecules.

Computes diversity, validity, and reward distribution statistics.
"""

import pickle
import sys
from pathlib import Path
from typing import List, Tuple
from collections import Counter
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

sys.path.append(str(Path(__file__).parent.parent))


def compute_validity(samples: List[Tuple]) -> float:
    """
    Compute fraction of valid molecules.

    Args:
        samples: List of (state, mol, reward) tuples (mol is RDKit Mol object)

    Returns:
        Validity rate (0-1)
    """
    valid_count = sum(1 for _, mol, _ in samples if mol is not None)
    return valid_count / len(samples) if samples else 0.0


def compute_uniqueness(samples: List[Tuple]) -> float:
    """
    Compute fraction of unique molecules.

    Args:
        samples: List of (state, mol, reward) tuples (mol is RDKit Mol object)

    Returns:
        Uniqueness rate (0-1)
    """
    smiles_list = []
    for _, mol, _ in samples:
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)
            if smiles:
                smiles_list.append(smiles)

    if not smiles_list:
        return 0.0

    unique_smiles = set(smiles_list)
    return len(unique_smiles) / len(smiles_list)


def compute_diversity(samples: List[Tuple], sample_size: int = 1000) -> float:
    """
    Compute average Tanimoto distance between molecules.

    Uses Morgan fingerprints (ECFP4).

    Args:
        samples: List of (state, mol, reward) tuples (mol is RDKit Mol object)
        sample_size: Number of pairs to sample (for efficiency)

    Returns:
        Average pairwise Tanimoto distance (0-1)
    """
    # Get valid unique molecules (convert to SMILES for deduplication)
    mol_dict = {}
    for _, mol, _ in samples:
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)
            if smiles:
                mol_dict[smiles] = mol

    if len(mol_dict) < 2:
        return 0.0

    # Compute fingerprints
    fps = []
    for mol in mol_dict.values():
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fps.append(fp)

    if len(fps) < 2:
        return 0.0

    # Sample pairs to avoid O(n^2) computation
    distances = []
    np.random.seed(42)

    num_pairs = min(sample_size, len(fps) * (len(fps) - 1) // 2)

    for _ in range(num_pairs):
        i, j = np.random.choice(len(fps), size=2, replace=False)
        similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
        distance = 1.0 - similarity
        distances.append(distance)

    return np.mean(distances)


def compute_reward_stats(samples: List[Tuple]) -> dict:
    """
    Compute reward distribution statistics.

    Args:
        samples: List of (state, smiles, reward) tuples

    Returns:
        Dict with mean, std, min, max, median
    """
    rewards = [reward for _, _, reward in samples if reward is not None]

    if not rewards:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0
        }

    return {
        'mean': np.mean(rewards),
        'std': np.std(rewards),
        'min': np.min(rewards),
        'max': np.max(rewards),
        'median': np.median(rewards)
    }


def compute_length_stats(samples: List[Tuple]) -> dict:
    """
    Compute molecule length (number of fragments) statistics.

    Args:
        samples: List of (state, smiles, reward) tuples

    Returns:
        Dict with mean, std, min, max, median
    """
    lengths = [len(state.frags) for state, _, _ in samples]

    return {
        'mean': np.mean(lengths),
        'std': np.std(lengths),
        'min': np.min(lengths),
        'max': np.max(lengths),
        'median': np.median(lengths),
        'distribution': dict(Counter(lengths))
    }


def evaluate_samples(samples: List[Tuple], verbose: bool = True) -> dict:
    """
    Compute all evaluation metrics.

    Args:
        samples: List of (state, smiles, reward) tuples
        verbose: Print results

    Returns:
        Dict with all metrics
    """
    metrics = {
        'num_samples': len(samples),
        'validity': compute_validity(samples),
        'uniqueness': compute_uniqueness(samples),
        'diversity': compute_diversity(samples),
        'reward_stats': compute_reward_stats(samples),
        'length_stats': compute_length_stats(samples)
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluation Metrics")
        print(f"{'='*60}")
        print(f"Total samples: {metrics['num_samples']}")
        print(f"\nValidity: {metrics['validity']:.3f}")
        print(f"Uniqueness: {metrics['uniqueness']:.3f}")
        print(f"Diversity (avg Tanimoto dist): {metrics['diversity']:.3f}")

        print(f"\nReward Statistics:")
        for key, val in metrics['reward_stats'].items():
            print(f"  {key:8s}: {val:7.3f}")

        print(f"\nMolecule Length (# fragments):")
        for key, val in metrics['length_stats'].items():
            if key != 'distribution':
                print(f"  {key:8s}: {val:7.2f}")

        print(f"\nLength Distribution:")
        for length, count in sorted(metrics['length_stats']['distribution'].items()):
            pct = 100 * count / metrics['num_samples']
            print(f"  {length} fragments: {count:4d} ({pct:5.1f}%)")

        print(f"{'='*60}\n")

    return metrics


def main():
    """Load sampled molecules and compute metrics."""
    samples_path = Path("evaluation/sampled_molecules.pkl")

    if not samples_path.exists():
        print(f"Error: {samples_path} not found")
        print("Run evaluation/sample.py first to generate samples")
        return

    # Load samples
    print(f"Loading samples from {samples_path}")
    with open(samples_path, 'rb') as f:
        samples = pickle.load(f)

    # Compute metrics
    metrics = evaluate_samples(samples, verbose=True)

    # Save metrics
    metrics_path = Path("evaluation/metrics.pkl")
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)

    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
