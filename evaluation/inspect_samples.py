"""
Quick script to inspect generated molecules.
"""

import pickle
import sys
from pathlib import Path
from rdkit import Chem

sys.path.append(str(Path(__file__).parent.parent))


def main():
    samples_path = Path("evaluation/sampled_molecules.pkl")

    print(f"Loading samples from {samples_path}")
    with open(samples_path, 'rb') as f:
        samples = pickle.load(f)

    print(f"\nInspecting first 10 valid molecules:")
    print("=" * 80)

    valid_count = 0
    for i, (state, mol, reward) in enumerate(samples):
        if mol is not None and valid_count < 10:
            smiles = Chem.MolToSmiles(mol)
            num_atoms = mol.GetNumAtoms()
            num_bonds = mol.GetNumBonds()
            num_frags = len(state.frags)

            print(f"\n{valid_count + 1}. SMILES: {smiles}")
            print(f"   Atoms: {num_atoms}, Bonds: {num_bonds}, Fragments: {num_frags}")
            print(f"   Reward: {reward:.3f}")

            valid_count += 1

    print(f"\n{'=' * 80}")
    print(f"Total samples: {len(samples)}")
    print(f"Valid: {sum(1 for _, mol, _ in samples if mol is not None)}")
    print(f"Invalid: {sum(1 for _, mol, _ in samples if mol is None)}")


if __name__ == "__main__":
    main()
