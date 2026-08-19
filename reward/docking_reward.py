"""
Docking-based reward for target protein binding.

This module provides reward functions based on predicted binding affinity
to a specific protein target.

IMPORTANT: Requires external tools (AutoDock Vina, protein structures)
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from rdkit.Contrib.SA_Score import sascorer


def compute_docking_reward(
    mol: Chem.Mol,
    beta: float = 1.0,
    target_protein: str = "EGFR",
    docking_weight: float = 0.5
) -> float:
    """
    Compute reward based on predicted protein binding affinity.

    This is a PLACEHOLDER implementation. For real use:
    1. Install AutoDock Vina or similar docking tool
    2. Prepare protein structure (PDB file)
    3. Train a surrogate model to predict docking scores
       (e.g., GNN trained on ChEMBL binding data)

    Args:
        mol: RDKit molecule
        beta: Temperature parameter
        target_protein: Protein target name (for lookup)
        docking_weight: Weight for docking score (0-1)

    Returns:
        Log-space reward
    """
    try:
        # Base drug-likeness properties
        q = QED.qed(mol)
        sa_score = sascorer.calculateScore(mol)
        s = np.clip((10 - sa_score) / 9.0, 0.0, 1.0)
        logp = Descriptors.MolLogP(mol)
        l = np.exp(-0.5 * ((logp - 2.5) / 2.0) ** 2)

        base_reward = 0.5 * q + 0.3 * s + 0.2 * l

        # Docking score prediction (PLACEHOLDER)
        # In real implementation:
        # docking_score = predict_binding_affinity(mol, target_protein)
        # affinity_normalized = sigmoid(docking_score)

        # Placeholder: Use molecular weight as proxy
        # (smaller molecules often have better binding efficiency)
        mw = Descriptors.MolWt(mol)
        mw_normalized = np.exp(-0.001 * (mw - 300) ** 2)  # Peak at 300 Da

        affinity_normalized = mw_normalized

        # Combine base reward + docking score
        R_raw = (1.0 - docking_weight) * base_reward + docking_weight * affinity_normalized

        # Log-space reward
        logR = beta * np.log(R_raw + 1e-4)

        return logR

    except Exception:
        return beta * np.log(1e-4)


def train_docking_surrogate():
    """
    Train a surrogate model to predict docking scores.

    This avoids running expensive docking for every molecule during training.

    Steps:
    1. Collect training data:
       - Download ChEMBL binding data for target protein
       - Or: Run docking on 10k-100k diverse molecules
    2. Train GNN to predict docking score from molecule
    3. Save model for use during GFlowNet training

    Dataset sources:
    - ChEMBL: https://www.ebi.ac.uk/chembl/
    - PDBbind: http://www.pdbbind.org.cn/
    - BindingDB: https://www.bindingdb.org/

    Model architectures:
    - MPNN (Message Passing Neural Network)
    - GCN (Graph Convolutional Network)
    - SchNet, DimeNet (3D geometry-aware)
    """
    raise NotImplementedError(
        "Surrogate model training requires external data. "
        "See docstring for implementation steps."
    )


def run_autodock_vina(mol_smiles: str, protein_pdb: str, exhaustiveness: int = 8):
    """
    Run AutoDock Vina docking simulation.

    Args:
        mol_smiles: Molecule SMILES
        protein_pdb: Path to protein structure (PDB file)
        exhaustiveness: Vina exhaustiveness parameter

    Returns:
        Docking score (kcal/mol, lower = better)

    Installation:
        conda install -c conda-forge vina

    Example:
        >>> score = run_autodock_vina("CCc1ccccc1", "protein.pdb")
        >>> print(f"Docking score: {score:.2f} kcal/mol")
    """
    raise NotImplementedError(
        "AutoDock Vina integration requires:\n"
        "1. Install Vina: conda install -c conda-forge vina\n"
        "2. Prepare protein structure (PDB + pdbqt conversion)\n"
        "3. Define binding site coordinates\n"
        "See: https://autodock-vina.readthedocs.io/"
    )


# Example protein targets with PDB IDs
PROTEIN_TARGETS = {
    "EGFR": "1M17",  # Epidermal Growth Factor Receptor
    "BCL2": "4LVT",  # B-cell lymphoma 2
    "CDK2": "1HCK",  # Cyclin-dependent kinase 2
    "HIV_PR": "3NU3",  # HIV-1 Protease
    "ACE2": "6M0J",  # SARS-CoV-2 target
}


if __name__ == "__main__":
    print("=" * 80)
    print("Docking-Based Reward (Placeholder)")
    print("=" * 80)

    print("\nThis module provides structure for protein binding rewards.")
    print("Full implementation requires:")
    print("  1. AutoDock Vina or similar docking tool")
    print("  2. Protein structures (PDB files)")
    print("  3. Surrogate model training on binding data")

    print("\nAvailable protein targets:")
    for name, pdb_id in PROTEIN_TARGETS.items():
        print(f"  {name:10s}: PDB {pdb_id}")

    print("\nFor quick start:")
    print("  - Use ChEMBL pre-trained binding affinity models")
    print("  - Or train surrogate on your target protein")
    print("=" * 80)
