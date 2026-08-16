"""
Data preparation: ZINC250K → BRICS fragments → vocabulary

Steps:
1. Check if cached vocab exists, load and return if yes
2. Load ZINC250K SMILES list
3. BRICS decompose all molecules, count fragment frequencies
4. Filter fragments (≤4 APs, 2-20 heavy atoms)
5. Take top-200 by frequency
6. Expand vocab: emit (base_idx, ap_idx) pairs
7. Precompute ECFP4 fingerprints for all expanded entries
8. Build FragmentVocab object
9. Cache to disk and return
"""

from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass
from collections import Counter
import pickle

import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS, AllChem
from tqdm import tqdm


@dataclass
class FragmentVocab:
    """Fragment vocabulary with ECFP embeddings."""
    base_fragments: List[str]              # [V] canonical SMILES with dummy atoms
    expanded_entries: List[Tuple[int, int]]  # [V'] (base_frag_idx, ap_idx)
    ecfp_matrix: np.ndarray                # [V', 2048] ECFP4 fingerprints
    fragment_to_idx: Dict[str, int]        # SMILES → base vocab index
    ap_counts: List[int]                   # [V] number of APs per base fragment


def count_attachment_points(smiles: str) -> int:
    """Count BRICS dummy atoms ([*]) in fragment SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    # Dummy atoms have atomic number 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0)


def count_heavy_atoms(smiles: str) -> int:
    """Count non-hydrogen atoms."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return mol.GetNumHeavyAtoms()


def compute_ecfp4(smiles: str, nbits: int = 2048) -> np.ndarray:
    """Compute ECFP4 (Morgan radius=2, 2048-bit) fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nbits, dtype=np.float32)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def decompose_molecule(smiles: str) -> List[str]:
    """BRICS decompose molecule into fragments."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    try:
        fragments = BRICS.BRICSDecompose(mol)
        return list(fragments)
    except Exception:
        return []


def load_zinc250k(data_dir: Path, subset_size: int = None) -> List[str]:
    """
    Load ZINC250K dataset.

    Args:
        data_dir: Directory containing ZINC data
        subset_size: If provided, only load first N molecules (for testing)

    Returns:
        List of SMILES strings
    """
    zinc_file = data_dir / "zinc250k.smi"

    if not zinc_file.exists():
        print(f"WARNING: {zinc_file} not found.")
        print("Please download ZINC250K or create a test file.")
        print("For testing, creating dummy data...")
        return []

    smiles_list = []
    with open(zinc_file, 'r') as f:
        for i, line in enumerate(f):
            if subset_size and i >= subset_size:
                break
            smiles = line.strip().split()[0]
            smiles_list.append(smiles)

    return smiles_list


def build_base_vocabulary(
    smiles_list: List[str],
    max_ap: int = 4,
    min_heavy: int = 2,
    max_heavy: int = 20,
    top_k: int = 200
) -> Tuple[List[str], List[int]]:
    """
    Build base fragment vocabulary from molecules.

    Args:
        smiles_list: List of molecule SMILES
        max_ap: Maximum attachment points (filter out fragments with more)
        min_heavy: Minimum heavy atoms
        max_heavy: Maximum heavy atoms
        top_k: Keep top-K most frequent fragments

    Returns:
        (base_fragments, ap_counts) where base_fragments is list of SMILES
    """
    print("Decomposing molecules with BRICS...")
    fragment_counts = Counter()

    for smiles in tqdm(smiles_list, desc="BRICS decomposition"):
        fragments = decompose_molecule(smiles)
        fragment_counts.update(fragments)

    print(f"Found {len(fragment_counts)} unique fragments")

    # Filter by AP count and heavy atom count
    print(f"Filtering: ≤{max_ap} APs, {min_heavy}-{max_heavy} heavy atoms...")
    valid_fragments = []

    for frag_smiles, count in fragment_counts.items():
        n_ap = count_attachment_points(frag_smiles)
        n_heavy = count_heavy_atoms(frag_smiles)

        if n_ap > 0 and n_ap <= max_ap and min_heavy <= n_heavy <= max_heavy:
            valid_fragments.append((frag_smiles, count, n_ap))

    print(f"Valid fragments after filtering: {len(valid_fragments)}")

    # Sort by frequency, take top-K
    valid_fragments.sort(key=lambda x: x[1], reverse=True)
    top_fragments = valid_fragments[:top_k]

    base_fragments = [frag for frag, _, _ in top_fragments]
    ap_counts = [n_ap for _, _, n_ap in top_fragments]

    print(f"Base vocabulary size: {len(base_fragments)}")

    return base_fragments, ap_counts


def expand_vocabulary(
    base_fragments: List[str],
    ap_counts: List[int]
) -> List[Tuple[int, int]]:
    """
    Expand vocabulary: create (fragment, attachment_point) entries.

    Args:
        base_fragments: List of base fragment SMILES
        ap_counts: Number of APs per base fragment

    Returns:
        List of (base_idx, ap_idx) tuples
    """
    expanded = []

    for base_idx, n_ap in enumerate(ap_counts):
        for ap_idx in range(n_ap):
            expanded.append((base_idx, ap_idx))

    print(f"Expanded vocabulary size: {len(expanded)}")
    return expanded


def precompute_ecfp_embeddings(
    base_fragments: List[str],
    expanded_entries: List[Tuple[int, int]],
    nbits: int = 2048
) -> np.ndarray:
    """
    Precompute ECFP4 fingerprints for expanded vocabulary.

    Args:
        base_fragments: Base fragment SMILES
        expanded_entries: (base_idx, ap_idx) pairs
        nbits: Fingerprint size

    Returns:
        [V', nbits] array of fingerprints
    """
    print("Computing ECFP4 fingerprints...")
    ecfp_matrix = np.zeros((len(expanded_entries), nbits), dtype=np.float32)

    for i, (base_idx, _) in enumerate(tqdm(expanded_entries, desc="ECFP4")):
        frag_smiles = base_fragments[base_idx]
        ecfp_matrix[i] = compute_ecfp4(frag_smiles, nbits)

    return ecfp_matrix


def prepare_vocabulary(
    data_dir: Path,
    fragments_dir: Path,
    subset_size: int = None,
    top_k: int = 200,
    force_rebuild: bool = False
) -> FragmentVocab:
    """
    Main pipeline: load ZINC250K, build and cache vocabulary.

    Args:
        data_dir: Directory with raw ZINC data
        fragments_dir: Directory to cache vocabulary
        subset_size: If provided, only use first N molecules (for testing)
        top_k: Vocabulary size (top-K fragments)
        force_rebuild: If True, rebuild even if cache exists

    Returns:
        FragmentVocab object
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    fragments_dir.mkdir(parents=True, exist_ok=True)

    cache_file = fragments_dir / f"vocab_top{top_k}.pkl"

    # Step 1: Check cache
    if cache_file.exists() and not force_rebuild:
        print(f"Loading cached vocabulary from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print("Building vocabulary from scratch...")

    # Step 2: Load ZINC250K
    smiles_list = load_zinc250k(data_dir, subset_size)

    if len(smiles_list) == 0:
        print(f"WARNING: No SMILES loaded. Creating dummy vocab with {top_k} fragments for testing.")
        # Create synthetic test vocab with diverse fragments
        base_fragments = []
        ap_counts = []

        # Simple alkyl chains with 1-2 APs
        base_fragments.extend([
            "[*]C", "[*]CC", "[*]CCC", "[*]CCCC",
            "[*]C([*])C", "[*]CC([*])C"
        ])
        ap_counts.extend([1, 1, 1, 1, 2, 2])

        # Aromatic rings with 1-3 APs
        base_fragments.extend([
            "[*]c1ccccc1", "[*]c1ccc([*])cc1", "[*]c1cc([*])cc([*])c1"
        ])
        ap_counts.extend([1, 2, 3])

        # Heterocycles with 1-2 APs
        base_fragments.extend([
            "[*]c1ccncc1", "[*]c1cccnc1", "[*]c1nc([*])ccn1"
        ])
        ap_counts.extend([1, 1, 2])

        # Functional groups with 1-2 APs
        base_fragments.extend([
            "[*]CO", "[*]C(=O)C", "[*]C(=O)N([*])C", "[*]OC"
        ])
        ap_counts.extend([1, 1, 2, 1])

        # Cycloalkanes with 1-3 APs
        base_fragments.extend([
            "[*]C1CCCC1", "[*]C1([*])CCCC1", "[*]C1CCC([*])CC1"
        ])
        ap_counts.extend([1, 2, 2])

        # Additional diversity (if needed)
        while len(base_fragments) < top_k:
            # Add methylated variants
            idx = len(base_fragments) % 5
            base_fragments.append(f"[*]C({'C' * idx})C")
            ap_counts.append(1)

        # Trim to exact size
        base_fragments = base_fragments[:top_k]
        ap_counts = ap_counts[:top_k]

        print(f"Created {len(base_fragments)} synthetic fragments")
    else:
        # Steps 3-5: Build base vocabulary
        base_fragments, ap_counts = build_base_vocabulary(smiles_list, top_k=top_k)

    # Step 6: Expand vocabulary
    expanded_entries = expand_vocabulary(base_fragments, ap_counts)

    # Step 7: Precompute ECFP embeddings
    ecfp_matrix = precompute_ecfp_embeddings(base_fragments, expanded_entries)

    # Step 8: Build FragmentVocab object
    fragment_to_idx = {smiles: idx for idx, smiles in enumerate(base_fragments)}

    vocab = FragmentVocab(
        base_fragments=base_fragments,
        expanded_entries=expanded_entries,
        ecfp_matrix=ecfp_matrix,
        fragment_to_idx=fragment_to_idx,
        ap_counts=ap_counts
    )

    # Step 9: Cache to disk
    print(f"Caching vocabulary to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(vocab, f)

    return vocab


if __name__ == "__main__":
    # Example usage
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "raw"
    fragments_dir = project_root / "data" / "fragments"

    # For testing, use small subset
    vocab = prepare_vocabulary(
        data_dir=data_dir,
        fragments_dir=fragments_dir,
        subset_size=1000,  # Only 1000 molecules for quick test
        top_k=200
    )

    print(f"\nVocabulary statistics:")
    print(f"  Base fragments: {len(vocab.base_fragments)}")
    print(f"  Expanded entries: {len(vocab.expanded_entries)}")
    print(f"  ECFP matrix shape: {vocab.ecfp_matrix.shape}")
    print(f"\nFirst 5 base fragments:")
    for i in range(min(5, len(vocab.base_fragments))):
        print(f"  {i}: {vocab.base_fragments[i]} ({vocab.ap_counts[i]} APs)")
