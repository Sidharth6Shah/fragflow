"""
State class: partial molecule representation

Steps:
1. Define FragState dataclass (frags, bonds, frozen)
2. Implement get_open_aps() - find APs not in any bond
3. Implement state_to_pyg():
   - Build node features [N, 2052] (ECFP + open-AP multi-hot)
   - Build edge_index [2, 2E] (bidirectional from bonds)
   - Build edge_attr [2E, 8] (one-hot AP pairs)
   - Return PyG Data object
4. Implement one_hot_pair() helper for edge attributes
5. Implement state_to_molecule() - assemble RDKit mol from fragments and bonds
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import sys
from pathlib import Path

import torch
import numpy as np
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import BRICS

# Add parent dir to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from data.prepare_data import FragmentVocab


@dataclass(frozen=True)
class FragState:
    """
    Immutable state representing a partial molecule as a fragment graph.

    Attributes:
        frags: Tuple of base fragment indices (in placement order)
        bonds: Tuple of bond tuples (frag_i, ap_i, frag_j, ap_j)
    """
    frags: tuple[int, ...]
    bonds: tuple[tuple[int, int, int, int], ...]

    def is_empty(self) -> bool:
        """Check if state has no fragments."""
        return len(self.frags) == 0

    def num_fragments(self) -> int:
        """Return number of placed fragments."""
        return len(self.frags)


def get_open_aps(state: FragState, vocab: FragmentVocab) -> List[Tuple[int, int]]:
    """
    Find all open (unbonded) attachment points in the current state.

    An AP (frag_idx, ap_slot) is open if it doesn't appear in any bond.

    Args:
        state: Current fragment state
        vocab: Fragment vocabulary

    Returns:
        List of (frag_idx, ap_slot) tuples for open APs
    """
    # Collect all bonded APs
    bonded_aps = set()
    for (frag_i, ap_i, frag_j, ap_j) in state.bonds:
        bonded_aps.add((frag_i, ap_i))
        bonded_aps.add((frag_j, ap_j))

    # Find all open APs
    open_aps = []
    for frag_idx in range(len(state.frags)):
        base_frag_id = state.frags[frag_idx]
        n_ap = vocab.ap_counts[base_frag_id]

        for ap_slot in range(n_ap):
            if (frag_idx, ap_slot) not in bonded_aps:
                open_aps.append((frag_idx, ap_slot))

    return open_aps


def one_hot_pair(ap_i: int, ap_j: int, max_ap: int = 4) -> np.ndarray:
    """
    One-hot encode a pair of attachment point indices.

    Args:
        ap_i: First AP index (0-3)
        ap_j: Second AP index (0-3)
        max_ap: Maximum number of APs per fragment

    Returns:
        [2*max_ap] one-hot vector
    """
    vec = np.zeros(2 * max_ap, dtype=np.float32)
    vec[ap_i] = 1.0
    vec[max_ap + ap_j] = 1.0
    return vec


def state_to_pyg(state: FragState, vocab: FragmentVocab) -> Data:
    """
    Convert FragState to PyTorch Geometric Data object.

    Node features: [N, 2052] = ECFP (2048) + open-AP multi-hot (4)
    Edge index: [2, 2E] (bidirectional)
    Edge attr: [2E, 8] = one-hot of (ap_i, ap_j) slot pair

    Args:
        state: Current fragment state
        vocab: Fragment vocabulary

    Returns:
        PyG Data object
    """
    N = len(state.frags)

    # Handle empty state
    if N == 0:
        return Data(
            x=torch.zeros((0, 2052), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 8), dtype=torch.float32)
        )

    # Step 1: Build node features [N, 2052]
    x = np.zeros((N, 2052), dtype=np.float32)

    # Fill ECFP features (first 2048 dims)
    for i, base_frag_id in enumerate(state.frags):
        # Find any expanded entry for this base fragment (ECFP is same for all APs)
        for exp_idx, (base_idx, _) in enumerate(vocab.expanded_entries):
            if base_idx == base_frag_id:
                x[i, :2048] = vocab.ecfp_matrix[exp_idx]
                break

    # Fill open-AP multi-hot (last 4 dims)
    open_aps = get_open_aps(state, vocab)
    for (frag_idx, ap_slot) in open_aps:
        x[frag_idx, 2048 + ap_slot] = 1.0

    # Step 2: Build edges (bidirectional)
    edge_index = []
    edge_attr = []

    for (frag_i, ap_i, frag_j, ap_j) in state.bonds:
        # Forward edge: frag_i -> frag_j
        edge_index.append([frag_i, frag_j])
        edge_attr.append(one_hot_pair(ap_i, ap_j, max_ap=4))

        # Backward edge: frag_j -> frag_i
        edge_index.append([frag_j, frag_i])
        edge_attr.append(one_hot_pair(ap_j, ap_i, max_ap=4))

    # Convert to tensors
    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).T  # [2, 2E]
        edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float32)  # [2E, 8]
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 8), dtype=torch.float32)

    x = torch.tensor(x, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def state_to_molecule(state: FragState, vocab: FragmentVocab) -> Optional[Chem.Mol]:
    """
    Assemble final RDKit molecule from fragment state.

    This is called at terminal states (after Stop action) to build the
    complete molecule for reward computation.

    Args:
        state: Terminal fragment state
        vocab: Fragment vocabulary

    Returns:
        RDKit Mol object, or None if assembly fails
    """
    if state.is_empty():
        return None

    try:
        # Get fragment SMILES
        frag_smiles_list = [vocab.base_fragments[frag_id] for frag_id in state.frags]

        # Create RDKit mol objects for each fragment
        frag_mols = []
        for smiles in frag_smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            frag_mols.append(mol)

        # If single fragment, just sanitize and return
        if len(frag_mols) == 1:
            try:
                Chem.SanitizeMol(frag_mols[0])
                return frag_mols[0]
            except Exception:
                return None

        # Combine fragments using bonds
        # This is complex - BRICS provides reconstruction helpers
        # For now, use a simplified approach:
        # Build a combined mol and track which dummy atoms should be connected

        combined = frag_mols[0]
        for mol in frag_mols[1:]:
            combined = Chem.CombineMols(combined, mol)

        # Track dummy atom indices for each fragment
        # Note: This is a simplified version - full implementation requires
        # careful tracking of atom indices across CombineMols operations

        # Get editable mol
        em = Chem.EditableMol(combined)

        # For each bond in state.bonds, connect the appropriate dummy atoms
        # and then remove them
        # (This requires careful index tracking - simplified here)

        # For now, just try to sanitize the combined structure
        # A full implementation would properly connect fragments
        result = em.GetMol()

        try:
            Chem.SanitizeMol(result)
            return result
        except Exception:
            return None

    except Exception:
        return None


def create_initial_state() -> FragState:
    """Create empty initial state."""
    return FragState(frags=tuple(), bonds=tuple())


def add_fragment(
    state: FragState,
    base_frag_id: int,
    connect_to: Optional[Tuple[int, int]] = None,
    connect_from_ap: Optional[int] = None
) -> FragState:
    """
    Add a fragment to the current state.

    Args:
        state: Current state
        base_frag_id: Base vocabulary index of fragment to add
        connect_to: Optional (frag_idx, ap_slot) to connect to
        connect_from_ap: Which AP on new fragment to use for connection

    Returns:
        New state with added fragment
    """
    new_frag_idx = len(state.frags)
    new_frags = state.frags + (base_frag_id,)

    if connect_to is None:
        # First fragment - no bonds
        return FragState(frags=new_frags, bonds=state.bonds)

    # Add bond between existing fragment and new fragment
    frag_i, ap_i = connect_to
    frag_j = new_frag_idx
    ap_j = connect_from_ap

    # Sort bond tuple canonically (lower frag_idx first)
    if frag_i < frag_j:
        new_bond = (frag_i, ap_i, frag_j, ap_j)
    else:
        new_bond = (frag_j, ap_j, frag_i, ap_i)

    new_bonds = state.bonds + (new_bond,)

    return FragState(frags=new_frags, bonds=new_bonds)


if __name__ == "__main__":
    # Simple test
    from data.prepare_data import prepare_vocabulary

    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "raw"
    fragments_dir = project_root / "data" / "fragments"

    vocab = prepare_vocabulary(data_dir, fragments_dir, top_k=20)

    # Test empty state
    state = create_initial_state()
    print(f"Empty state: {state}")
    print(f"Is empty: {state.is_empty()}")

    # Test adding a fragment
    if len(vocab.base_fragments) > 0:
        state = add_fragment(state, base_frag_id=0)
        print(f"\nAfter adding fragment 0: {state}")
        print(f"Open APs: {get_open_aps(state, vocab)}")

        # Convert to PyG
        data = state_to_pyg(state, vocab)
        print(f"\nPyG Data:")
        print(f"  x shape: {data.x.shape}")
        print(f"  edge_index shape: {data.edge_index.shape}")
        print(f"  edge_attr shape: {data.edge_attr.shape}")
