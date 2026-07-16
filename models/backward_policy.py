"""
Backward policy P_B

Samples which fragment to remove (inverse of Add action).
Only removable leaves (degree-1 fragments) are valid.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from env.molecule_state import FragState


class BackwardPolicy(nn.Module):
    """
    Backward policy for fragment removal.

    Computes log-probs over which fragment to remove.
    Only degree-1 fragments (leaves) can be removed.
    """

    def __init__(self, d: int = 128):
        """
        Initialize backward policy.

        Args:
            d: Hidden dimension
        """
        super().__init__()

        self.d = d

        # MLP: h_v → logit
        self.mlp = nn.Sequential(
            nn.Linear(d, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def get_removable_leaves(self, state: FragState) -> list:
        """
        Find fragments that can be removed (degree-1 leaves).

        A fragment is removable if it has degree 1 in the fragment graph.

        Args:
            state: Current state

        Returns:
            List of fragment indices that can be removed
        """
        N = len(state.frags)

        if N == 0:
            return []

        if N == 1:
            # Single fragment - parent is source (deterministic)
            return [0]

        # Count degree for each fragment
        degree = [0] * N

        for (frag_i, _, frag_j, _) in state.bonds:
            degree[frag_i] += 1
            degree[frag_j] += 1

        # Leaves have degree 1
        leaves = [i for i in range(N) if degree[i] == 1]

        return leaves

    def forward(self, h_v: torch.Tensor, state: FragState):
        """
        Compute backward policy log-probs.

        Args:
            h_v: [N, d] node embeddings
            state: Current state

        Returns:
            log_probs: Log probabilities over removable fragments
            num_actions: Number of valid actions
        """
        N = h_v.size(0)

        if N == 0:
            # Empty state - no backward actions
            return torch.zeros(0, device=h_v.device), 0

        # Get removable leaves
        leaves = self.get_removable_leaves(state)

        if len(leaves) == 0:
            # No removable fragments
            return torch.zeros(0, device=h_v.device), 0

        if len(leaves) == 1 and N == 1:
            # Single fragment - deterministic backward (P_B = 1)
            return torch.zeros(1, device=h_v.device), 1

        # Compute logits for all fragments
        logits = self.mlp(h_v).squeeze(-1)  # [N]

        # Mask to only leaves
        mask = torch.zeros(N, dtype=torch.bool, device=h_v.device)
        for leaf_idx in leaves:
            mask[leaf_idx] = True

        # Extract leaf logits
        leaf_logits = logits[mask]  # [num_leaves]

        # Log-softmax
        log_probs = F.log_softmax(leaf_logits, dim=0)

        return log_probs, len(leaves)


if __name__ == "__main__":
    from data.prepare_data import prepare_vocabulary
    from env.molecule_state import create_initial_state, add_fragment, state_to_pyg
    from models.state_encoder import StateEncoder

    project_root = Path(__file__).parent.parent
    vocab = prepare_vocabulary(
        project_root / "data" / "raw",
        project_root / "data" / "fragments",
        top_k=20
    )

    encoder = StateEncoder(d=128)
    policy = BackwardPolicy(d=128)

    # Test with empty state
    state = create_initial_state()
    data = state_to_pyg(state, vocab)
    h_v, h_G = encoder(data)
    log_probs, num_actions = policy(h_v, state)
    print(f"Empty state:")
    print(f"  Num actions: {num_actions} (should be 0)")

    # Test with one fragment
    if len(vocab.base_fragments) > 0:
        state = add_fragment(state, base_frag_id=0)
        data = state_to_pyg(state, vocab)
        h_v, h_G = encoder(data)
        log_probs, num_actions = policy(h_v, state)
        print(f"\nOne fragment:")
        print(f"  Num actions: {num_actions} (should be 1, deterministic)")

        # Test with two fragments
        if vocab.ap_counts[0] > 0:
            state = add_fragment(
                state,
                base_frag_id=0,
                connect_to=(0, 0),
                connect_from_ap=0
            )
            data = state_to_pyg(state, vocab)
            h_v, h_G = encoder(data)
            log_probs, num_actions = policy(h_v, state)
            leaves = policy.get_removable_leaves(state)
            print(f"\nTwo fragments:")
            print(f"  Num actions: {num_actions}")
            print(f"  Removable leaves: {leaves}")
