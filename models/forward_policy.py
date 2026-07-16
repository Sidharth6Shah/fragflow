"""
Forward policy P_F

Factored action space: Stop or Add (focus_ap, block)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from env.molecule_state import FragState, get_open_aps
from data.prepare_data import FragmentVocab


class ForwardPolicy(nn.Module):
    """
    Forward policy for fragment assembly.

    Three heads:
    - Stop: MLP(h_G) → scalar
    - Add: MLP([h_ap; block_emb; h_G]) → [num_AP, V']
    - Source (empty state): Linear(h_G) → [V]
    """

    def __init__(self, d: int = 128):
        """
        Initialize forward policy.

        Args:
            d: Hidden dimension
        """
        super().__init__()

        self.d = d

        # Stop head
        self.stop_mlp = nn.Sequential(
            nn.Linear(d, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Add head: [d + d + d] → 64 → 1
        self.add_mlp = nn.Sequential(
            nn.Linear(3 * d, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Source head (empty state)
        self.source_head = nn.Linear(d, d)  # Will be expanded to V later

    def forward(
        self,
        h_G: torch.Tensor,
        h_ap: torch.Tensor,
        block_embs: torch.Tensor,
        state: FragState,
        vocab: FragmentVocab
    ):
        """
        Compute forward policy logits.

        Args:
            h_G: [d] graph embedding
            h_ap: [num_open_AP, d] AP embeddings (empty for source step)
            block_embs: [V', d] block embeddings
            state: Current state
            vocab: Fragment vocabulary

        Returns:
            log_probs: Log probabilities over actions
            num_actions: Total number of actions
        """
        # Source step (empty state)
        if state.is_empty():
            # Pick any base fragment
            V = len(vocab.base_fragments)
            logits = self.source_head(h_G).unsqueeze(0)  # [1, d]

            # Expand to base vocab size
            # Simple approach: project to V dimensions
            # (In practice, would select first expanded entry per base frag)
            base_logits = []
            for base_idx in range(V):
                # Find first expanded entry
                for exp_idx, (b_idx, _) in enumerate(vocab.expanded_entries):
                    if b_idx == base_idx:
                        # Score this block
                        block_emb = block_embs[exp_idx].unsqueeze(0)  # [1, d]
                        score = (logits * block_emb).sum(dim=-1)  # [1]
                        base_logits.append(score)
                        break

            logits = torch.cat(base_logits)  # [V]
            log_probs = F.log_softmax(logits, dim=0)
            return log_probs, V

        # Regular step: Stop + Add
        num_open_ap = h_ap.size(0)
        V_prime = block_embs.size(0)

        # Stop logit
        stop_logit = self.stop_mlp(h_G)  # [1]

        # Add logits: for each (ap, block) pair
        if num_open_ap > 0:
            add_logits = []

            for ap_idx in range(num_open_ap):
                ap_emb = h_ap[ap_idx]  # [d]

                for block_idx in range(V_prime):
                    block_emb = block_embs[block_idx]  # [d]

                    # Concatenate [ap_emb; block_emb; h_G]
                    combined = torch.cat([ap_emb, block_emb, h_G], dim=0)  # [3d]
                    logit = self.add_mlp(combined)  # [1]
                    add_logits.append(logit)

            add_logits = torch.cat(add_logits)  # [num_AP * V']
        else:
            add_logits = torch.zeros(0, device=h_G.device)

        # Combine: [stop_logit, add_logits]
        all_logits = torch.cat([stop_logit, add_logits])  # [1 + num_AP * V']

        # Log-softmax
        log_probs = F.log_softmax(all_logits, dim=0)

        return log_probs, all_logits.size(0)


if __name__ == "__main__":
    from data.prepare_data import prepare_vocabulary
    from env.molecule_state import create_initial_state, add_fragment, state_to_pyg, get_open_aps
    from models.state_encoder import StateEncoder
    from models.fragment_embed import FragmentEmbedding

    project_root = Path(__file__).parent.parent
    vocab = prepare_vocabulary(
        project_root / "data" / "raw",
        project_root / "data" / "fragments",
        top_k=20
    )

    encoder = StateEncoder(d=128)
    frag_embed = FragmentEmbedding(vocab, d=128)
    policy = ForwardPolicy(d=128)

    # Get block embeddings
    block_embs = frag_embed()  # [V', 128]

    # Test source step (empty state)
    state = create_initial_state()
    data = state_to_pyg(state, vocab)
    h_v, h_G = encoder(data)
    h_ap = torch.zeros((0, 128))  # No APs yet

    log_probs, num_actions = policy(h_G, h_ap, block_embs, state, vocab)
    print(f"Source step:")
    print(f"  Num actions: {num_actions} (should be ~{len(vocab.base_fragments)})")
    print(f"  Log probs shape: {log_probs.shape}")

    # Test regular step
    if len(vocab.base_fragments) > 0:
        state = add_fragment(state, base_frag_id=0)
        data = state_to_pyg(state, vocab)
        h_v, h_G = encoder(data)
        open_aps = get_open_aps(state, vocab)
        h_ap = encoder.compute_ap_embeddings(h_v, open_aps)

        log_probs, num_actions = policy(h_G, h_ap, block_embs, state, vocab)
        print(f"\nRegular step:")
        print(f"  Num actions: {num_actions} (1 Stop + {len(open_aps)} * {len(vocab.expanded_entries)})")
        print(f"  Log probs shape: {log_probs.shape}")
