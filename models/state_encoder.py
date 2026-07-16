"""
State encoder: GINE over fragment graph

3-layer edge-aware GNN that processes partial molecule state.
Outputs node embeddings and graph-level embedding.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool
from torch_geometric.data import Data, Batch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


class StateEncoder(nn.Module):
    """
    GINE encoder for fragment graphs.

    Input: PyG Data with x [N, 2052], edge_index [2, 2E], edge_attr [2E, 8]
    Output: h_v [N, 128], h_G [1, 128], h_ap [num_open_AP, 128]
    """

    def __init__(self, d: int = 128, num_layers: int = 3, max_ap: int = 4):
        """
        Initialize encoder.

        Args:
            d: Hidden dimension
            num_layers: Number of GNN layers
            max_ap: Maximum attachment points per fragment
        """
        super().__init__()

        self.d = d
        self.num_layers = num_layers
        self.max_ap = max_ap

        # Input projection: [2052] → [d]
        self.input_proj = nn.Linear(2052, d)

        # Edge projection: [8] → [d]
        self.edge_proj = nn.Linear(8, d)

        # GINE layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(d, d),
                nn.ReLU(),
                nn.Linear(d, d)
            )
            self.convs.append(GINEConv(mlp, edge_dim=d))

        # Graph pooling: [N, d] → [2d] → [d]
        self.graph_pool = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU()
        )

        # AP slot embeddings
        self.slot_embedding = nn.Embedding(max_ap, 32)

        # AP embedding MLP: [d + 32] → [d]
        self.ap_mlp = nn.Sequential(
            nn.Linear(d + 32, d),
            nn.ReLU(),
            nn.Linear(d, d)
        )

    def forward(self, data: Data):
        """
        Encode fragment graph state.

        Args:
            data: PyG Data object

        Returns:
            h_v: [N, d] node embeddings
            h_G: [d] graph embedding (single graph)
        """
        x = data.x  # [N, 2052]
        edge_index = data.edge_index  # [2, 2E]
        edge_attr = data.edge_attr  # [2E, 8]

        # Handle empty graph
        if x.size(0) == 0:
            # Return zero embeddings
            h_v = torch.zeros((0, self.d), device=x.device)
            h_G = torch.zeros(self.d, device=x.device)
            return h_v, h_G

        # Project inputs
        h = self.input_proj(x)  # [N, d]
        edge_emb = self.edge_proj(edge_attr)  # [2E, d]

        # GINE layers
        for conv in self.convs:
            h = conv(h, edge_index, edge_emb)
            h = torch.relu(h)

        h_v = h  # [N, d]

        # Graph pooling: combine mean and sum
        batch = torch.zeros(h_v.size(0), dtype=torch.long, device=h_v.device)
        h_mean = global_mean_pool(h_v, batch)  # [1, d]
        h_sum = global_add_pool(h_v, batch)    # [1, d]
        h_combined = torch.cat([h_mean, h_sum], dim=-1)  # [1, 2d]
        h_G = self.graph_pool(h_combined).squeeze(0)  # [d]

        return h_v, h_G

    def compute_ap_embeddings(
        self,
        h_v: torch.Tensor,
        open_aps: list
    ) -> torch.Tensor:
        """
        Compute embeddings for open attachment points.

        Args:
            h_v: [N, d] node embeddings
            open_aps: List of (frag_idx, ap_slot) tuples

        Returns:
            [num_open_AP, d] AP embeddings
        """
        if len(open_aps) == 0:
            return torch.zeros((0, self.d), device=h_v.device)

        ap_embs = []
        for frag_idx, ap_slot in open_aps:
            # Get fragment node embedding
            frag_emb = h_v[frag_idx]  # [d]

            # Get slot embedding
            slot_emb = self.slot_embedding(
                torch.tensor(ap_slot, device=h_v.device)
            )  # [32]

            # Combine
            combined = torch.cat([frag_emb, slot_emb], dim=-1)  # [d + 32]
            ap_emb = self.ap_mlp(combined)  # [d]
            ap_embs.append(ap_emb)

        return torch.stack(ap_embs)  # [num_open_AP, d]


if __name__ == "__main__":
    from data.prepare_data import prepare_vocabulary
    from env.molecule_state import create_initial_state, add_fragment, state_to_pyg, get_open_aps

    project_root = Path(__file__).parent.parent
    vocab = prepare_vocabulary(
        project_root / "data" / "raw",
        project_root / "data" / "fragments",
        top_k=20
    )

    encoder = StateEncoder(d=128)

    # Test with empty state
    state = create_initial_state()
    data = state_to_pyg(state, vocab)
    h_v, h_G = encoder(data)
    print(f"Empty state:")
    print(f"  h_v shape: {h_v.shape}")  # [0, 128]
    print(f"  h_G shape: {h_G.shape}")  # [128]

    # Test with one fragment
    if len(vocab.base_fragments) > 0:
        state = add_fragment(state, base_frag_id=0)
        data = state_to_pyg(state, vocab)
        h_v, h_G = encoder(data)
        print(f"\nOne fragment:")
        print(f"  h_v shape: {h_v.shape}")  # [1, 128]
        print(f"  h_G shape: {h_G.shape}")  # [128]

        # Compute AP embeddings
        open_aps = get_open_aps(state, vocab)
        h_ap = encoder.compute_ap_embeddings(h_v, open_aps)
        print(f"  h_ap shape: {h_ap.shape}")  # [num_AP, 128]
