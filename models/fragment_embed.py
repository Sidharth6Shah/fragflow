"""
Fragment embeddings: ECFP → learned projection to d=128

Loads precomputed ECFP4 fingerprints and projects to embedding space.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from data.prepare_data import FragmentVocab


class FragmentEmbedding(nn.Module):
    """
    Projects ECFP fingerprints to learned embeddings.

    Fixed ECFP4 [V', 2048] → Learned Linear → [V', d=128]
    """

    def __init__(self, vocab: FragmentVocab, d: int = 128):
        """
        Initialize fragment embeddings.

        Args:
            vocab: Fragment vocabulary with precomputed ECFP matrix
            d: Embedding dimension
        """
        super().__init__()

        self.d = d
        self.V_prime = len(vocab.expanded_entries)

        # Register ECFP matrix as buffer (not trained)
        ecfp_tensor = torch.tensor(vocab.ecfp_matrix, dtype=torch.float32)
        self.register_buffer('ecfp_matrix', ecfp_tensor)

        # Learned projection
        self.projection = nn.Linear(2048, d)

    def forward(self, block_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Get embeddings for blocks.

        Args:
            block_ids: Optional [B] tensor of block indices. If None, returns all.

        Returns:
            [B, d] or [V', d] embeddings
        """
        # Project all ECFP vectors
        all_embeddings = self.projection(self.ecfp_matrix)  # [V', d]

        if block_ids is None:
            return all_embeddings

        # Select specific blocks
        return all_embeddings[block_ids]  # [B, d]


if __name__ == "__main__":
    from data.prepare_data import prepare_vocabulary

    project_root = Path(__file__).parent.parent
    vocab = prepare_vocabulary(
        project_root / "data" / "raw",
        project_root / "data" / "fragments",
        top_k=20
    )

    embed = FragmentEmbedding(vocab, d=128)

    # Get all embeddings
    all_emb = embed()
    print(f"All embeddings shape: {all_emb.shape}")  # [V', 128]

    # Get specific blocks
    block_ids = torch.tensor([0, 1, 2])
    emb = embed(block_ids)
    print(f"Selected embeddings shape: {emb.shape}")  # [3, 128]
