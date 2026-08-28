"""
Print model architecture and parameter counts.
"""
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from data.prepare_data import prepare_vocabulary
from models.fragment_embed import FragmentEmbedding
from models.state_encoder import StateEncoder
from models.forward_policy import ForwardPolicy
from models.backward_policy import BackwardPolicy


def count_params(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_size(num_params):
    """Format parameter count with K/M suffix."""
    if num_params >= 1_000_000:
        return f"{num_params/1_000_000:.2f}M"
    elif num_params >= 1_000:
        return f"{num_params/1_000:.2f}K"
    else:
        return f"{num_params}"


def main():
    print("=" * 60)
    print("FragFlow Model Architecture")
    print("=" * 60)

    # Load vocab
    project_root = Path(__file__).parent.parent
    vocab = prepare_vocabulary(
        project_root / "data" / "raw",
        project_root / "data" / "fragments",
        top_k=200
    )

    print(f"\nVocabulary:")
    print(f"  Base fragments:     {len(vocab.base_fragments)}")
    print(f"  Expanded entries:   {len(vocab.expanded_entries)}")

    # Initialize models
    d = 128
    frag_embed = FragmentEmbedding(vocab, d=d)
    encoder = StateEncoder(d=d, num_layers=3)
    forward_policy = ForwardPolicy(d=d)
    backward_policy = BackwardPolicy(d=d)

    # Count parameters
    frag_params = count_params(frag_embed)
    encoder_params = count_params(encoder)
    forward_params = count_params(forward_policy)
    backward_params = count_params(backward_policy)
    log_z_params = 1
    total_params = frag_params + encoder_params + forward_params + backward_params + log_z_params

    print("\n" + "=" * 60)
    print("Model Parameters")
    print("=" * 60)
    print(f"Fragment Embedding:  {frag_params:>10,}  ({format_size(frag_params):>8})")
    print(f"State Encoder:       {encoder_params:>10,}  ({format_size(encoder_params):>8})")
    print(f"Forward Policy:      {forward_params:>10,}  ({format_size(forward_params):>8})")
    print(f"Backward Policy:     {backward_params:>10,}  ({format_size(backward_params):>8})")
    print(f"log_Z:               {log_z_params:>10,}  ({format_size(log_z_params):>8})")
    print("-" * 60)
    print(f"Total:               {total_params:>10,}  ({format_size(total_params):>8})")

    # Checkpoint size
    checkpoint_path = project_root / "checkpoints" / "checkpoint_10000.pt"
    if checkpoint_path.exists():
        import os
        ckpt_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print("\n" + "=" * 60)
        print("Checkpoint Info")
        print("=" * 60)
        print(f"File:                checkpoint_10000.pt")
        print(f"Size:                {ckpt_size_mb:.2f} MB")

        # Load checkpoint to get iteration
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"Iteration:           {checkpoint['iteration']}")
        print(f"log_Z:               {checkpoint['log_Z'].item():.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
