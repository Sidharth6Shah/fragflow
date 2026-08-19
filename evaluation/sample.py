"""
Sample molecules from trained GFlowNet model.

Load checkpoint, sample trajectories, convert to SMILES.
"""

import torch
from pathlib import Path
import sys
from tqdm import tqdm
from typing import List
import pickle

sys.path.append(str(Path(__file__).parent.parent))

from data.prepare_data import prepare_vocabulary
from env.molecule_state import create_initial_state, state_to_pyg, get_open_aps, state_to_molecule
from env.molecule_env import MoleculeEnv
from env.actions import STOP_ACTION
from models.fragment_embed import FragmentEmbedding
from models.state_encoder import StateEncoder
from models.forward_policy import ForwardPolicy
from reward.reward_fn import compute_reward


class MoleculeSampler:
    """Sample molecules from trained GFlowNet."""

    def __init__(self, checkpoint_path: str, vocab_size: int = 200, max_frags: int = 8):
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            vocab_size: Fragment vocabulary size
            max_frags: Maximum fragments per molecule
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.vocab_size = vocab_size
        self.max_frags = max_frags

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load vocab
        project_root = Path(__file__).parent.parent
        self.vocab = prepare_vocabulary(
            project_root / "data" / "raw",
            project_root / "data" / "fragments",
            top_k=vocab_size
        )

        # Initialize models
        self.frag_embed = FragmentEmbedding(self.vocab, d=128).to(self.device)
        self.encoder = StateEncoder(d=128, num_layers=3).to(self.device)
        self.forward_policy = ForwardPolicy(d=128).to(self.device)

        # Environment
        self.env = MoleculeEnv(self.vocab, max_frags=max_frags)

        # Load checkpoint
        self._load_checkpoint()

    def _load_checkpoint(self):
        """Load model weights from checkpoint."""
        print(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        self.frag_embed.load_state_dict(checkpoint['frag_embed'])
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.forward_policy.load_state_dict(checkpoint['forward_policy'])

        print(f"Loaded checkpoint from iteration {checkpoint['iteration']}")
        print(f"log_Z: {checkpoint['log_Z'].item():.4f}")

        # Set to eval mode
        self.frag_embed.eval()
        self.encoder.eval()
        self.forward_policy.eval()

    @torch.no_grad()
    def sample_molecule(self):
        """
        Sample a single molecule.

        Returns:
            (state, smiles, reward) tuple
        """
        block_embs = self.frag_embed()

        state = self.env.reset()
        done = False

        while not done:
            # Encode state
            data = state_to_pyg(state, self.vocab).to(self.device)
            h_v, h_G = self.encoder(data)

            # Get AP embeddings
            open_aps = get_open_aps(state, self.vocab)
            h_ap = self.encoder.compute_ap_embeddings(h_v, open_aps)

            # Forward policy
            log_probs, num_actions = self.forward_policy(
                h_G, h_ap, block_embs, state, self.vocab
            )

            # Sample action (greedy or stochastic)
            action_idx = torch.multinomial(torch.exp(log_probs), 1).item()

            # Convert to action object
            from env.actions import enumerate_actions
            valid_actions = enumerate_actions(state, self.vocab, self.max_frags)
            if action_idx < len(valid_actions):
                action = valid_actions[action_idx]
            else:
                action = STOP_ACTION

            # Step
            state, done = self.env.step(action)

        # Convert to SMILES
        smiles = state_to_molecule(state, self.vocab)

        # Compute reward
        reward = compute_reward(state, self.vocab, beta=1.0, mode="full")

        return state, smiles, reward

    def sample_batch(self, num_samples: int = 100):
        """
        Sample multiple molecules.

        Args:
            num_samples: Number of molecules to generate

        Returns:
            List of (state, smiles, reward) tuples
        """
        samples = []

        for _ in tqdm(range(num_samples), desc="Sampling molecules"):
            try:
                state, smiles, reward = self.sample_molecule()
                samples.append((state, smiles, reward))
            except Exception as e:
                print(f"Error sampling molecule: {e}")
                continue

        return samples


def main():
    """Sample molecules from checkpoint and save."""
    checkpoint_path = "checkpoints/checkpoint_15000.pt"
    output_path = "evaluation/sampled_molecules.pkl"

    # Sample molecules
    sampler = MoleculeSampler(checkpoint_path, vocab_size=200, max_frags=8)
    samples = sampler.sample_batch(num_samples=1000)

    # Save
    with open(output_path, 'wb') as f:
        pickle.dump(samples, f)

    # Print stats
    print(f"\nSampled {len(samples)} molecules")
    print(f"Saved to {output_path}")

    valid_smiles = [s for _, s, _ in samples if s is not None]
    print(f"Valid SMILES: {len(valid_smiles)}/{len(samples)} ({100*len(valid_smiles)/len(samples):.1f}%)")

    if valid_smiles:
        print(f"\nFirst 10 molecules:")
        for i, (_, smiles, reward) in enumerate(samples[:10]):
            if smiles:
                print(f"  {i+1}. {smiles} (reward: {reward:.3f})")


if __name__ == "__main__":
    main()
