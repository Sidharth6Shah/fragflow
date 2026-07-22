"""
Main training loop: TB loss, trajectory sampling, optimization

Simplified implementation - full torchgfn integration needed later.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
from tqdm import tqdm
from dataclasses import dataclass
from typing import List

sys.path.append(str(Path(__file__).parent.parent))

from data.prepare_data import prepare_vocabulary
from env.molecule_state import create_initial_state, state_to_pyg, get_open_aps
from env.molecule_env import MoleculeEnv
from env.actions import enumerate_actions, STOP_ACTION
from models.fragment_embed import FragmentEmbedding
from models.state_encoder import StateEncoder
from models.forward_policy import ForwardPolicy
from models.backward_policy import BackwardPolicy
from reward.reward_fn import compute_reward
from training.config import TrainingConfig, MILESTONE_CONFIG


@dataclass
class Trajectory:
    """Single trajectory: sequence of states and actions."""
    states: List
    actions: List
    log_pfs: List  # log P_F for each action
    reward: float


class GFlowNetTrainer:
    """Trainer for fragment GFlowNet."""

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Load vocab
        project_root = Path(__file__).parent.parent
        self.vocab = prepare_vocabulary(
            project_root / "data" / "raw",
            project_root / "data" / "fragments",
            top_k=config.vocab_size
        )

        # Initialize models
        self.frag_embed = FragmentEmbedding(self.vocab, d=config.d)
        self.encoder = StateEncoder(d=config.d, num_layers=config.num_gnn_layers)
        self.forward_policy = ForwardPolicy(d=config.d)
        self.backward_policy = BackwardPolicy(d=config.d)

        # log_Z parameter
        self.log_Z = nn.Parameter(torch.zeros(1))

        # Optimizer
        params = (
            list(self.frag_embed.parameters()) +
            list(self.encoder.parameters()) +
            list(self.forward_policy.parameters()) +
            list(self.backward_policy.parameters()) +
            [self.log_Z]
        )
        self.optimizer = torch.optim.Adam(params, lr=config.lr)

        # Environment
        self.env = MoleculeEnv(self.vocab, max_frags=config.max_frags)

        # Get block embeddings (cached)
        self.block_embs = self.frag_embed()

    def sample_trajectory(self) -> Trajectory:
        """Sample single trajectory using forward policy."""
        states = []
        actions = []
        log_pfs = []

        state = self.env.reset()
        done = False

        while not done:
            states.append(state)

            # Encode state
            data = state_to_pyg(state, self.vocab)
            h_v, h_G = self.encoder(data)

            # Get AP embeddings
            open_aps = get_open_aps(state, self.vocab)
            h_ap = self.encoder.compute_ap_embeddings(h_v, open_aps)

            # Forward policy
            log_probs, num_actions = self.forward_policy(
                h_G, h_ap, self.block_embs, state, self.vocab
            )

            # Sample action
            action_idx = torch.multinomial(torch.exp(log_probs), 1).item()
            log_pf = log_probs[action_idx]

            # Convert to action object (simplified - needs proper mapping)
            valid_actions = enumerate_actions(state, self.vocab, self.config.max_frags)
            if action_idx < len(valid_actions):
                action = valid_actions[action_idx]
            else:
                action = STOP_ACTION

            actions.append(action)
            log_pfs.append(log_pf)

            # Step
            state, done = self.env.step(action)

        # Compute reward
        reward = compute_reward(
            state, self.vocab,
            beta=self.config.beta,
            mode=self.config.reward_mode
        )

        return Trajectory(states, actions, log_pfs, reward)

    def compute_tb_loss(self, trajectories: List[Trajectory]) -> torch.Tensor:
        """Simplified TB loss."""
        losses = []
        
        for traj in trajectories:
            # Sum log P_F over trajectory
            log_pf_sum = sum(traj.log_pfs)
            
            # Simple loss: encourage high-reward trajectories
            loss = -(traj.reward * log_pf_sum)
            losses.append(loss)
        
        # Stack and mean
        return torch.stack(losses).mean()

    def train(self):
        """Main training loop."""
        print(f"Starting training with config: {self.config}")
        print(f"Vocab size: {len(self.vocab.base_fragments)}")
        print(f"Expanded vocab: {len(self.vocab.expanded_entries)}")

        for iteration in tqdm(range(self.config.num_iterations)):
            # Sample trajectories
            trajectories = []
            for _ in range(self.config.batch_size):
                traj = self.sample_trajectory()
                trajectories.append(traj)

            # Compute loss
            loss = self.compute_tb_loss(trajectories)

            # Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Log
            if iteration % self.config.log_every == 0:
                avg_reward = sum(t.reward for t in trajectories) / len(trajectories)
                print(f"\nIter {iteration}: loss={loss.item():.4f}, "
                      f"avg_reward={avg_reward:.4f}, log_Z={self.log_Z.item():.4f}")

            # Save checkpoint
            if iteration % self.config.save_every == 0 and iteration > 0:
                self.save_checkpoint(iteration)

    def save_checkpoint(self, iteration: int):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            'iteration': iteration,
            'frag_embed': self.frag_embed.state_dict(),
            'encoder': self.encoder.state_dict(),
            'forward_policy': self.forward_policy.state_dict(),
            'backward_policy': self.backward_policy.state_dict(),
            'log_Z': self.log_Z,
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }

        path = checkpoint_dir / f"checkpoint_{iteration}.pt"
        torch.save(checkpoint, path)
        print(f"\nSaved checkpoint to {path}")


if __name__ == "__main__":
    # Run milestone training
    config = MILESTONE_CONFIG
    trainer = GFlowNetTrainer(config)
    trainer.train()
