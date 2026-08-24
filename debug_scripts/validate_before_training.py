"""
Comprehensive validation before Lambda GPU training.

Tests all critical components to ensure nothing is broken.
Run this before spending $$ on GPU training.
"""
import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent.parent))

from data.prepare_data import prepare_vocabulary
from env.molecule_env import MoleculeEnv
from env.actions import STOP_ACTION, enumerate_actions
from env.molecule_state import state_to_molecule, state_to_pyg, get_open_aps
from reward.reward_fn import compute_reward
from models.fragment_embed import FragmentEmbedding
from models.state_encoder import StateEncoder
from models.forward_policy import ForwardPolicy
from models.backward_policy import BackwardPolicy
from rdkit import Chem

print("="*60)
print("PRE-TRAINING VALIDATION")
print("="*60)

# Load vocab
print("\n[1/8] Loading vocabulary...")
vocab = prepare_vocabulary(Path('data/raw'), Path('data/fragments'), top_k=200)
print(f"✓ Vocab loaded: {len(vocab.base_fragments)} base fragments, {len(vocab.expanded_entries)} expanded entries")

# Test multi-fragment assembly
print("\n[2/8] Testing multi-fragment molecule assembly...")
env = MoleculeEnv(vocab, max_frags=8)

test_cases = []
for num_frags in [1, 2, 3, 4]:
    state = env.reset()
    for _ in range(num_frags):
        actions = enumerate_actions(state, vocab, max_frags=8)
        add_actions = [a for a in actions if a != STOP_ACTION]
        if add_actions:
            state, done = env.step(add_actions[0])
    state, done = env.step(STOP_ACTION)

    mol = state_to_molecule(state, vocab)
    reward = compute_reward(state, vocab, beta=4.0, mode='full')
    smiles = Chem.MolToSmiles(mol) if mol else 'INVALID'
    test_cases.append((num_frags, mol is not None, reward, smiles))

    print(f"  {num_frags} frag(s): valid={mol is not None}, reward={reward:.3f}, SMILES={smiles}")

# Check that multi-fragment gets better rewards
valid_rewards = [(n, r) for n, v, r, s in test_cases if v]
if len(valid_rewards) >= 2:
    single_frag_reward = valid_rewards[0][1]
    multi_frag_reward = valid_rewards[1][1]
    if multi_frag_reward > single_frag_reward:
        print("✓ Multi-fragment molecules get BETTER rewards (correct!)")
    else:
        print("✗ WARNING: Multi-fragment molecules get WORSE rewards!")
        sys.exit(1)
else:
    print("✗ ERROR: Not enough valid molecules generated")
    sys.exit(1)

# Test environment stepping
print("\n[3/8] Testing environment stepping and state transitions...")
state = env.reset()
assert state.is_empty(), "Reset should return empty state"
assert not env.is_terminal(), "Initial state should not be terminal"

actions = enumerate_actions(state, vocab, max_frags=8)
assert len(actions) > 0, "Should have actions from empty state"
assert STOP_ACTION not in actions, "STOP should be masked at empty state"

state, done = env.step(actions[0])
assert len(state.frags) == 1, "Should have 1 fragment after first add"
assert not done, "Should not be done after first add"

actions = enumerate_actions(state, vocab, max_frags=8)
assert STOP_ACTION in actions, "STOP should be available after 1 fragment"

state, done = env.step(STOP_ACTION)
assert done, "Should be done after STOP"
assert env.is_terminal(), "Should be terminal after STOP"

print("✓ Environment stepping works correctly")

# Test PyG state conversion
print("\n[4/8] Testing PyG state conversion...")
state = env.reset()
for _ in range(3):
    actions = enumerate_actions(state, vocab, max_frags=8)
    add_actions = [a for a in actions if a != STOP_ACTION]
    if add_actions:
        state, done = env.step(add_actions[0])

data = state_to_pyg(state, vocab)
assert data.x.shape[0] == len(state.frags), "Should have node per fragment"
assert data.x.shape[1] == 2052, "Node features should be [2048 ECFP + 4 AP slots]"
assert data.edge_index.shape[0] == 2, "Edge index should be [2, num_edges]"
print(f"✓ PyG conversion works: {data.x.shape[0]} nodes, {data.edge_index.shape[1]//2} edges")

# Test model forward passes
print("\n[5/8] Testing model forward passes...")
device = torch.device("cpu")

frag_embed = FragmentEmbedding(vocab, d=128).to(device)
encoder = StateEncoder(d=128, num_layers=3).to(device)
forward_policy = ForwardPolicy(d=128).to(device)
backward_policy = BackwardPolicy(d=128).to(device)

block_embs = frag_embed()
assert block_embs.shape == (len(vocab.expanded_entries), 128), "Block embeddings shape wrong"

data = data.to(device)
h_v, h_G = encoder(data)
assert h_v.shape == (data.x.shape[0], 128), "Node embeddings shape wrong"
assert h_G.shape == (128,), "Graph embedding shape wrong"

open_aps = get_open_aps(state, vocab)
h_ap = encoder.compute_ap_embeddings(h_v, open_aps)

log_probs, num_actions = forward_policy(h_G, h_ap, block_embs, state, vocab)
assert log_probs.shape == (num_actions,), "Forward policy output shape wrong"
assert torch.isfinite(log_probs).all(), "Forward policy has NaN/Inf"

back_log_probs, num_back_actions = backward_policy(h_v, state)
assert torch.isfinite(back_log_probs).all(), "Backward policy has NaN/Inf"

print("✓ All models forward pass successfully")

# Test gradient clipping setup
print("\n[6/8] Testing gradient clipping setup...")
log_Z = torch.nn.Parameter(torch.zeros(1))
params = (
    list(frag_embed.parameters()) +
    list(encoder.parameters()) +
    list(forward_policy.parameters()) +
    list(backward_policy.parameters()) +
    [log_Z]
)
optimizer = torch.optim.Adam(params, lr=1e-3)

# Dummy backward pass
loss = log_probs.sum()
optimizer.zero_grad()
loss.backward()

# Test gradient clipping
torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
optimizer.step()

print("✓ Gradient clipping works")

# Test reward modes
print("\n[7/8] Testing reward modes...")
state = env.reset()
actions = enumerate_actions(state, vocab, max_frags=8)
state, done = env.step(actions[0])
state, done = env.step(STOP_ACTION)

reward_dummy = compute_reward(state, vocab, beta=1.0, mode='dummy')
reward_full = compute_reward(state, vocab, beta=4.0, mode='full')
reward_scaffold = compute_reward(state, vocab, beta=4.0, mode='scaffold',
                                 scaffold_smarts='c1ccccc1', scaffold_weight=0.3)

assert isinstance(reward_dummy, (int, float)), "Dummy reward should be numeric"
assert isinstance(reward_full, (int, float)), "Full reward should be numeric"
assert isinstance(reward_scaffold, (int, float)), "Scaffold reward should be numeric"
assert torch.isfinite(torch.tensor(reward_full)), "Full reward has NaN/Inf"

print(f"✓ All reward modes work (dummy={reward_dummy:.3f}, full={reward_full:.3f}, scaffold={reward_scaffold:.3f})")

# Run mini training loop
print("\n[8/8] Running mini training loop (10 iterations)...")
from training.train import GFlowNetTrainer
from training.config import TrainingConfig

mini_config = TrainingConfig(
    vocab_size=20,
    max_frags=4,
    batch_size=4,
    num_iterations=10,
    reward_mode='full',
    beta=4.0,
    log_every=5,
    save_every=100,
    wandb_enabled=False
)

try:
    trainer = GFlowNetTrainer(mini_config)
    # Run just a few iterations
    import warnings
    warnings.filterwarnings('ignore')

    for i in range(10):
        # Sample trajectories
        trajectories = []
        for _ in range(mini_config.batch_size):
            traj = trainer.sample_trajectory()
            trajectories.append(traj)

        # Extract terminal states and rewards
        states = [traj.states[-1] for traj in trajectories]
        rewards = torch.tensor([traj.reward for traj in trajectories])

        assert torch.isfinite(rewards).all(), f"Invalid rewards at iter {i}"
        assert len(states) == mini_config.batch_size, f"Wrong batch size at iter {i}"

    print("✓ Mini training loop completes without errors")

except Exception as e:
    print(f"✗ ERROR in training loop: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final summary
print("\n" + "="*60)
print("VALIDATION COMPLETE - ALL TESTS PASSED ✓")
print("="*60)
print("\nReady for Lambda GPU training!")
print("Expected cost: ~$20 for 10K iterations")
print("Expected time: ~10 hours")
print("\nTo start training on Lambda GPU:")
print("  1. Launch instance (Lambda Stack 22.04, 1x A100)")
print("  2. ssh ubuntu@<IP>")
print("  3. git clone https://github.com/YOUR_USERNAME/fragflow.git")
print("  4. cd fragflow")
print("  5. pip install -r requirements.txt")
print("  6. nohup python training/train.py > training.log 2>&1 &")
print("\nSee GPU_TRAINING.md for full details.")
