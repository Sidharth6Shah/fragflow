"""Properly test multi-fragment assembly."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from data.prepare_data import prepare_vocabulary
from env.molecule_env import MoleculeEnv
from env.actions import STOP_ACTION, enumerate_actions
from env.molecule_state import state_to_molecule
from reward.reward_fn import compute_reward
from rdkit import Chem

# Load vocab
vocab = prepare_vocabulary(
    Path('data/raw'),
    Path('data/fragments'),
    top_k=200
)

env = MoleculeEnv(vocab, max_frags=8)

print('Testing proper multi-fragment assembly:\n')

# Build a 1-fragment molecule
state = env.reset()
actions = enumerate_actions(state, vocab, max_frags=8)
add_actions = [a for a in actions if a != STOP_ACTION]
state, done = env.step(add_actions[0])  # Add first fragment
state, done = env.step(STOP_ACTION)  # Stop

mol_1 = state_to_molecule(state, vocab)
reward_1 = compute_reward(state, vocab, beta=4.0, mode='full')
smiles_1 = Chem.MolToSmiles(mol_1) if mol_1 else 'INVALID'
print(f'1 fragment:')
print(f'  state.frags = {state.frags}')
print(f'  state.bonds = {state.bonds}')
print(f'  valid = {mol_1 is not None}')
print(f'  reward = {reward_1:.3f}')
print(f'  SMILES = {smiles_1}')

# Build a 2-fragment molecule
state = env.reset()
actions = enumerate_actions(state, vocab, max_frags=8)
add_actions = [a for a in actions if a != STOP_ACTION]
state, done = env.step(add_actions[0])  # Add first fragment

# Now add second fragment
actions = enumerate_actions(state, vocab, max_frags=8)
add_actions = [a for a in actions if a != STOP_ACTION]
if len(add_actions) > 0:
    print(f'\nBefore adding 2nd fragment: state.frags={state.frags}')
    print(f'  Available add actions: {len(add_actions)}')
    print(f'  First action: focus_ap={add_actions[0].focus_ap}, block_id={add_actions[0].block_id}')

    state, done = env.step(add_actions[0])  # Add second fragment
    state, done = env.step(STOP_ACTION)  # Stop

    mol_2 = state_to_molecule(state, vocab)
    reward_2 = compute_reward(state, vocab, beta=4.0, mode='full')
    smiles_2 = Chem.MolToSmiles(mol_2) if mol_2 else 'INVALID'
    print(f'\n2 fragments:')
    print(f'  state.frags = {state.frags}')
    print(f'  state.bonds = {state.bonds}')
    print(f'  valid = {mol_2 is not None}')
    print(f'  reward = {reward_2:.3f}')
    print(f'  SMILES = {smiles_2}')

    print(f'\nReward comparison:')
    print(f'  1 frag: {reward_1:.3f}')
    print(f'  2 frags: {reward_2:.3f} ({"BETTER" if reward_2 > reward_1 else "WORSE"})')
