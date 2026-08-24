"""Test if multi-fragment molecules work and get better rewards."""
import sys
from pathlib import Path
import torch

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

print('Testing multi-fragment molecules:\n')

# Test 1: Build a 1-fragment molecule
state = env.reset()
state, done = env.step(STOP_ACTION)
mol_1 = state_to_molecule(state, vocab)
reward_1 = compute_reward(state, vocab, beta=4.0, mode='full')
smiles_1 = Chem.MolToSmiles(mol_1) if mol_1 else 'INVALID'
print(f'1 fragment: valid={mol_1 is not None}, reward={reward_1:.3f}, SMILES={smiles_1}')

# Test 2: Build a 2-fragment molecule
state = env.reset()
actions = enumerate_actions(state, vocab, max_frags=8)
add_actions = [a for a in actions if a != STOP_ACTION]
if len(add_actions) > 0:
    state, done = env.step(add_actions[0])
    mol_2 = state_to_molecule(state, vocab)
    reward_2 = compute_reward(state, vocab, beta=4.0, mode='full')
    smiles_2 = Chem.MolToSmiles(mol_2) if mol_2 else 'INVALID'
    print(f'2 fragments: valid={mol_2 is not None}, reward={reward_2:.3f}, SMILES={smiles_2}')
    print(f'  DEBUG: state.frags={state.frags}, state.bonds={state.bonds}')

    # Test 3: Build a 3-fragment molecule
    actions = enumerate_actions(state, vocab, max_frags=8)
    add_actions = [a for a in actions if a != STOP_ACTION]
    if len(add_actions) > 0:
        state, done = env.step(add_actions[0])
        mol_3 = state_to_molecule(state, vocab)
        reward_3 = compute_reward(state, vocab, beta=4.0, mode='full')
        smiles_3 = Chem.MolToSmiles(mol_3) if mol_3 else 'INVALID'
        print(f'3 fragments: valid={mol_3 is not None}, reward={reward_3:.3f}, SMILES={smiles_3}')

        print('\nReward comparison:')
        print(f'  1 frag: {reward_1:.3f}')
        print(f'  2 frags: {reward_2:.3f} ({"BETTER" if reward_2 > reward_1 else "WORSE"})')
        print(f'  3 frags: {reward_3:.3f} ({"BETTER" if reward_3 > reward_2 else "WORSE"})')
