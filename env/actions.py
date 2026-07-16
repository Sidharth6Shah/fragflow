"""
Action space definition (attach/stop)

Two action types:
- Stop: Terminate building
- Add: (focus_ap, block_id) - attach expanded vocab entry at open AP
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from env.molecule_state import FragState, get_open_aps
from data.prepare_data import FragmentVocab


@dataclass
class Action:
    """Action in the fragment assembly MDP."""
    type: str  # "stop" or "add"
    focus_ap: Optional[Tuple[int, int]] = None  # (frag_idx, ap_slot)
    block_id: Optional[int] = None  # expanded vocab index


# Sentinel for Stop action
STOP_ACTION = Action(type="stop")


def enumerate_actions(
    state: FragState,
    vocab: FragmentVocab,
    max_frags: int = 8
) -> List[Action]:
    """
    List all valid actions from current state.

    Args:
        state: Current fragment state
        vocab: Fragment vocabulary
        max_frags: Maximum fragments allowed

    Returns:
        List of valid actions
    """
    actions = []

    # Source step: pick any base fragment
    if state.is_empty():
        for base_idx in range(len(vocab.base_fragments)):
            # Use first expanded entry for each base fragment
            for exp_idx, (b_idx, _) in enumerate(vocab.expanded_entries):
                if b_idx == base_idx:
                    actions.append(Action(type="add", focus_ap=None, block_id=exp_idx))
                    break
        return actions

    # Regular step: Stop or Add
    # Stop is valid if we have at least 1 fragment
    if state.num_fragments() > 0:
        actions.append(STOP_ACTION)

    # Don't add more if at max capacity
    if state.num_fragments() >= max_frags:
        return actions

    # Add actions: for each open AP, try each block
    open_aps = get_open_aps(state, vocab)

    for focus_ap in open_aps:
        for block_id in range(len(vocab.expanded_entries)):
            # Could add more filtering here (chemical validity, etc.)
            actions.append(Action(type="add", focus_ap=focus_ap, block_id=block_id))

    return actions


def action_to_index(action: Action, state: FragState, vocab: FragmentVocab) -> int:
    """
    Convert action to flat index for categorical distribution.

    Action space size: 1 (Stop) + |open_APs| * V'

    Args:
        action: Action object
        state: Current state
        vocab: Fragment vocabulary

    Returns:
        Flat index
    """
    if action.type == "stop":
        return 0

    # For add actions, index = 1 + (focus_ap_idx * V' + block_id)
    open_aps = get_open_aps(state, vocab)
    focus_ap_idx = open_aps.index(action.focus_ap)

    V_prime = len(vocab.expanded_entries)
    return 1 + focus_ap_idx * V_prime + action.block_id


def index_to_action(
    index: int,
    state: FragState,
    vocab: FragmentVocab
) -> Action:
    """
    Convert flat index back to action.

    Args:
        index: Flat action index
        state: Current state
        vocab: Fragment vocabulary

    Returns:
        Action object
    """
    if index == 0:
        return STOP_ACTION

    # Decode add action
    open_aps = get_open_aps(state, vocab)
    V_prime = len(vocab.expanded_entries)

    add_idx = index - 1
    focus_ap_idx = add_idx // V_prime
    block_id = add_idx % V_prime

    return Action(type="add", focus_ap=open_aps[focus_ap_idx], block_id=block_id)


if __name__ == "__main__":
    from data.prepare_data import prepare_vocabulary
    from env.molecule_state import create_initial_state, add_fragment

    project_root = Path(__file__).parent.parent
    vocab = prepare_vocabulary(
        project_root / "data" / "raw",
        project_root / "data" / "fragments",
        top_k=20
    )

    # Test empty state
    state = create_initial_state()
    actions = enumerate_actions(state, vocab)
    print(f"Empty state: {len(actions)} actions (should be ~20 base fragments)")

    # Test with one fragment
    if len(vocab.base_fragments) > 0:
        state = add_fragment(state, base_frag_id=0)
        actions = enumerate_actions(state, vocab)
        print(f"One fragment: {len(actions)} actions (1 Stop + open_APs * V')")
        print(f"  Stop action included: {any(a.type == 'stop' for a in actions)}")
