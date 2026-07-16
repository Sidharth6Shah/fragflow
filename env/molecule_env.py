"""
torchgfn Environment: step(), reset(), is_terminal()

Manages the fragment assembly process.
"""

from typing import Tuple, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from env.molecule_state import FragState, create_initial_state, add_fragment
from env.actions import Action, enumerate_actions
from data.prepare_data import FragmentVocab


class MoleculeEnv:
    """
    Fragment assembly environment.

    Tracks state transitions as fragments are added.
    """

    def __init__(self, vocab: FragmentVocab, max_frags: int = 8):
        """
        Initialize environment.

        Args:
            vocab: Fragment vocabulary
            max_frags: Maximum fragments per molecule
        """
        self.vocab = vocab
        self.max_frags = max_frags
        self.state = create_initial_state()
        self._is_terminal = False

    def reset(self) -> FragState:
        """
        Reset to empty state.

        Returns:
            Initial empty state
        """
        self.state = create_initial_state()
        self._is_terminal = False
        return self.state

    def step(self, action: Action) -> Tuple[FragState, bool]:
        """
        Apply action and transition to new state.

        Args:
            action: Action to apply

        Returns:
            (new_state, done)
        """
        if self._is_terminal:
            # Already terminal, no change
            return self.state, True

        if action.type == "stop":
            # Stop action - mark as terminal
            self._is_terminal = True
            return self.state, True

        # Add action
        if self.state.is_empty():
            # Source step - add first fragment
            base_idx, _ = self.vocab.expanded_entries[action.block_id]
            self.state = add_fragment(self.state, base_frag_id=base_idx)
        else:
            # Regular step - connect to existing fragment
            base_idx, new_frag_ap = self.vocab.expanded_entries[action.block_id]
            self.state = add_fragment(
                self.state,
                base_frag_id=base_idx,
                connect_to=action.focus_ap,
                connect_from_ap=new_frag_ap
            )

        return self.state, False

    def is_terminal(self) -> bool:
        """Check if in terminal state."""
        return self._is_terminal

    def get_valid_actions(self) -> list:
        """Get all valid actions from current state."""
        return enumerate_actions(self.state, self.vocab, self.max_frags)


if __name__ == "__main__":
    from data.prepare_data import prepare_vocabulary

    project_root = Path(__file__).parent.parent
    vocab = prepare_vocabulary(
        project_root / "data" / "raw",
        project_root / "data" / "fragments",
        top_k=20
    )

    env = MoleculeEnv(vocab, max_frags=4)

    # Test reset
    state = env.reset()
    print(f"Reset state: {state}")
    print(f"Is terminal: {env.is_terminal()}")

    # Test adding first fragment
    actions = env.get_valid_actions()
    print(f"\nAvailable actions: {len(actions)}")

    if len(actions) > 0:
        state, done = env.step(actions[0])
        print(f"\nAfter first fragment:")
        print(f"  State: {state}")
        print(f"  Done: {done}")
        print(f"  Num fragments: {state.num_fragments()}")

        # Test stop
        from env.actions import STOP_ACTION
        state, done = env.step(STOP_ACTION)
        print(f"\nAfter stop:")
        print(f"  Done: {done}")
        print(f"  Is terminal: {env.is_terminal()}")
