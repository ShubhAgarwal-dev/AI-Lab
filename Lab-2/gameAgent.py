from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class Problem:
    """Extend this class to make a problem"""

    start_state: List[Tuple]
    final_state: List[Tuple]
    visited: int = 0

    def is_goal_state(self, state) -> bool:
        if (state == self.final_state):
            return True
        return False

    def get_successor(self, state):
        """It will return list of possible successor_states"""
        # NOTE: ignore the code written over here
        successors = [state]
        return successors

    def get_initial_state(self):
        return self.start_state

    def get_goal_state(self):
        return self.final_state


class BlockWorldDiagram(Problem):

    def __init__(self, start_state: List[Tuple[int, int, str]],
                 final_state: List[Tuple[int, int, str]]):
        super().__init__(start_state, final_state)

# definig heuristics
