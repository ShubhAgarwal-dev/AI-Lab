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

    def get_successor(self, state):
        stack1 = []
        stack2 = []
        stack3 = []
        for x in state:
            if (x[0] == 0):
                stack1.append(x)
            elif (x[0] == 1):
                stack2.append(x)
            elif (x[0] == 2):
                stack3.append(x)
            else:
                Exception("Stack Out of Range")
        successor = []
        for i in range(0, 3):
            if (i == 0 and len(stack1) != 0):
                for j in range(0, 2):
                    if (j == 0):
                        stack2.append(stack1.pop())  # updating the new stack
                        successor.append(stack1+stack2+stack3)
                        # resetting the original stack
                        stack1.append(stack2.pop())
                    if (j == 1):
                        stack3.append(stack1.pop())
                        successor.append(stack1+stack2+stack3)
                        stack1.append(stack3.pop())
            if (i == 1 and len(stack2) != 0):
                for j in range(0, 2):
                    if (j == 0):
                        stack1.append(stack2.pop())
                        successor.append(stack1+stack2+stack3)
                        stack2.append(stack1.pop())
                    if (j == 1):
                        stack3.append(stack2.pop())
                        successor.append(stack1+stack2+stack3)
                        stack2.append(stack3.pop())
            if (i == 2 and len(stack3) != 0):
                for j in range(0, 2):
                    if (j == 0):
                        stack1.append(stack3.pop())
                        successor.append(stack1+stack2+stack3)
                        stack3.append(stack1.pop())
                    if (j == 1):
                        stack2.append(stack3.pop())
                        successor.append(stack1+stack2+stack3)
                        stack3.append(stack2.pop())
        return successor

# definig heuristics
