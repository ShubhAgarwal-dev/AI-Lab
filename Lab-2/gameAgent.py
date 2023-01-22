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
        # A function that returns all possible successor states.
        stack1 = []
        stack2 = []
        stack3 = []
        for block in state:
            if (block[0] == 0):
                stack1.append(block)
            elif (block[0] == 1):
                stack2.append(block)
            elif (block[0] == 2):
                stack3.append(block)
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

# defining heuristics

def manhattan_heuristic(initial_state: List[Tuple[int, int, str]],
                        final_state: List[Tuple[int, int, str]]) -> int:
    return sum(abs(initial_state[i][0] - final_state[i][0]) + abs(initial_state[i][1] - final_state[i][1])
               for i in range(len(initial_state)))


def manhattan_heuristic_maxi(initial_state: List[Tuple[int, int, str]],
                             final_state: List[Tuple[int, int, str]]) -> int:
    """modifyinfg the heuristic s.t. you will maximize the output"""
    return (-1)*manhattan_heuristic(initial_state, final_state)


def xnor_heuristic(initial_state: List[Tuple[int, int, str]],
                   final_state: List[Tuple[int, int, str]]) -> int:
    sum = 0
    for i in range(len(initial_state)):
        if initial_state[i] == final_state[i]:
            sum += 1
        else:
            sum -= 1
    return sum


def xnor_heuristic_modified(initial_state: List[Tuple[int, int, str]],
                            final_state: List[Tuple[int, int, str]]) -> int:
    sum = 0
    for i in range(len(initial_state)):
        if initial_state[i] == final_state[i]:
            sum += initial_state[i][1]
        else:
            sum -= initial_state[i][1]
    return sum


def ascii_heuristic(initial_state: List[Tuple[int, int, str]],
                    final_state: List[Tuple[int, int, str]]) -> int:
    sum = 0
    for i in range(len(initial_state)):
        if initial_state[i] == final_state[i]:
            sum += ord(initial_state[i][2].upper())
        else:
            sum -= ord(initial_state[i][2].upper())
    return sum
