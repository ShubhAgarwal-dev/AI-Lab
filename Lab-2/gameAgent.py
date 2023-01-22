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

    def makeStack(self, state, index):
        returnStack = []
        for block in state:
            if (block[0] == index):
                returnStack.append(block)
        return returnStack

    def get_valid_moves(self, stack0, stack1, stack2, index1, index2):
        successor = []
        copyStackA = []
        copyStackB = []
        remainStack = []
        if (index1 == 0 and index2 == 1):
            copyStackA += stack0
            copyStackB += stack1
            remainStack += stack2
        elif (index1 == 1 and index2 == 0):
            copyStackA += stack1
            copyStackB += stack0
            remainStack += stack2
        elif (index1 == 0 and index2 == 2):
            copyStackA += stack0
            copyStackB += stack2
            remainStack += stack1
        elif (index1 == 2 and index2 == 0):
            copyStackA += stack2
            copyStackB += stack0
            remainStack += stack1
        elif (index1 == 1 and index2 == 2):
            copyStackA += stack1
            copyStackB += stack2
            remainStack += stack0
        elif (index1 == 2 and index2 == 1):
            copyStackA += stack2
            copyStackB += stack1
            remainStack += stack0
        else:
            raise Exception("Index Out of Range")
        if (copyStackA == []):
            return [copyStackA+copyStackB+remainStack]
        sizeCopyStackB = len(copyStackB)
        topValStackA = copyStackA.pop()
        newTup = (index2, sizeCopyStackB+1, topValStackA[2])
        copyStackB = copyStackB+[newTup]
        successor = copyStackA+copyStackB+remainStack
        successor.sort(key=lambda x: x[2])
        return successor

    def get_successor(self, state):
        # A function that returns all possible successor states.
        stack0 = self.makeStack(state, 0)
        stack1 = self.makeStack(state, 1)
        stack2 = self.makeStack(state, 2)
        succcessor = []
        for i in range(0, 3):
            if (i == 0):
                succcessor.append(self.get_valid_moves(
                    stack0, stack1, stack2, 0, 1))
                succcessor.append(self.get_valid_moves(
                    stack0, stack1, stack2, 0, 2))
            elif (i == 1):
                succcessor.append(self.get_valid_moves(
                    stack0, stack1, stack2, 1, 0))
                succcessor.append(self.get_valid_moves(
                    stack0, stack1, stack2, 1, 2))
            elif (i == 2):
                succcessor.append(self.get_valid_moves(
                    stack0, stack1, stack2, 2, 0))
                succcessor.append(self.get_valid_moves(
                    stack0, stack1, stack2, 2, 1))
            else:
                raise Exception("Stack Index Out of Range")
        return succcessor


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
            sum += ord(initial_state[i][2].upper())*initial_state[i][1]
        else:
            sum -= ord(initial_state[i][2].upper())*initial_state[i][1]
    return sum
