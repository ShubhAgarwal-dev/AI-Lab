from typing import List, Tuple, Union, Callable
from pathlib import Path
from dataclasses import dataclass


def file_reader(file: Union[str, Path]) -> List[Tuple[int, int, str]]:
    state = []
    with open(file, 'r') as file1:
        content = file1.readlines()

    for line in content:
        line = line.strip('\n')
        line = line.split(" ")
        block = (int(line[0]), int(line[1]), line[2])
        state.append(block)

    state.sort(key=lambda x: x[2])
    return state


def encoder(state: List[Tuple[int, int, str]]) -> List[List[str]]:
    state_stack = [[], [], []]
    state.sort(key=lambda x: (len(state) + 2)*x[0]+x[1])
    for x in state:
        state_stack[x[0]].append(x[2])
    state.sort(key=lambda x: x[2])
    return state_stack


def decoder(state_stack: List[List[str]]) -> List[Tuple[int, int, str]]:
    state = []
    for i, tower in enumerate(state_stack):
        for j, label in enumerate(tower):
            state.append((i, j, label))
    state.sort(key=lambda x: x[2])
    return state


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

    def get_successor(self, state: List[Tuple[int, int, str]]) -> List[List[Tuple[int, int, str]]]:
        state_stacks = encoder(state)
        new_states = []
        for i, tower in enumerate(state_stacks):
            if len(tower) != 0:
                top_block = tower.pop()
                for j in range(len(state_stacks)):
                    if i != j:
                        state_stacks[j].append(top_block)
                        new_states.append(decoder(state_stacks))
                        state_stacks[j].pop()
                state_stacks[i].append(top_block)
        return new_states


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
            sum += (initial_state[i][1] + 1)
        else:
            sum -= (initial_state[i][1] + 1)
    return sum


def ascii_heuristic(initial_state: List[Tuple[int, int, str]],
                    final_state: List[Tuple[int, int, str]]) -> int:
    sum = 0
    for i in range(len(initial_state)):
        if initial_state[i] == final_state[i]:
            sum += ord(initial_state[i][2].upper()) * \
                abs((initial_state[i][1])-final_state[i][1])
        else:
            sum -= ord(initial_state[i][2].upper()) * \
                abs(initial_state[i][1]-final_state[i][1])
    return sum


def hillClimb(problem: Problem, heuristic: Callable[..., int]) -> bool:
    """It will return if the end_goal is rechable or not using 
    the current heuristics, using greedy approach, currently maximizing
    the heuristics value"""
    initial_state = problem.get_initial_state()
    final_state = problem.get_goal_state()
    current_node = initial_state
    current_node_heu = heuristic(current_node, final_state)
    while True:
        if problem.is_goal_state(current_node):
            break
        successors = problem.get_successor(current_node)
        heuristic_vals = [heuristic(successor, final_state) for successor in successors]
        max_heu = max(heuristic_vals)
        if max_heu > current_node_heu:
            current_node = successors[heuristic_vals.index(max_heu)]
            current_node_heu = max_heu
        else:
            return False
    return True


def hillClimbMod(problem: Problem, heuristic: Callable[..., int]) -> Tuple[int, bool]:
    """It will return if the end_goal is rechable or not using 
    the current heuristics, using greedy approach, currently maximizing
    the heuristics value"""
    initial_state = problem.get_initial_state()
    final_state = problem.get_goal_state()
    current_node = initial_state
    current_node_heu = heuristic(current_node, final_state)
    count = 0
    while True:
        if problem.is_goal_state(current_node):
            break
        successors = problem.get_successor(current_node)
        heuristic_vals = [heuristic(successor, final_state) for successor in successors]
        max_heu = max(heuristic_vals)
        if max_heu > current_node_heu:
            count += 1
            current_node = successors[heuristic_vals.index(max_heu)]
            current_node_heu = max_heu
        else:
            return (count, False)
    return (count, True)