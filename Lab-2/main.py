from typing import Union, List, Tuple
from pathlib import Path
from gameAgent import BlockWorldDiagram


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


if __name__ == '__main__':
    initial_state = file_reader(r'input.txt')
    goal_state = file_reader(r'goal.txt')

    prob1 = BlockWorldDiagram(initial_state, goal_state)
    print(prob1.get_successor(initial_state))
