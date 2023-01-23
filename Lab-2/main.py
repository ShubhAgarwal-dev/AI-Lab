from typing import Union, List, Tuple
from pathlib import Path
from gameAgent import *
from searchAlgo import *


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
    initial_state = file_reader(r'test\input2.txt')
    goal_state = file_reader(r'test\goal2.txt')

    print(initial_state) 
    stacks = encoder(initial_state)
    print(stacks)
    print(decoder(stacks))

    # prob1 = BlockWorldDiagram(initial_state, goal_state)
    # succ = prob1.get_successor(initial_state)
    # for su in succ:
    #     print(manhattan_heuristic_maxi(su, goal_state))
    #     print(xnor_heuristic(su, goal_state))
    #     print(xnor_heuristic_modified(su, goal_state))
    #     print(ascii_heuristic(su, goal_state))
    #     print()

    # print(hillClimbMod(prob1, xnor_heuristic_modified))
