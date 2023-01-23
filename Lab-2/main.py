from typing import Union, List, Tuple, Callable
from pathlib import Path
from gameAgent import *
# from searchAlgo import *


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

def hillClimbMod(problem: Problem, heuristic: Callable[..., int]) -> Tuple[int, bool]:
    """It will return if the end_goal is rechable or not using 
    the current heuristics, using greedy approach, currently maximizing
    the heuristics value"""
    initial_state = problem.get_initial_state()
    final_state = problem.get_goal_state()
    current_node = initial_state
    current_node_heu = heuristic(current_node, final_state)
    print(current_node_heu)
    count = 0
    while True:
        if problem.is_goal_state(current_node):
            break
        successors = moveGen(initial_state)
        print(successors)
        heuristic_vals = [heuristic(successor, final_state) for successor in successors]
        print(heuristic_vals)
        max_heu = max(heuristic_vals)
        if max_heu != current_node_heu:
            count += 1
            current_node = successors[heuristic_vals.index(max_heu)]
            print(current_node)
            current_node_heu = max_heu
        else:
            return (count, False)
    return (count, True)

if __name__ == '__main__':
    initial_state = file_reader(r'test\input2.txt')
    goal_state = file_reader(r'test\goal2.txt')

    prob1 = BlockWorldDiagram(initial_state, goal_state)
    # succ = prob1.get_successor(initial_state)
    # for su in succ:
    #     print(manhattan_heuristic_maxi(su, goal_state))
    #     print(xnor_heuristic(su, goal_state))
    #     print(xnor_heuristic_modified(su, goal_state))
    #     print(ascii_heuristic(su, goal_state))
    #     print()

    print(hillClimbMod(prob1, xnor_heuristic_modified))
