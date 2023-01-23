from typing import Callable, Tuple
from gameAgent import Problem


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
        successors = problem.get_successor(initial_state)
        heuristic_vals = [heuristic(successor, final_state)
                          for successor in successors]
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
    print(current_node_heu)
    count = 0
    while True:
        if problem.is_goal_state(current_node):
            break
        successors = problem.get_successor(initial_state)
        heuristic_vals = [heuristic(successor, final_state)
                          for successor in successors]
        print(heuristic_vals)
        max_heu = max(heuristic_vals)
        if max_heu > current_node_heu:
            count += 1
            current_node = successors[heuristic_vals.index(max_heu)]
            current_node_heu = max_heu
        else:
            return (count, False)
    return (count, True)
