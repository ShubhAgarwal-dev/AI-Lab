from typing import Callable
from gameAgent import Problem


def hillClimb(problem: Problem, heuristic: Callable[..., int]) -> bool:
    """It will return if the end_goal is rechable or not using 
    the current heuristics, using greedy approach"""
    initial_state = problem.get_initial_state()
    final_state = problem.get_goal_state()
    return True
