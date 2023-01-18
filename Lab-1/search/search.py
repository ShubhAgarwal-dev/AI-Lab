# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import heapq

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    stack = util.Stack() # stack will include coordinates and path
    visited = []
    path = [] # list of the directions agent is required to take
    initial_state = problem.getStartState()
    stack.push((initial_state, []))

    while True:
        if stack.isEmpty():
            # empty stack represents FALIURE
            return []
        selected_state, path = stack.pop()
        visited.append(selected_state)
        if problem.isGoalState(selected_state):
            break
        successors = problem.getSuccessors(selected_state)
        if successors:
            for successor in successors:
                new_coordinate, new_direction, _ = successor
                if new_coordinate not in visited:
                    new_path = path + [new_direction]
                    stack.push((new_coordinate, new_path))

    return path

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    queue_path = util.Queue()
    queue_state = util.Queue()
    visited = []
    path = []
    initial_state = problem.getStartState()
    queue_path.push(path)
    queue_state.push(initial_state)

    while True:
        if queue_state.isEmpty():
            return [] # represents failed state
        selected_state = queue_state.pop()
        path = queue_path.pop()
        visited.append(selected_state)
        if problem.isGoalState(selected_state):
            break
        successors = problem.getSuccessors(selected_state)
        if successors:
            for successor in successors:
                new_coordinate, new_direction, _ = successor
                if new_coordinate not in visited and new_coordinate not in queue_state.list:
                    new_path = path + [new_direction]
                    queue_path.push(new_path)
                    queue_state.push(new_coordinate)
    return path

class ExtendedPriorotyQueue(util.PriorityQueue):
    
    def __init__(self):
        util.PriorityQueue.__init__(self)

    def pop_new(self):
        (cost, _, item) = heapq.heappop(self.heap)
        return (item, cost)

    def get_cost(self, item):
        for cost, _, i in self.heap:
            if i == item:
                return cost
        return 0
def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    initial_state = problem.getStartState()
    frontier = ExtendedPriorotyQueue()
    frontier.push(initial_state, 0)
    path_dic = {initial_state:[]}
    explored =[]
    while True:
        if frontier.isEmpty():
            return [] 
        sel_state, cost= frontier.pop_new()
        explored.append(sel_state)
        path = path_dic.pop(sel_state)
        if problem.isGoalState(sel_state):
            return path
        successors = problem.getSuccessors(sel_state)
        if successors:
            for child in successors:
                new_state, new_dir, add_cost = child
                new_path = path + [new_dir]
                new_cost = cost + add_cost
                if new_state not in explored and new_state not in path_dic.keys():
                    frontier.push(new_state, new_cost)
                    path_dic.update({new_state:new_path})
                elif new_state in path_dic.keys():
                    old_cost = frontier.get_cost(new_state)
                    if old_cost > new_cost:
                        frontier.update(new_state, new_cost)
                        path_dic[new_state] = new_path
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    initial_state = problem.getStartState()
    frontier = ExtendedPriorotyQueue()
    frontier.push(initial_state, heuristic(initial_state, problem))
    path_dic = {initial_state:[]}
    explored = []
    while True:
        if frontier.isEmpty():
            return [] 
        sel_state, cost= frontier.pop_new()
        explored.append(sel_state)
        path = path_dic.pop(sel_state)
        if problem.isGoalState(sel_state):
            return path
        successors = problem.getSuccessors(sel_state)
        if successors:
            old_heu = heuristic(sel_state, problem)
            for child in successors:
                new_state, new_dir, add_cost = child
                new_path = path + [new_dir]
                new_cost = cost + add_cost + heuristic(new_state, problem) - old_heu
                if new_state not in explored and new_state not in path_dic.keys():
                    frontier.push(new_state, new_cost)
                    path_dic.update({new_state:new_path})
                    # print(frontier.heap[i][2] for i in range(len(frontier.heap)))
                elif new_state in path_dic.keys():
                    old_cost = frontier.get_cost(new_state)
                    if old_cost >= new_cost:
                        frontier.update(new_state, new_cost)
                        path_dic[new_state] = new_path
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
