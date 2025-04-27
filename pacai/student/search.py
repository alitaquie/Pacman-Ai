"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""
from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """

    # *** Your Code Here ***
    start = problem.startingState()  # starting node ?froniter?
    fringe = Stack()  # nodes we need to visit
    visited = set()  # nodes we've visited
    # push start state into fringe
    fringe.push((start, [], 0))
    visited.add(start)
    while not fringe.isEmpty():
        if fringe.isEmpty():
            return None

        state, path, cost = fringe.pop()

        if state not in visited:
            visited.add(state)

        if problem.isGoal(state) is True:
            return path

        for child, c_path, ccost in problem.successorStates(state):
            if child not in visited:
                next_path = path + [c_path]
                total_cost = cost + ccost
                fringe.push((child, next_path, total_cost))
    return None

    # choose a leaf node and remove it from the frontier


#  raise NotImplementedError()


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # *** Your Code Here ***
    start = problem.startingState()  # starting node ?froniter?
    fringe = Queue()  # nodes we need to visit
    visited = set()  # nodes we've visited
    # push start state into fringe
    fringe.push((start, [], 0))
    while not fringe.isEmpty():
        state, path, cost = fringe.pop()
        if problem.isGoal(state):
            return path
        if state not in visited:
            visited.add(state)
            for child, c_path, ccost in problem.successorStates(state):
                next_path = path + [c_path]
                total_cost = cost + ccost
                fringe.push((child, next_path, total_cost))
    return []

    # # raise NotImplementedError()


def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """
    # *** Your Code Here ***
    start = problem.startingState()
    fringe = PriorityQueue()
    visited = set()
    fringe.push((start, [], 0), 0)

    while not fringe.isEmpty():
        state, path, cost = fringe.pop()

        if problem.isGoal(state):
            return path

        if state not in visited:
            visited.add(state)
            for child, c_path, ccost in problem.successorStates(state):
                next_path = path + [c_path]
                total_cost = cost + ccost
                fringe.push((child, next_path, total_cost), total_cost)
    return []

    # raise NotImplementedError()


def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***
    start = problem.startingState()
    fringe = PriorityQueue()
    visited = set()
    fringe.push((start, [], 0), 0)

    while not fringe.isEmpty():
        state, path, cost = fringe.pop()
        if problem.isGoal(state) is True:
            return path
        if state not in visited:
            visited.add(state)
            for child, c_path, ccost in problem.successorStates(state):
                next_path = path + [c_path]
                total_cost = cost + ccost
                h_val = heuristic(state, problem) + total_cost
                fringe.push((child, next_path, total_cost), h_val)
    return []
    # raise NotImplementedError()
