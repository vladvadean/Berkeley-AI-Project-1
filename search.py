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
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    '''
        current_state=tuple((x,y),direction_from_parent,cost)
        current_state[0][0]=x
        current_state[0][1]=y
        current_state[1]=direction_from_parent
        current_state[2]=costSS
    '''
    start_state = problem.getStartState()
    stack = util.Stack()
    visited = set()
    visited.add(start_state)
    start_node = [start_state, []]
    stack.push(start_node)

    while not stack.isEmpty():
        current_state, moves = stack.pop()
        visited.add(current_state)
        if problem.isGoalState(current_state):
            return moves
        for next_state in problem.getSuccessors(current_state):
            if next_state[0] not in visited:
                new_moves = moves + [next_state[1]]
                node_aux = [next_state[0], new_moves]
                stack.push(node_aux)


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    start_state = problem.getStartState()
    queue = util.Queue()
    visited = set()
    visited.add(start_state)
    start_node = [start_state, []]
    queue.push(start_node)

    while not queue.isEmpty():
        current_state, moves = queue.pop()
        if problem.isGoalState(current_state):
            return moves
        for next_state in problem.getSuccessors(current_state):
            if next_state[0] not in visited:
                new_moves = moves + [next_state[1]]
                node_aux = [next_state[0], new_moves]
                visited.add(next_state[0])
                queue.push(node_aux)


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    """Search the node of least total cost first."""
    start_state = problem.getStartState()
    priority_queue = util.PriorityQueue()
    visited = set()
    # (x,y),action,cost
    start_node = (start_state, [], 0)
    priority_queue.push(start_node, 0)

    while not priority_queue.isEmpty():
        current_state, moves, cost = priority_queue.pop()
        if current_state not in visited:
            visited.add(current_state)
            if problem.isGoalState(current_state):
                return moves
            for next_state in problem.getSuccessors(current_state):
                new_moves = moves + [next_state[1]]
                node_aux = (next_state[0], new_moves, next_state[2] + cost)
                priority_queue.push(node_aux, next_state[2] + cost)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    priority_queue = util.PriorityQueue()
    # (x,y),cost
    visited = set()
    start_state = problem.getStartState()
    start_node = (start_state, [], 0)
    priority_queue.push(start_node, 0)

    while not priority_queue.isEmpty():
        current_state, moves, cost = priority_queue.pop()
        visited.add((current_state, cost))
        if problem.isGoalState(current_state):
            return moves
        for next_node in problem.getSuccessors(current_state):
            new_moves = moves + [next_node[1]]
            new_cost = problem.getCostOfActions(new_moves)
            node_aux = (next_node[0], new_moves, new_cost)
            in_visited = False
            for visited_node in visited:
                if (next_node[0] == visited_node[0]) and (new_cost >= visited_node[1]):
                    in_visited = True
            if not in_visited:
                priority_queue.push(node_aux, new_cost + heuristic(next_node[0], problem))
                visited.add((next_node[0], new_cost))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
