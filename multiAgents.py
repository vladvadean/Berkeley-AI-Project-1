# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newGhostPositions = successorGameState.getGhostPositions()

        "*** YOUR CODE HERE ***"
        # a bigger weight for the total distance away from ghosts emerges victorious but at a very low score
        # currentGameState.getScore() + ghostDistanceComponent * 1000 + foodDistanceComponent
        # ghostDistanceComponent = (nextTotalDistanceAwayFromGhosts - currentTotalDistanceAwayFromGhosts)
        # foodDistanceComponent = (currentTotalDistanceAwayFromFood - nextTotalDistanceAwayFromFood)
        # the more complex formula the worse

        foodComponent = 9999999
        minGhostDistance = 999999999

        for position in newGhostPositions:
            dist = manhattanDistance(position, newPos)
            if minGhostDistance < dist:
                minGhostDistance = dist

        if minGhostDistance == 999999999:
            minGhostDistance = 1

        for food in successorGameState.getFood().asList():
            foodComponent = min(foodComponent, manhattanDistance(newPos, food))

        for ghost in newGhostPositions:
            if (manhattanDistance(newPos, ghost) < 2):
                return -999999

        return successorGameState.getScore() + 1.0 / minGhostDistance + 1.0 / foodComponent


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # agentIndex == 0 for pacman
        # agentIndex !=0 for ghosts
        def processingMinMax(gameState, depth, agentIndex):

            if agentIndex == 0:
                value = -999999
            else:
                value = 999999

            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            if agentIndex == 0 and depth + 1 == self.depth:
                return self.evaluationFunction(gameState)

            legalActions = gameState.getLegalActions(agentIndex)

            for action in legalActions:
                succesorState = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == 0:
                    value = max(value, processingMinMax(succesorState, depth + 1, 1))
                else:
                    if agentIndex == (gameState.getNumAgents() - 1):
                        value = min(value, processingMinMax(succesorState, depth, 0))
                    else:
                        value = min(value, processingMinMax(succesorState, depth, agentIndex + 1))

            return value

        # start to generate the decision tree starting with the pacman
        actions = gameState.getLegalActions(0)
        currentScore = -999999
        result = ''

        for action in actions:
            nextState = gameState.generateSuccessor(0, action)
            score = processingMinMax(nextState, 0, 1)
            if score > currentScore:
                result = action
                currentScore = score

        return result


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # agentIndex == 0 for pacman
        # agentIndex != 0 for ghosts
        def processingMinMax(gameState, depth, agentIndex, alpha, beta):

            if gameState.isWin() == True or gameState.isLose() == True:
                return self.evaluationFunction(gameState)

            if agentIndex == 0 and depth + 1 == self.depth:
                return self.evaluationFunction(gameState)

            if agentIndex == 0:
                value = -9999999
            else:
                value = 9999999

            legalActions = gameState.getLegalActions(agentIndex)
            alphaCopy = alpha
            betaCopy = beta

            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == 0:
                    value = max(value, processingMinMax(successor, depth + 1, 1, alphaCopy, beta))
                    if value > beta:
                        return value
                    alphaCopy = max(alphaCopy, value)
                else:
                    if agentIndex == (gameState.getNumAgents() - 1):
                        value = min(value, processingMinMax(successor, depth, 0, alpha, betaCopy))
                    else:
                        value = min(value, processingMinMax(successor, depth, agentIndex + 1, alpha, betaCopy))
                    if value < alpha:
                        return value
                    betaCopy = min(betaCopy, value)
            return value

        # start by the root processing the pacman
        actions = gameState.getLegalActions(0)
        currentScore = -999999
        returnAction = ''
        alpha = -999999
        beta = 999999
        for action in actions:
            nextState = gameState.generateSuccessor(0, action)
            score = processingMinMax(nextState, 0, 1, alpha, beta)
            if score > currentScore:
                returnAction = action
                currentScore = score
            if score > beta:
                return returnAction
            alpha = max(alpha, score)
        return returnAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
