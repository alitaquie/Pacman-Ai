import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core.distance import manhattan
from pacai.core.directions import Directions


class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Useful information you can extract.
        # newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        # *** Your Code Here ***
        newPosition = successorGameState.getPacmanPosition()
        newGhostStates = successorGameState.getGhostStates()
        currentGhostStates = currentGameState.getGhostStates()
        newFoodStates = successorGameState.getFood().asList()
        score = 0
        ghost_dist = []
        ghost_dist2 = []
        food_dist = []
        scared = []

        if successorGameState.isWin():
            return 99999

        score = 0
        if action == "Stop":
            score -= 5

        for ghost in currentGhostStates:
            ghost = ghost.getPosition()
            ghost_dist.append(manhattan(newPosition, ghost))
            n_ghost = min(ghost_dist)

        for ghost in newGhostStates:
            ghost_pos = ghost.getPosition()
            ghost_dist2.append(manhattan(newPosition, ghost_pos))
            scared.append(ghost.getScaredTimer())
            n2_ghost = min(ghost_dist2)

        if scared:
            min_scared = min(scared)

        for food in newFoodStates:
            food_dist.append(manhattan(newPosition, food))
        near_food = min(food_dist)

        if n2_ghost < n_ghost:
            nearest_ghost = n2_ghost
            score -= 150
        else:
            nearest_ghost = n_ghost
            score += 300

        score += (float(1 / near_food) + 0.1) - (1 / (int(nearest_ghost) + 0.1))

        score -= len(newFoodStates)

        return successorGameState.getScore() + score + min_scared


class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        self.depth = self.getTreeDepth()

    def getAction(self, state):
        LegalMoves = state.getLegalActions(0)
        score = -999999
        new_move = Directions.STOP
        for moves in LegalMoves:
            successorState = state.generateSuccessor(0, moves)
            new_score = self.minimin(successorState, 0, 1)
            if new_score > score:
                new_move = moves
                score = new_score
        return new_move

    def terminal(self, state, depth, moves):
        # terminal states check
        if state.isWin():
            return True
        if state.isLose():
            return True
        if state.isOver():
            return True

        # depth check
        if depth == self.depth:
            return True

        # terminal state action checks
        if not moves:
            return True
        if len(moves) == 0:
            return True

    def maximin(self, state, depth):
        depth_level = depth + 1
        max_val = -999999
        turn = 1
        LegalMoves = state.getLegalActions(0)

        if self.terminal(state, depth_level, LegalMoves):  # Terminal Test
            return self.getEvaluationFunction()(state)

        for move in LegalMoves:
            successorState = state.generateSuccessor(0, move)
            max_val = max(max_val, self.minimin(successorState, depth_level, turn))
        return max_val

    def minimin(self, state, depth, turn):
        min_val = 999999
        LegalMoves = state.getLegalActions(turn)
        num_agents = state.getNumAgents()

        if self.terminal(state, depth, LegalMoves):
            return self.getEvaluationFunction()(state)

        for moves in LegalMoves:
            successorStates = state.generateSuccessor(turn, moves)
            if turn == num_agents - 1:
                min_val = min(min_val, self.maximin(successorStates, depth))
            else:
                min_val = min(min_val, self.minimin(successorStates, depth, turn + 1))
            return min_val


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        self.depth = self.getTreeDepth()

    def getAction(self, state):
        alpha, beta = float("-inf"), float("inf")
        LegalMoves = state.getLegalActions(0)
        score = -999999
        new_move = Directions.STOP
        for moves in LegalMoves:
            successorState = state.generateSuccessor(0, moves)
            new_score = self.minimin(successorState, 0, 1, alpha, beta)
            if new_score > score:
                new_move = moves
                score = new_score
        return new_move

    def terminal(self, state, depth, moves):
        # terminal states check
        if state.isWin():
            return True
        if state.isLose():
            return True
        if state.isOver():
            return True

        # depth check
        if depth == self.depth:
            return True

        # terminal state action checks
        if not moves:
            return True
        if len(moves) == 0:
            return True

    def maximin(self, state, depth, alpha, beta):
        depth_level = depth + 1
        score = -999999
        turn = 1
        LegalMoves = state.getLegalActions(0)

        if self.terminal(state, depth_level, LegalMoves):  # Terminal Test
            return self.getEvaluationFunction()(state)

        for move in LegalMoves:
            successorState = state.generateSuccessor(0, move)
            score = max(
                score, self.minimin(successorState, depth_level, turn, alpha, beta)
            )
            if score > beta:
                return score
            alpha = max(alpha, score)
        return score

    def minimin(self, state, depth, turn, alpha, beta):
        score = 999999
        LegalMoves = state.getLegalActions(turn)
        num_agents = state.getNumAgents()

        if self.terminal(state, depth, LegalMoves):
            return self.getEvaluationFunction()(state)

        for moves in LegalMoves:
            successorStates = state.generateSuccessor(turn, moves)
            if turn == num_agents - 1:
                score = min(score, self.maximin(successorStates, depth, alpha, beta))
            else:
                score = min(
                    score, self.minimin(successorStates, depth, turn + 1, alpha, beta)
                )
            if score < alpha:
                return score
            beta = min(beta, score)
            return score


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        self.depth = self.getTreeDepth()

    def getAction(self, state):
        LegalMoves = state.getLegalActions(0)
        score = -999999
        new_move = Directions.STOP
        for moves in LegalMoves:
            successorState = state.generateSuccessor(0, moves)
            new_score = self.minimin(successorState, 0, 1)
            if new_score > score:
                new_move = moves
                score = new_score
        return new_move

    def terminal(self, state, depth, moves):
        # terminal states check
        if state.isWin():
            return True
        if state.isLose():
            return True
        if state.isOver():
            return True

        # depth check
        if depth == self.depth:
            return True

        # terminal state action checks
        if not moves:
            return True
        if len(moves) == 0:
            return True

    def maximin(self, state, depth):
        depth_level = depth + 1
        max_val = -999999
        turn = 1
        LegalMoves = state.getLegalActions(0)

        if self.terminal(state, depth_level, LegalMoves):  # Terminal Test
            return self.getEvaluationFunction()(state)

        for move in LegalMoves:
            successorState = state.generateSuccessor(0, move)
            max_val = max(max_val, self.minimin(successorState, depth_level, turn))
        return max_val

    def minimin(self, state, depth, turn):
        min_val = 0
        LegalMoves = state.getLegalActions(turn)
        num_agents = state.getNumAgents()
        num_moves = len(LegalMoves)

        if self.terminal(state, depth, LegalMoves):
            return self.getEvaluationFunction()(state)

        for moves in LegalMoves:
            successorStates = state.generateSuccessor(turn, moves)
            if turn == num_agents - 1:
                min_val = self.maximin(successorStates, depth)
            else:
                min_val = self.minimin(successorStates, depth, turn + 1)
        if num_moves == 0:
            return 0
        return float(min_val) / float(num_moves)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>
    """

    currentGhostStates = currentGameState.getGhostStates()
    currentPosition = currentGameState.getPacmanPosition()
    currentFoodStates = currentGameState.getFood().asList()
    score = 0
    min_scared = 0
    ghost_dist = []
    food_dist = []
    scared = []

    if currentGameState.isWin():
        return 99999

    score = 0

    for ghost in currentGhostStates:
        ghost = ghost.getPosition()
        ghost_dist.append(manhattan(currentPosition, ghost))
        n_ghost = min(ghost_dist)

    if scared:
        min_scared = min(scared)

    for food in currentFoodStates:
        food_dist.append(manhattan(currentPosition, food))
    near_food = min(food_dist)

    score += (10 / (near_food + 0.1)) - (10 / (int(n_ghost) + 0.1))

    return currentGameState.getScore() + score + min_scared


class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
