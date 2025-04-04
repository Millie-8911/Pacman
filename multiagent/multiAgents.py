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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        
        foodList = newFood.asList()
        score = successorGameState.getScore()
        minFoodDist = float("inf")
        nearbyFoodDist = []

        # Higher rewards being closer to food
        # Add (1/distance) multiplied by 10
        if foodList:
            minFoodDist = min(manhattanDistance(newPos, food) for food in foodList)
            score += 10 / minFoodDist

        # Rewards being closer to multiple food pellets
        # Considers food within 3 units
        # Uses sum of inverse distances * 3
        for food in foodList:
            foodDist = manhattanDistance(newPos, food)
            if foodDist < 3:  
                nearbyFoodDist.append(1 / foodDist)
        score += sum(nearbyFoodDist) * 3  

        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostDist = manhattanDistance(newPos, ghostState.getPosition())

            # Scared ghosts: Rewards being closer
            if scaredTime > 0:
                if ghostDist > 0:
                    score += 15 / ghostDist  
                else:
                    score += 100  # Bonus if Pacman reaches a scared ghost

            # Normal ghosts
            else:
                if ghostDist < 3:  # Heavy penalty for being too close
                    score -= 500
                else:
                    score -= 2 / ghostDist  # Small penalty for ghost proximity

        # Penalty if more food left
        score -= len(foodList) * 4

        # Penalty if stay still
        if action == Directions.STOP:
            score -= 20

        return score

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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

        def getNextAgent(agentIndex):
            if agentIndex == gameState.getNumAgents() -1:
                return 0
            else:
                return agentIndex + 1

        def minimax(gameState, depth, agentIndex):
            # Terminate conditions: when reach to the last depth or game end
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            # Pacman's turn - MAX
            # Look for max scores among all the possible actions
            if agentIndex == 0:
                return max(
                    minimax(gameState.generateSuccessor(agentIndex, action), depth, 1)
                    for action in gameState.getLegalActions(agentIndex)
                )

            # Ghost's turn - MIN
            else:
                # After all the ghosts finish moving
                if getNextAgent(agentIndex) == 0:
                    depth += 1

                return min(
                    minimax(gameState.generateSuccessor(agentIndex, action), depth, getNextAgent(agentIndex))
                    for action in gameState.getLegalActions(agentIndex)
                )

        legalSteps = gameState.getLegalActions(0)
        scores = [minimax(gameState.generateSuccessor(0, action), 0, 1) for action in legalSteps]
        return legalSteps[scores.index(max(scores))]


class AlphaBetaAgent(MultiAgentSearchAgent):  
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action with alpha-beta pruning
        """

        # Initialize a(alpha) and b(beta)
        a = float("-inf")
        b = float("inf")
        bestVal = float("-inf")
        bestAct = None
        
        # Retrieve all legal actions for Pacman (agentIndex = 0)
        legalSteps = gameState.getLegalActions(0) 
        
        for action in legalSteps:
            successor = gameState.generateSuccessor(0, action)
            val = self.minValue(successor, 0, 1, a, b)
            
            # If the value > current alpha, then replace alpha
            if val > bestVal:
                bestVal = val
                bestAct = action
            
            a = max(a, bestVal)
        
        return bestAct

    def maxValue(self, gameState, depth, agentIndex, a, b):
        """
        Calculate the value for max nodes for Pacman's.
        """
        
        val = float("-inf")
        legalSteps = gameState.getLegalActions(agentIndex)

        # Terminate conditions: when reach to the last depth or game end
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
            
        for action in legalSteps:
            successor = gameState.generateSuccessor(agentIndex, action)
            val = max(val, self.minValue(successor, depth, agentIndex + 1, a, b))
            
            # Prune when value greater than b
            if val > b:  
                return val
            a = max(a, val)
            
        return val

    def minValue(self, gameState, depth, agentIndex, a, b):
        """
        Calculate the value for a min nodes for ghosts.
        """

        # Get the number of ghosts
        numGhosts = gameState.getNumAgents() - 1
        val = float("inf")
        legalSteps = gameState.getLegalActions(agentIndex)

        # Terminate conditions: when reach to the last depth or game end
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        for action in legalSteps:
            successor = gameState.generateSuccessor(agentIndex, action)
            
            # If this is the last ghost
            if agentIndex == numGhosts:
                # Evaluate the next depth for Pacman
                val = min(val, self.maxValue(successor, depth + 1, 0, a, b))
            else:
                # Evaluate the next ghost
                val = min(val, self.minValue(successor, depth, agentIndex + 1, a, b))
            
            # Prune when value is less than a
            if val < a:  # Prune
                return val
            b = min(b, val)
            
        return val


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.

        Returns optimal action for Pacman
        """
        "*** YOUR CODE HERE ***"
        
        def expectimax(agentIndex, depth, gameState):

            # Terminate conditions: when reach to the last depth or game end
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pacman - Max
            if agentIndex == 0:
                val = float('-inf') # Initialize variable to store the max value
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    val = max(val, expectimax(1, depth, successor))
                return val
            else:
                # Ghosts - Expect
                numGhosts = gameState.getNumAgents() - 1
                val = 0 # # Initialize sum of values
                legalActs = gameState.getLegalActions(agentIndex)
                numActs = len(legalActs)

                for action in legalActs:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    # Calculate the expected value
                    if agentIndex == numGhosts: # If current ghost is the last ghost,
                        val += expectimax(0, depth + 1, successor)  # Go back to Pacman
                    else:
                        val += expectimax(agentIndex + 1, depth, successor) # Continue to the next ghost

                # Return the average value for ghost
                return val / numActs if numActs > 0 else 0

        # Find the best score/action for Pacman
        # Use max with key to select the action with the highest score
        actions = gameState.getLegalActions(0)
        return max(actions, key=lambda action: expectimax(1, 0, gameState.generateSuccessor(0, action)))


def betterEvaluationFunction(currentGameState: GameState):
    """
    This function includes:
    Left food and the distance in between,
    Position and the distance to ghosts,
    Current game score
    """

    # Get the current position
    pos = currentGameState.getPacmanPosition()
    
    # Get the position of food
    food = currentGameState.getFood()
    foodList = food.asList()
    
    # Get the position of ghosts
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    
    # Get current score
    score = currentGameState.getScore()
    
    # Left food and the distance in between
    if len(foodList) > 0: # Still have left food
        # Calculate the distance to all the food
        foodDist = [manhattanDistance(pos, food) for food in foodList]
        minDist = min(foodDist)
        # Penalty for left food and encourage to be closer
        penalty = 5 * len(foodList) + minDist 
        score -= penalty
    
    # Position and the distance to ghosts
    for i, ghostState in enumerate(ghostStates):
        ghostDist = manhattanDistance(pos, ghostState.getPosition())
        # If it's a scared ghost
        if scaredTimes[i] > 0:
            # Encourage Pacman to be closer
            if ghostDist == 0:
                score += 500
            score += 100 / (ghostDist + 1)
        else:
            # Avoid normal ghost
            if ghostDist <= 1:
                score -= 500
            elif ghostDist <= 3:
                score -= 200
            else: # Keep suitable distance from the ghosts
                score -=  3 * ghostDist
    
    return score


# Abbreviation
better = betterEvaluationFunction
