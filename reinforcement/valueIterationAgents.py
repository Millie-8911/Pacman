# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
        Run the value iteration algorithm. Note that in standard
        value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        for i in range(self.iterations):
            # Store the updated values for this iteration
            values = util.Counter()
            
            # Iterate through all states
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state): 
                    values[state] = 0 # If the state is terminal
                else:
                    # Store Q-values for ossible actions
                    action_q_values = []
                    
                    for action in self.mdp.getPossibleActions(state):
                        q = self.computeQValueFromValues(state, action)
                        action_q_values.append(q)
                        
                    if action_q_values:
                        values[state] = max(action_q_values) # Update the state's value
                    else:
                        values[state] = 0 # No actions can do
            
            self.values = values


    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
        Compute the Q-value of action in state from the
        value function stored in self.values.
        According to the formula: 
        Q(s, a) = sum_{s'} P(s'|s, a) * [R(s, a, s') + gamma * V(s')]
        """
        q = 0 # Initialize the Q-value
        # Iterate through all the states and the corresponding probability
        for next_state, pro in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, next_state)
            q += pro * (reward + self.discount * self.values[next_state])
        return q


    def computeActionFromValues(self, state):
        """
        The policy is the best action in the given state
        according to the values currently stored in self.values.

        You may break ties any way you see fit.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
        """
        best_action = None
        max_q = float('-inf')

        if self.mdp.isTerminal(state):
            return None # When it's already at terminal states
            
        actions = self.mdp.getPossibleActions(state)
        for action in actions:
            # Compute the Q-value for the current action
            q_value = self.computeQValueFromValues(state, action)
            # Update the manimum Q-value and the action
            if q_value > max_q:
                max_q = q_value
                best_action = action

        return best_action


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
