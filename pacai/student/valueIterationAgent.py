from pacai.agents.learning.value import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent.

    Make sure to read `pacai.agents.learning` before working on this class.

    A `ValueIterationAgent` takes a `pacai.core.mdp.MarkovDecisionProcess` on initialization,
    and runs value iteration for a given number of iterations using the supplied discount factor.

    Some useful mdp methods you will use:
    `pacai.core.mdp.MarkovDecisionProcess.getStates`,
    `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
    `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
    `pacai.core.mdp.MarkovDecisionProcess.getReward`.

    Additional methods to implement:

    `pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
    The q-value of the state action pair (after the indicated number of value iteration passes).
    Note that value iteration does not necessarily create this quantity,
    and you may have to derive it on the fly.

    `pacai.agents.learning.value.ValueEstimationAgent.getPolicy`:
    The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should return None.
    """

    def __init__(self, index, mdp, discountRate=0.9, iters=100, **kwargs):
        super().__init__(index, **kwargs)

        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        self.values = {}  # A dictionary which holds the q-values for each state.

        # Compute the values here.
        states = self.mdp.getStates()  # states

        for iteration in range(self.iters):  # cubes on grid lock
            newValues = dict()
            for state in states:
                max_val = -99999  # worst max vax for each state
                # check if state is terminal??
                # reward if terminal state?
                actions = self.mdp.getPossibleActions(state)  # actions of state
                for action in actions:
                    trans = self.mdp.getTransitionStatesAndProbs(state, action)
                    q_value = 0
                    for nextState, probability in trans:
                        #   temp = max_val
                        reward = self.mdp.getReward(state, action, nextState)
                        prev_val = self.discountRate * self.getValue(nextState)
                        q_value += probability * (reward + prev_val)
                    if q_value > max_val:
                        max_val = q_value
                        newValues[state] = q_value

            self.values = newValues

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """

        return self.values.get(state, 0.0)

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """

        return self.getPolicy(state)

    def getPolicy(self, state):
        policy = None
        max_val = -9999999
        actions = self.mdp.getPossibleActions(state)
        for action in actions:
            q_val = self.getQValue(state, action)
            if q_val > max_val:
                max_val = q_val
                policy = action
        return policy

    def getQValue(self, state, action):
        trans = self.mdp.getTransitionStatesAndProbs(state, action)
        q_value = 0
        for nextState, probability in trans:
            reward = self.mdp.getReward(state, action, nextState)
            prev_val = self.discountRate * self.getValue(nextState)
            q_value += probability * (reward + prev_val)
        return q_value
