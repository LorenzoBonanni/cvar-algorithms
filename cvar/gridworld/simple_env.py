from collections import namedtuple

import numpy as np

Transition = namedtuple('Transition', ['state', 'prob', 'reward'])  # transition to state with probability prob

class State:
    def __init__(self, id):
        self.id = id

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return self.id

class SimpleEnv:
    """ Simple environment with three states and two actions. """

    STAY = 0
    SWITCH = 1
    ACTIONS = [STAY, SWITCH]
    FALL_REWARD = -2
    ACTION_NAMES = {STAY: "Stay", SWITCH: "Switch"}

    def __init__(self):
        # Transition probabilities: P[s, a, s']
        self.P = np.array([
            # State 0 (Start)
            [[0.8, 0.2, 0.0],  # Action 0: likely stay in start
             [0.3, 0.7, 0.0]],  # Action 1: likely go to risky

            # State 1 (Risky)
            [[0.0, 0.5, 0.5],  # Action 0: might end
             [0.0, 0.2, 0.8]],  # Action 1: likely end

            # State 2 (Terminal)
            [[1.0, 0.0, 0.0],  # Action 0: stay terminal (dummy)
             [1.0, 0.0, 0.0]]  # Action 1: stay terminal (dummy)
        ])

        # Rewards: R[s, a, s']
        self.R = np.array([
            # State 0 (Start)
            [[1.0, 0.0, 0.0],  # Action 0: safe, steady reward
             [0.0, 2.0, 0.0]],  # Action 1: potential higher reward

            # State 1 (Risky)
            [[0.0, 1.0, -2.0],  # Action 0: moderate risk
             [0.0, 3.0, -4.0]],  # Action 1: high risk, high reward

            # State 2 (Terminal)
            [[0.0, 0.0, 0.0],  # Action 0: no more rewards
             [0.0, 0.0, 0.0]]  # Action 1: no more rewards
        ])
        self.Ns = 3

    def states(self):
        """ iterator over all possible states """
        for s in range(2):
            yield State(s)

    def transitions(self, s):
        """
        returns a list of Transitions from the state s for each action, only non zero probabilities are given
        serves the lists for all actions at once
        """
        states = [State(0), State(1), State(2)]
        all_transitions = []
        for a in self.ACTIONS:
            action_transitions = []
            for s_ in range(3):
                if self.P[s.id, a, s_] > 0:
                    action_transitions.append(Transition(state=states[s_], prob=float(self.P[s.id, a, s_]), reward=float(self.R[s.id, a, s_])))
            all_transitions.append(action_transitions)

        return all_transitions
        # return [[Transition(state=states[s_], prob=float(self.P[s, a, s_]), reward=float(self.R[s, a, s_])) for s_ in range(3) if self.P[s, a, s_] > 0] for a in self.ACTIONS]

if __name__ == '__main__':
    world = SimpleEnv()
    for s in world.states():
        print(f"State {s}:")
        for a in world.ACTIONS:
            for t in world.transitions(s)[a]:
                print(f"  Action {world.ACTION_NAMES[a]} -> State {t.state}, prob {t.prob}, reward {t.reward}")
        print()