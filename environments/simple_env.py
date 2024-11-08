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
    Rewards_s_a_s1 = False


    def __init__(self):
        self.initial_state = State(0)
        if self.Rewards_s_a_s1:
            self.P = np.array([
                # State 0 (Start)
                [[0.0, 0.3, 0.7],  # Action 0: less likely to terminate
                 [0.0, 0.7, 0.3]],  # Action 1: more likely to terminate

                # State 1 (Terminal)
                [[0.0, 1.0, 0.0],  # Action 0: stay terminal (dummy)
                 [0.0, 1.0, 0.0]],  # Action 1: stay terminal (dummy)

                # State 2 (Terminal)
                [[0.0, 0.0, 1.0],  # Action 0: stay terminal (dummy)
                 [0.0, 0.0, 1.0]]  # Action 1: stay terminal (dummy)
            ])

            # Rewards: R[s, a, s']
            self.R = np.array([
                # State 0 (Start)
                [[0.0, 3, 4],   # Action 0: safe action with small loss risk
                 [0.0, 1, 2]],  # Action 1: risky action with bigger loss risk

                # State 1 (Terminal)
                [[0.0, 0.0, 0.0],  # Action 0: no more rewards
                 [0.0, 0.0, 0.0]],  # Action 1: no more rewards

                # State 1 (Terminal)
                [[0.0, 0.0, 0.0],  # Action 0: no more rewards
                 [0.0, 0.0, 0.0]]  # Action 1: no more rewards
            ])
        else:
            self.P = np.array([
                # State 0 (Start)
                [[0.0, 0.3, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Action 0
                 [0.0, 0.0, 0.0, 0.7, 0.3, 0.0, 0.0, 0.0, 0.0]],  # Action 1

                # State 1
                [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Action 0
                 [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],  # Action 1

                # State 2
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # Action 0
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],  # Action 1

                # State 3
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # Action 0
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]],  # Action 1

                # State 4
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Action 0
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],  # Action 1

                # State 5 (Terminal)
                [[0.0, 0.0 , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Action 0
                 [0.0, 0.0 , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], # Action 1

                # State 6 (Terminal)
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Action 0
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],  # Action 1

                # State 7 (Terminal)
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Action 0
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],  # Action 1

                # State 8 (Terminal)
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Action 0
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],  # Action 1
            ])

            # Rewards: R[s, a, s']
            self.R = np.array([
                # State 0 (Start)
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Action 0
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],  # Action 1

                # State 1
                [[0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0],  # Action 0
                 [0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0]],  # Action 1

                # State 2
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0],  # Action 0
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0]],  # Action 1

                # State 3
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0],  # Action 0
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0]],  # Action 1

                # State 4
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Action 0
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],  # Action 1

                # State 5 (Terminal)
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Action 0
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],  # Action 1

                # State 6 (Terminal)
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Action 0
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],  # Action 1

                # State 7 (Terminal)
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Action 0
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],  # Action 1

                # State 8 (Terminal)
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Action 0
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],  # Action 1
            ])

        self.Ns = len(self.P)

    def states(self):
        """ iterator over all possible states """
        if self.Rewards_s_a_s1:
            Ns = self.Ns - 2
        else:
            Ns = self.Ns - 4
        for s in range(Ns):
            yield State(s)

    def transitions(self, s):
        """
        returns a list of Transitions from the state s for each action, only non zero probabilities are given
        serves the lists for all actions at once
        """
        states = [State(s_) for s_ in range(self.Ns)]
        all_transitions = []
        for a in self.ACTIONS:
            action_transitions = []
            for s_ in range(self.Ns):
                if self.P[s.id, a, s_] > 0:
                    action_transitions.append(Transition(state=states[s_], prob=float(self.P[s.id, a, s_]), reward=float(self.R[s.id, a, s_])))
            all_transitions.append(action_transitions)

        return all_transitions

    def actions(self, s):
        return self.ACTIONS

    def sample_transition(self, s, a):
        """ Sample a transition from state s given action a """
        transitions = self.transitions(s)[a]
        probs = [t.prob for t in transitions]
        idx = np.random.choice(len(transitions), p=probs)
        return transitions[idx]

    def is_terminal(self, s):
        if self.Rewards_s_a_s1:
            return s.id >= self.Ns - 2
        else:
            return s.id >= self.Ns - 4

if __name__ == '__main__':
    world = SimpleEnv()
    for s in world.states():
        print(f"State {s}:")
        for a in world.ACTIONS:
            for t in world.transitions(s)[a]:
                print(f"  Action {world.ACTION_NAMES[a]} -> State {t.state}, prob {t.prob}, reward {t.reward}")
        print()