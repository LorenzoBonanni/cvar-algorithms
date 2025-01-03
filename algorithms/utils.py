import numpy as np


class Policy:
    def __init__(self, env):
        self.env = env

    def get_action(self, state):
        raise NotImplementedError

class RandomPolicy(Policy):
    def get_action(self, state):
        return np.random.choice(self.env.ACTIONS)

class ProbabilisticPolicy(Policy):
    def __init__(self, env, policy):
        super().__init__(env)
        self.policy = policy

    def get_action(self, state):
        return np.random.choice(self.env.ACTIONS, p=self.policy[state.id])

class UniformProbabilisticPolicy(ProbabilisticPolicy):
    def __init__(self, env):
        policy = np.ones((env.Ns, len(env.ACTIONS))) / len(env.ACTIONS)
        super().__init__(env, policy)

class FixedPolicy(Policy):
    def __init__(self, env, policy):
        super().__init__(env)

        self.policy = np.zeros((env.Ns, len(env.ACTIONS)), dtype=int)
        for s in env.states():
            self.policy[s.id, policy[s.id]] = 1

    def get_action(self, state):
        return self.policy[state.id].argmax()