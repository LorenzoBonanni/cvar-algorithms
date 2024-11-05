import numpy as np

from compare_V import policy
from simple_env import SimpleEnv, State

class Policy:
    def __init__(self, env):
        self.env = env

    def get_action(self, state):
        raise NotImplementedError

class RandomPolicy(Policy):
    def get_action(self, state):
        return np.random.choice(self.env.ACTIONS)

def gen_sample():
    num_samples = 100_000
    gamma = 1
    env = SimpleEnv()
    policy = RandomPolicy(env)
    returns = []
    for _ in range(num_samples):
        ret = 0
        state = State(0)
        i = 0
        while True:
            action = policy.get_action(state)
            t = env.sample_transition(state, action)
            ret += gamma ** i * t.reward
            i+=1
            if t.state != State(0) :
                break

        returns.append(ret)

    return returns

alpha = 1
s = np.array(gen_sample())
s.sort()
cvar = np.mean(s[s <= np.quantile(s, alpha)])
print(cvar)