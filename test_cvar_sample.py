import numpy as np
from joblib import delayed, Parallel

from algorithms.utils import UniformProbabilisticPolicy
from environments.simple_env import SimpleEnv, State

def get_return(env, policy, gamma):
    ret = 0
    state = State(0)
    i = 0
    while not env.is_terminal(state):
        action = policy.get_action(state)
        t = env.sample_transition(state, action)
        ret += gamma ** i * t.reward
        i += 1
        state = t.state
    return ret

def gen_sample():
    num_samples = 100_000
    gamma = 0.95
    env = SimpleEnv()
    policy = UniformProbabilisticPolicy(env)
    returns = Parallel(n_jobs=-1, verbose=False)(delayed(get_return)(env, policy, gamma) for i in range(num_samples))

    return returns

Ny = 21
alphas = np.concatenate(([0], np.logspace(-2, 0, Ny - 1)))
for alpha in alphas:
    s = np.array(gen_sample())
    s.sort()
    cvar = np.mean(s[s <= np.quantile(s, alpha)])
    print(f"CVaR({alpha}) = {cvar}")