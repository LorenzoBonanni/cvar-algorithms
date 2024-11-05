import numpy as np
from joblib import delayed, Parallel

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

def policy_eval_montecarlo(alphas, policy, gamma, env, num_samples=1000):
    s = np.array(Parallel(n_jobs=-1, verbose=False)(delayed(get_return)(env, policy, gamma) for _ in range(num_samples)))
    s.sort()
    values = []
    for alpha in alphas:
        cvar = np.mean(s[s <= np.quantile(s, alpha)])
        values.append(cvar)

    return values