import pickle

import numpy as np

from algorithms.cvar_policy_eval_montecarlo import policy_eval_montecarlo
from algorithms.cvar_policy_evaluation import cvar_policy_evaluation
from algorithms.standard_policy_eval import policy_evaluation_standard
from algorithms.utils import UniformProbabilisticPolicy
from environments.simple_env import SimpleEnv


def main():
    MAX_ITERS = 1000
    TOLL = 1e-3
    Ny = 21
    DISCOUNT = 0.95
    alphas = np.concatenate(([0], np.logspace(-2, 0, Ny - 1)))
    world = SimpleEnv()
    policy = UniformProbabilisticPolicy(world)

    np.random.seed(2)
    print('Standard policy evaluation')
    V_exp = policy_evaluation_standard(world, max_iters=MAX_ITERS, eps_convergence=TOLL, Pol=policy, discount=DISCOUNT)
    pickle.dump(V_exp, open('exp_v.pkl', mode='wb'))

    np.random.seed(2)
    print('CVaR policy evaluation')
    V_cvar = cvar_policy_evaluation(world, max_iters=MAX_ITERS, eps_convergence=TOLL, discount=DISCOUNT, alpha_set=alphas, policy=policy)
    pickle.dump(V_cvar, open('cvar_v.pkl', mode='wb'))

    np.random.seed(2)
    print('CVaR policy evaluation Monte Carlo')
    V_cvar_montecarlo = policy_eval_montecarlo(alphas, policy, DISCOUNT, world, num_samples=10_000)
    pickle.dump(V_cvar_montecarlo, open('cvar_montecarlo.pkl', mode='wb'))

if __name__ == '__main__':
    main()