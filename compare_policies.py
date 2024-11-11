import os
import pickle
import random

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from algorithms.utils import FixedPolicy
from environments.autonomous_car import AutonomousCarNavigation
import matplotlib.pyplot as plt

NUM_TRAJECTORIES = 500_000
GAMMA = 0.95
EXP_IDX = 0
DATA = {'cvar_exp_policy': [], 'cvar_cvar_policy': [], 'exp_exp_policy': [], 'exp_cvar_policy': [], 'std_exp_policy': [], 'std_cvar_policy': []}
BUFFER = None

def get_return(env, policy, gamma):
    ret = 0
    i = 0
    state = env.initial_state
    while not env.is_terminal(state):
        action = policy.get_action(state)
        t = env.sample_transition(state, action)
        ret += gamma ** i * t.reward
        i += 1
        state = t.state
    return ret

def plot_distributions(r_cvar, r_exp, alpha, cvar_cvar_policy, cvar_exp_policy, exp_exp_policy, exp_cvar_policy):
    """
    Plots the distributions of returns for CVaR and Expected policies.

    Args:
        r_cvar (np.ndarray): Array of returns for the CVaR policy.
        r_exp (np.ndarray): Array of returns for the Expected policy.
        alpha (float): The alpha value used for CVaR calculation.
        cvar_cvar_policy (float): CVaR return for the CVaR policy.
        cvar_exp_policy (float): CVaR return for the Expected policy.
        exp_exp_policy (float): Mean return for the Expected policy.
        exp_cvar_policy (float): Mean return for the CVaR policy.
    """
    plt.figure(figsize=(12, 6))

    plt.hist(r_cvar, color='b', bins=100, alpha=0.5, label='CVaR', )
    plt.axvline(exp_cvar_policy, color='b', linestyle='solid', linewidth=2, label=r"Mean Return $\pi_{cvar}$:" + f"{round(exp_cvar_policy, 2)}")
    plt.axvline(cvar_cvar_policy, color='b', linestyle='dashed', linewidth=2, label=r"Cvar Return $\pi_{cvar}$:" + f"{round(cvar_cvar_policy, 2)}")

    plt.hist(r_exp, color='g', bins=100, alpha=0.5, label='Expected')
    plt.axvline(exp_exp_policy, color='g', linestyle='solid', linewidth=2, label=r"Mean Return $\pi_{exp}$:" + f"{round(exp_exp_policy, 2)}")
    plt.axvline(cvar_exp_policy, color='g', linestyle='dashed', linewidth=2, label=r"Cvar Return $\pi_{exp}$:" + f"{round(cvar_exp_policy, 2)}")

    plt.legend()
    plt.title(f'alpha={alpha}')
    plt.savefig(f'plots/policy_comparison/alpha={alpha}_returns_distribution.png')


def seed_everything(param):
    random.seed(param)
    np.random.seed(param)
    os.environ["PYTHONHASHSEED"] = str(param)


def run_experiment(env, alpha, CvarPolicy, StandardPolicy):
    seed_everything(42)
    r_cvar = np.array(Parallel(n_jobs=-1, verbose=False)(delayed(get_return)(env, CvarPolicy, GAMMA) for _ in range(NUM_TRAJECTORIES)))
    r_cvar.sort()
    cvar_cvar_policy = float(np.mean(r_cvar[r_cvar <= np.quantile(r_cvar, alpha)]))
    exp_cvar_policy = float(np.mean(r_cvar))

    seed_everything(42)
    global BUFFER, EXP_IDX
    if EXP_IDX == 0:
        r_exp = np.array(Parallel(n_jobs=-1, verbose=False)(delayed(get_return)(env, StandardPolicy, GAMMA) for _ in range(NUM_TRAJECTORIES)))
        r_exp.sort()
        BUFFER = r_exp
        EXP_IDX+=1
    else:
        r_exp = BUFFER

    cvar_exp_policy = float(np.mean(r_exp[r_exp <= np.quantile(r_exp, alpha)]))
    exp_exp_policy = float(np.mean(r_exp))

    DATA['cvar_exp_policy'].append(cvar_exp_policy)
    DATA['cvar_cvar_policy'].append(cvar_cvar_policy)
    DATA['exp_exp_policy'].append(exp_exp_policy)
    DATA['exp_cvar_policy'].append(exp_cvar_policy)
    DATA['std_exp_policy'].append(np.std(r_exp))
    DATA['std_cvar_policy'].append(np.std(r_cvar))

    plot_distributions(r_cvar, r_exp, alpha, cvar_cvar_policy, cvar_exp_policy, exp_exp_policy, exp_cvar_policy)

def main():
    _, CvarPolicy = pickle.load(open('policies/cvar_vi.pkl', 'rb'))
    _, StandardPolicy = pickle.load(open('policies/standard_vi.pkl', 'rb'))
    Ny = 21
    alphas = np.concatenate(([0], np.logspace(-2, 0, Ny - 1)))
    indexes = [1, 11, 17, len(alphas) - 1]
    alphas_to_eval = alphas[indexes]
    env = AutonomousCarNavigation()
    DATA['alphas'] = alphas_to_eval
    for idx, alpha in zip(indexes, alphas_to_eval):
        run_experiment(env, alpha, FixedPolicy(env, CvarPolicy[idx]), FixedPolicy(env, StandardPolicy))

    data_df = pd.DataFrame(DATA)
    data_df.set_index('alphas', inplace=True)
    data_df.to_csv('comparison.csv')

main()