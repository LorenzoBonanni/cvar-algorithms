import copy
import pickle

import numpy as np
from matplotlib.style.core import available

from environments.autonomous_car import AutonomousCarNavigation
from environments.cliffwalker import GridWorld
from environments.simple_env import SimpleEnv


def value_update(world, V, Pol, i, discount):
    V_ = copy.deepcopy(V)
    for s in world.states():
        q_values = []
        available_actions = world.actions(s)
        for a in available_actions:
            Q = 0
            for t in world.transitions(s)[a]:
                Q += t.prob * (t.reward + discount * V_[t.state.id])
            q_values.append(Q)

        max_idx = np.argmax(q_values)
        Pol[s.id] = available_actions[max_idx]
        V[s.id] = q_values[max_idx]

    return V, Pol




def value_iteration(world, max_iters=1e3, eps_convergence=1e-3):
    V = np.zeros(world.Ns)
    Pol = np.zeros_like(V, dtype=int)
    DISCOUNT = 0.95

    i = 0
    while True:
        V_prev = copy.deepcopy(V)
        V_new, Pol = value_update(world, V, Pol, i, DISCOUNT)
        error = np.max(np.abs(V_new - V_prev))
        print('Iteration:{}, error={}'.format(i, error))
        V = V_new
        if error < eps_convergence:
            print("value fully learned after %d iterations" % (i,))
            print('Error:', error)
            break
        elif i > max_iters:
            print("value finished without convergence after %d iterations" % (i,))
            break
        i += 1

    return V, Pol


def main():
    PERFORM_VI = True
    # MAX_ITERS = 40
    MAX_ITERS = 1000
    TOLL = 1e-3

    np.random.seed(2)
    if PERFORM_VI:
        world = AutonomousCarNavigation()
        # world = GridWorld(14, 16, random_action_p=0.05, path='gridworld3.png')
        V, Policy = value_iteration(world, max_iters=MAX_ITERS, eps_convergence=TOLL)
        # Policy_ = Policy.reshape(world.height, world.width).astype(str)
        # for i in range(world.height):
        #     for j in range(world.width):
        #         Policy_[i, j] = world.ACTION_NAMES[int(Policy_[i, j])]
        # print(Policy_)
        print(V)
        pickle.dump((V, Policy), open('standard_vi.pkl', mode='wb'))
        world.generate_plots(Policy, V, 'standard_vi')
if __name__ == '__main__':
    main()