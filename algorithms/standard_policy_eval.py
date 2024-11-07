import copy

import numpy as np

def value_update(world, V, Pol, i, discount):
    V_ = copy.deepcopy(V)
    for s in world.states():
        q_values = np.full(len(world.ACTIONS), -np.inf)
        available_actions = world.actions(s)
        for a in available_actions:
            Q = 0
            for t in world.transitions(s)[a]:
                Q += t.prob * (t.reward + discount * V_[t.state.id])
            q_values[a] = Q

        policy_probs = Pol.policy[s.id]
        V[s.id] = (policy_probs * np.array(q_values)).sum()

    return V


def policy_evaluation_standard(world, max_iters=1e3, eps_convergence=1e-3, Pol=None, discount=0.95):
    V = np.zeros(world.Ns)
    i = 0
    while True:
        V_prev = copy.deepcopy(V)
        V_new = value_update(world, V, Pol, i, discount)
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

    return V

# def main():
#     PERFORM_VI = True
#     # MAX_ITERS = 40
#     MAX_ITERS = 1000
#     TOLL = 1e-3
#
#     np.random.seed(2)
#     if PERFORM_VI:
#         # world = GridWorld(14, 16, random_action_p=0.05, path='gridworld3.png')
#         world = SimpleEnv()
#         V = policy_evaluation_standard(world, max_iters=MAX_ITERS, eps_convergence=TOLL)
#         pickle.dump(V, open('../standard_vi.pkl', mode='wb'))
#         print('Value function:', V)
#
#
# if __name__ == '__main__':
#     main()
