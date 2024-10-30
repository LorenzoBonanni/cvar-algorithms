import copy

import numpy as np
from docplex.mp.model import Model
from tqdm import tqdm

from cvar.gridworld.cliffwalker import GridWorld


def get_position_and_probabilities(action_transitions):
    transitions_pos = []
    transitions_probabilities = []
    for trans in action_transitions:
        transitions_pos.append(trans.state)
        transitions_probabilities.append(trans.prob)

    return np.array(transitions_pos), np.array(transitions_probabilities)


def solve_problem(solver):
    solver.print_information()
    solution = solver.solve()
    if solution.solve_status.value == 2:
        print('Optimal solution found')
        solved_xi = []
        solved_t = []
        for v in solver.iter_variables():
            if v.name.startswith('xi'):
                solved_xi.append(v.solution_value)
            elif v.name.startswith('t'):
                solved_t.append(v.solution_value)
    else:
        print('No optimal solution found')
        exit(1)


def value_update(world, V, id=0, alpha_set_all=None):
    V_ = copy.deepcopy(V)
    for s in tqdm(world.states(), desc='Value Update %d' % id):
        alpha_set = alpha_set_all[s.y, s.x]
        transitions = world.transitions(s)
        for alpha_idx, alpha in enumerate(alpha_set):
            solver = Model(name='cvar_value')

            counter = 0
            objective = np.zeros((len(world.ACTIONS), len(alpha_set)))

            for a in world.ACTIONS:
                transitions_pos, transitions_probabilities = get_position_and_probabilities(transitions[a])
                n_trans = len(transitions_pos)

                if alpha == 0:
                    # when alpha is 0, the cvar is simply the worst case value, so no expectation over some distribution
                    transitions_pos = np.array(transitions_pos)
                    objective[a, alpha_idx] = min(V_[alpha_idx, transitions_pos[:, 0], transitions_pos[:, 1]])
                    continue

                for i in range(len(alpha_set) - 1):
                    alpha_i = alpha_set[i]
                    alpha_i_next = alpha_set[i + 1]
                    v_i = V_[i, transitions_pos[:, 0], transitions_pos[:, 1]]
                    v_i_next = V_[i + 1, transitions_pos[:, 0], transitions_pos[:, 1]]
                    slope = (alpha_i_next * v_i_next - alpha_i * v_i) / (alpha_i_next - alpha_i)


                    xi = np.array([solver.continuous_var(0, solver.infinity, f'xi_{counter + tnum}') for tnum in range(n_trans)])
                    counter += n_trans
                    t = np.array([solver.continuous_var(-1e6, 1e6, f't_{counter + tnum}') for tnum in range(n_trans)])
                    counter += n_trans

                    right_ineq = alpha_i * v_i / alpha - slope * alpha_i / alpha * transitions_probabilities
                    left_ineq = t - slope * xi * transitions_probabilities
                    for idx in range(len(right_ineq)):
                        solver.add_constraint(left_ineq[idx] >= right_ineq[idx])
                        solver.add_constraint(xi[idx] <= 1 / alpha)

                    solver.add_constraint(xi @ transitions_probabilities == 1)
                    solver.maximize(sum(t))
                    solve_problem(solver)

    return V_


def value_iteration(world, V=None, max_iters=1e3, eps_convergence=1e-3):
    Ny = 21
    if V is None:
        V = np.zeros((Ny, world.height, world.width))
    Y_set_all = np.ones((world.height, world.width, 1)) * np.concatenate(([0], np.logspace(-2, 0, Ny-1)))
    i = 0
    figax = None
    while True:
        V_ = value_update(world, V, i, Y_set_all)

        # error, worst_state = value_difference(V, V_, world)
        # if error < eps_convergence:
        #     print("value fully learned after %d iterations" % (i,))
        #     print('Error:', error)
        #     break
        # elif i > max_iters:
        #     print("value finished without convergence after %d iterations" % (i,))
        #     break
        # V = V_
        # i += 1
        #
        # print('Iteration:{}, error={} ({})'.format(i, error, worst_state))

    return V


if __name__ == '__main__':
    import pickle
    from cvar.gridworld.plots.grid import InteractivePlotMachine, PlotMachine

    PERFORM_VI = True
    # MAX_ITERS = 40
    MAX_ITERS = 100
    TOLL = 1e-3

    np.random.seed(2)
    if PERFORM_VI:
        world = GridWorld(14, 16, random_action_p=0.05, path='gridworld3.png')
        # # world = AutonomousCarNavigation()
        V = value_iteration(world, max_iters=MAX_ITERS, eps_convergence=TOLL)
        # pickle.dump((world, V), open('data/models/vi_test.pkl', mode='wb'))

    # # ============================= load
    # world, V = pickle.load(open('data/models/vi_test.pkl', 'rb'))
    # # indexes = [2, 11, 20]
    # alphas = np.concatenate(([0], np.logspace(-2, 0, 20)))
    # # alphas = alphas[1:]
    # # ============================= RUN
    # CvarValues = [np.array([V.V[ix].cvar_alpha(alpha) for ix in np.ndindex(V.V.shape)]).reshape(V.V.shape) for alpha in alphas]
    # CvarValues = np.array(CvarValues)
    # # CvarValues = np.array([CvarValues[i].flatten(order='F') for i in range(CvarValues.shape[0])])
    # matlabValues =  -scipy.io.loadmat('data/models/value.mat')['im']
    # # CvarValues[matlabValues == 0] = 0
    # matlab_alpha1 = matlabValues[-1, :, :]
    # python_alpha1 = CvarValues[-1, :, :]
    # # print(np.isclose(matlab_alpha1, python_alpha1, atol=1e-2).all())
    # alphas = [1.0]
    # # ============================= PLOT
    # for alpha in alphas:
    #     print(alpha)
    #     pm = InteractivePlotMachine(world, V, alpha=alpha)
    #     pm.fig.savefig('data/figures/vi_{}.png'.format(alpha))
    #     # pm.show()

    # =============== VI stats
    # nb_epochs = int(100)
    # rewards_sample = []
    # for alpha in [0.1, 0.25, 0.5, 1.]:
    #     _, rewards = policy_stats(world, XiBasedPolicy(V, alpha), alpha, nb_epochs=nb_epochs)
    #     rewards_sample.append(rewards)
    # np.save('files/sample_rewards_tamar.npy', np.array(rewards_sample))
    # policy_stats(world, var_policy, alpha, nb_epochs=nb_epochs)

    # =============== plot dynamic
    # alpha = 0.5
    # V_visual = np.array([[V.V[i, j].cvar_alpha(alpha) for j in range(len(V.V[i]))] for i in range(len(V.V))])
    # # print(V_visual)
    # plot_machine = PlotMachine(world, V_visual)
    # # policy = var_policy
    # policy = XiBasedPolicy(V, alpha)
    # for i in range(10):
    #     S, A, R = epoch(world, policy, plot_machine=plot_machine)
    #     print('{}: {}'.format(i, np.sum(R)))
    #     policy.reset()
