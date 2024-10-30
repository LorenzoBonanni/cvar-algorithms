import copy

import numpy as np
from docplex.mp.model import Model
from tqdm import tqdm

from cvar.gridworld.cliffwalker import GridWorld


def get_transition_information(action_transitions):
    """
    Extracts positions and probabilities from action transitions.

    Parameters:
    action_transitions (list): A list of transition objects, where each object has 'state' and 'prob' attributes.

    Returns:
    tuple: A tuple containing two numpy arrays:
        - transitions_pos (np.array): Array of transition states.
        - transitions_probabilities (np.array): Array of transition probabilities.
    """
    transitions_pos = []
    transitions_probabilities = []
    transitions_rewards = []
    for trans in action_transitions:
        transitions_pos.append(trans.state)
        transitions_probabilities.append(trans.prob)
        transitions_rewards.append(trans.reward)

    return np.array(transitions_pos), np.array(transitions_probabilities), np.array(transitions_rewards)


def solve_problem(solver):
    """
    Solves the optimization problem using the provided solver.

    Parameters:
    solver (Model): An instance of the CPLEX Model class used to define and solve the optimization problem.

    Returns:
    tuple: A tuple containing two numpy arrays:
        - solved_xi (np.array): Array of solution values for variables starting with 'xi'.
        - solved_t (np.array): Array of solution values for variables starting with 't'.

    Raises:
    SystemExit: If no optimal solution is found.
    """
    # solver.print_information()
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
        return np.array(solved_xi), np.array(solved_t)
    else:
        print('No optimal solution found')
        exit(1)


def create_decision_variables(solver, prefix, n_vars, bounds=None, start_index=0):
    """
    Create an array of decision variables with consistent naming and bounds.

    Parameters:
        solver (Model): An instance of the CPLEX Model class
        prefix: String prefix for variable names (e.g., 'xi', 't')
        n_vars: Number of variables to create
        bounds: Tuple of (lower_bound, upper_bound) or None for default bounds
        start_index: Starting index for variable naming

    Returns:
        array of decision variables, next available index
    """
    if bounds is None:
        bounds = (0, solver.infinity)  # Default bounds

    variables = np.array([
        solver.continuous_var(
            lb=bounds[0],
            ub=bounds[1],
            name=f'{prefix}_{i + start_index}'
        ) for i in range(n_vars)
    ])

    return variables, start_index + n_vars


def value_update(world, V, id=0, alpha_set_all=None, discount=0.95):
    """
    Updates the value function for the given world.

    Parameters:
    world (GridWorld): The grid world environment.
    V (np.array): The current value function.
    id (int, optional): The iteration id for progress display. Defaults to 0.
    alpha_set_all (np.array, optional): Array of alpha values for each state. Defaults to None.
    discount (float, optional): The discount factor for future rewards. Defaults to 0.95.

    Returns:
    np.array: The updated value function.
    """
    Pol = np.zeros_like(V, dtype=int)
    V_ = copy.deepcopy(V)

    states = list(world.states())
    for s in tqdm(states, desc='Value Update %d' % id):
        alpha_set = alpha_set_all[s.y, s.x]
        transitions = world.transitions(s)
        ts = np.array([])
        solver = Model(name='cvar_value')
        objective = np.zeros((len(world.ACTIONS), len(alpha_set)))

        counter = 0
        for alpha_idx, alpha in enumerate(alpha_set):
            for a in world.ACTIONS:
                transitions_pos, transitions_probabilities, _ = get_transition_information(transitions[a])
                n_trans = len(transitions_pos)

                if alpha == 0:
                    # when alpha is 0, the cvar is simply the worst case value, so no expectation over some distribution
                    transitions_pos = np.array(transitions_pos)
                    objective[a, alpha_idx] = min(V_[alpha_idx, transitions_pos[:, 0], transitions_pos[:, 1]])
                    continue

                # Create xi variables (non-negative)
                xi, counter = create_decision_variables(
                    solver=solver,
                    prefix='xi',
                    n_vars=n_trans,
                    bounds=(0, solver.infinity),
                    start_index=counter
                )
                solver.add_constraint(xi @ transitions_probabilities == 1)

                # Create t variables (unrestricted)
                t, counter = create_decision_variables(
                    solver=solver,
                    prefix='t',
                    n_vars=n_trans,
                    bounds=(-1e6, 1e6),
                    start_index=counter
                )
                ts = np.append(ts, t)
                for i in range(len(alpha_set) - 1):
                    alpha_i = alpha_set[i]
                    alpha_i_next = alpha_set[i + 1]
                    v_i = V_[i, transitions_pos[:, 0], transitions_pos[:, 1]]
                    v_i_next = V_[i + 1, transitions_pos[:, 0], transitions_pos[:, 1]]
                    slope = (alpha_i_next * v_i_next - alpha_i * v_i) / (alpha_i_next - alpha_i)

                    right_ineq = alpha_i * v_i / alpha - slope * alpha_i / alpha * transitions_probabilities
                    left_ineq = t - slope * xi * transitions_probabilities
                    for idx in range(len(right_ineq)):
                        solver.add_constraint(left_ineq[idx] >= right_ineq[idx])
                        solver.add_constraint(xi[idx] <= 1 / alpha)


        solver.minimize(sum(ts))
        _, t_values = solve_problem(solver)
        # xi_values = xi_values.reshape((len(world.ACTIONS), len(alpha_set)-1, -1))
        t_values = t_values.reshape((len(world.ACTIONS), len(alpha_set)-1, -1))
        t_values = t_values.sum(axis=-1)
        objective[:, 1:] = t_values

        rewards = np.array([get_transition_information(transitions[a])[2] for a in world.ACTIONS])

        Q = np.min(rewards[:, np.newaxis, :] + discount * objective[:, :, np.newaxis], axis=-1)
        for alpha_idx2 in range(1, len(alpha_set)):
            Pol[alpha_idx2, s.y, s.x] = np.argmax(Q[:, alpha_idx2])
            V_[alpha_idx2, s.y, s.x] = Q[Pol[alpha_idx2, s.y, s.x], alpha_idx2]

    return V_


def value_iteration(world, V=None, max_iters=1e3, eps_convergence=1e-3):
    Ny = 21
    if V is None:
        V = np.zeros((Ny, world.height, world.width))
    Y_set_all = np.ones((world.height, world.width, 1)) * np.concatenate(([0], np.logspace(-2, 0, Ny-1)))
    i = 0
    figax = None
    while True:
        V_ = value_update(world, V, i, Y_set_all, discount=0.95)
        error = np.max(np.abs(V - V_))
        print('Iteration:{}, error={}'.format(i, error))
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
