import copy

import numpy as np
from pulp import LpProblem, LpVariable, LpMinimize, LpStatusOptimal, LpStatus, CPLEX_PY
from tqdm import tqdm

# IMPLEMENTATION OF VALUE ITERATION WHERE THE ENVIRONMENT HAS A REWARD IN THE FORM R(s,a,s')

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
    transitions_ids = []
    transitions_probabilities = []
    transitions_rewards = []
    for trans in action_transitions:
        transitions_ids.append(trans.state.id)
        transitions_probabilities.append(trans.prob)
        transitions_rewards.append(trans.reward)

    return np.array(transitions_ids), np.array(transitions_probabilities), np.array(transitions_rewards)


def solve_problem(solver):
    """
    Solves the optimization problem using the provided solver.

    Parameters:
    solver (LpProblem): An instance of the PuLP LpProblem class used to define and solve the optimization problem.

    Returns:
    tuple: A tuple containing two numpy arrays:
        - solved_xi (np.array): Array of solution values for variables starting with 'xi'.
        - solved_t (np.array): Array of solution values for variables starting with 't'.

    Raises:
    SystemExit: If no optimal solution is found.
    """
    solver.solve(CPLEX_PY(msg=False))
    if solver.status == LpStatusOptimal:
        solved_xi = []
        solved_t = []
        solved_xi_names = []
        solved_t_names = []
        for v in solver.variables():
            if v.name.startswith('xi'):
                solved_xi.append(v.varValue)
                solved_xi_names.append(v.name)
            elif v.name.startswith('t'):
                solved_t.append(v.varValue)
                solved_t_names.append(v.name)

        # Sort t values by their original order
        solved_t = [t for _, t in sorted(zip(solved_t_names, solved_t), key=lambda x: int(x[0].split('_')[1]))]
        solved_xi = [t for _, t in sorted(zip(solved_xi_names, solved_xi), key=lambda x: int(x[0].split('_')[1]))]
        return np.array(solved_xi), np.array(solved_t)
    else:
        print('No optimal solution found')
        print("STATUS:", LpStatus[solver.status])
        exit(1)


def create_decision_variables(prefix, n_vars, bounds=None, start_index=0):
    """
    Create an array of decision variables with consistent naming and bounds.

    Parameters:
        prefix: String prefix for variable names (e.g., 'xi', 't')
        n_vars: Number of variables to create
        bounds: Tuple of (lower_bound, upper_bound) or None for default bounds
        start_index: Starting index for variable naming

    Returns:
        array of decision variables, next available index
    """

    variables = np.array([
        LpVariable(f'{prefix}_{i + start_index}', lowBound=bounds[0], upBound=bounds[1])
        for i in range(n_vars)
    ])

    return variables, start_index + n_vars


def get_deterministic_reward(transitions):
    rewards = []
    for action_transitions in transitions:
        _, transitions_probabilities, transitions_rewards = get_transition_information(action_transitions)
        idx_max_prob = np.where(transitions_probabilities == np.max(transitions_probabilities))[0][0]
        rewards.append(transitions_rewards[idx_max_prob])

    return np.array(rewards)

def dynamic_reshape(arr, n_trans_list, num_alpha):
    # Calculate cumulative sizes for splitting
    split_indices = np.cumsum([0] + [(num_alpha-1) * n for n in n_trans_list])

    # Split the array based on n_trans_list
    split_arrays = np.split(arr, split_indices[1:-1])

    # Reshape each split array
    reshaped_arrays = [arr_split.reshape(num_alpha-1, n_trans)
                       for arr_split, n_trans in zip(split_arrays, n_trans_list)]
    return reshaped_arrays

def cvar_value_update(world, V, Pol, id=0, alpha_set_all=None, discount=0.95):
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
    V_ = copy.deepcopy(V)
    # np.save('vi_{}.npy'.format(id), V_)

    states = list(world.states())
    # TODO this loop is parallelizable
    for s in tqdm(states, desc='Value Update %d' % id):
        alpha_set = alpha_set_all[s.id]
        transitions = world.transitions(s)
        ts = np.array([])
        solver = LpProblem(name='cvar_value', sense=LpMinimize)
        objective = np.zeros((len(world.ACTIONS), len(alpha_set)))

        counter = 0
        n_trans_list = []
        available_actions = world.actions(s)
        for a in available_actions:
            transitions_ids, transitions_probabilities, transitions_rewards = get_transition_information(transitions[a])
            n_trans = len(transitions_ids)
            n_trans_list.append(n_trans)
            for alpha_idx, alpha in enumerate(alpha_set):
                if alpha == 0:
                    # when alpha is 0, the cvar is simply the worst case value, so no expectation over some distribution
                    objective[a, alpha_idx] = min((transitions_rewards + discount * V_[alpha_idx, transitions_ids]) * transitions_probabilities)
                    continue

                # Create xi variables (non-negative)
                xi, counter = create_decision_variables(
                    prefix='xi',
                    n_vars=n_trans,
                    bounds=(0, None),
                    start_index=counter
                )
                solver += xi @ transitions_probabilities == 1

                t, counter = create_decision_variables(
                    prefix='t',
                    n_vars=n_trans,
                    bounds=(-1e6, 1e6),
                    start_index=counter
                )
                ts = np.append(ts, xi * transitions_probabilities * transitions_rewards + t)
                for i in range(len(alpha_set) - 1):
                    alpha_i = alpha_set[i]
                    alpha_i_next = alpha_set[i + 1]
                    v_i = V_[i, transitions_ids]
                    v_i_next = V_[i + 1, transitions_ids]
                    slope = (alpha_i_next * v_i_next - alpha_i * v_i) / (alpha_i_next - alpha_i)

                    right_ineq = (alpha_i * v_i / alpha - slope * alpha_i / alpha) * transitions_probabilities
                    left_ineq = t - slope * xi * transitions_probabilities
                    for idx in range(len(right_ineq)):
                        solver += left_ineq[idx] >= right_ineq[idx]
                        solver += xi[idx] <= 1 / alpha

        solver += sum(ts)
        xi_values, t_values = solve_problem(solver)
        xi_values = dynamic_reshape(xi_values, n_trans_list, len(alpha_set))
        t_values = dynamic_reshape(t_values, n_trans_list, len(alpha_set))
        for idx, a in enumerate(available_actions):
            _, transitions_probabilities, transitions_rewards = get_transition_information(transitions[a])
            t_values[a] = (xi_values[a] * transitions_rewards * transitions_probabilities + discount * t_values[a]).sum(-1)

        objective[available_actions, 1:] = np.array(t_values)
        unavailable_actions = set(range(len(world.ACTIONS))) - set(available_actions)
        objective[list(unavailable_actions), :] = -np.inf
        Q = objective

        policy_probs = np.expand_dims(Pol.policy[s.id], axis=1)
        V[:, s.id] = (policy_probs * Q).sum(axis=0)

    return V


def cvar_policy_evaluation(world, max_iters=1e3, eps_convergence=1e-3, alpha_set=None, discount=0.95, policy=None):
    V = np.zeros((len(alpha_set), world.Ns))
    Y_set_all = np.ones((world.Ns, 1)) * alpha_set
    i = 0
    Pol = policy
    while True:
        V_prev = copy.deepcopy(V)
        V_new = cvar_value_update(world, V, Pol, i, Y_set_all, discount=discount)
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
#     MAX_ITERS = 1000
#     TOLL = 1e-3
#
#     np.random.seed(2)
#     if PERFORM_VI:
#         # world = GridWorld(14, 16, random_action_p=0.05, path='gridworld3.png')
#         world = SimpleEnv()
#         V = cvar_policy_evaluation(world, max_iters=MAX_ITERS, eps_convergence=TOLL)
#         # pickle.dump((V.reshape(21, world.height, world.width), Policy.reshape(21, world.height, world.width)), open('vi_test.pkl', mode='wb'))
#         print(V[:, 0])
#         pickle.dump(V, open('../cvar_pe.pkl', mode='wb'))
#
#
# if __name__ == '__main__':
#     main()