import scipy
from scipy.stats import alpha

from cvar.gridworld.autonomous_car import AutonomousCarNavigation
from cvar.gridworld.cliffwalker import GridWorld
from cvar.gridworld.core import cvar_computation
from cvar.gridworld.core.constants import gamma
from cvar.gridworld.core.policies import XiBasedPolicy
from cvar.gridworld.core.runs import epoch
from cvar.gridworld.algorithms.value_iteration import value_iteration
import numpy as np


def several_epochs(arg):
    np.random.seed()
    world, policy, nb_epochs = arg
    rewards = np.zeros(nb_epochs)

    for i in range(nb_epochs):
        S, A, R = epoch(world, policy)
        policy.reset()
        rewards[i] = np.sum(R)
        rewards[i] = np.dot(R, np.array([gamma ** i for i in range(len(R))]))

    return rewards


def policy_stats(world, policy, alpha, nb_epochs, verbose=True):
    import copy
    import multiprocessing as mp
    threads = 14

    with mp.Pool(threads) as p:
        rewards = p.map(several_epochs,
                        [(world, copy.deepcopy(policy), int(nb_epochs / threads)) for _ in range(threads)])

    rewards = np.array(rewards).flatten()

    var, cvar = cvar_computation.var_cvar_from_samples(rewards, alpha)
    if verbose:
        print('----------------')
        print(policy.__name__)
        print('expected value=', np.mean(rewards))
        print('cvar_{}={}'.format(alpha, cvar))
        # print('var_{}={}'.format(alpha, var))

    return cvar, rewards


def exhaustive_stats(world, epochs, *args):
    V = value_iteration(world)

    alphas = np.array([1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001])

    cvars = np.zeros((len(args), len(alphas)))
    names = []

    for i, policy in enumerate(args):
        names.append(policy.__name__)
        for j, alpha in enumerate(alphas):
            pol = policy(V, alpha)

            cvars[i, j], _ = policy_stats(world, pol, alpha=alpha, nb_epochs=int(epochs), verbose=False)

            print('{}_{} done...'.format(pol.__name__, alpha))

    import pickle
    pickle.dump({'cvars': cvars, 'alphas': alphas, 'names': names}, open('data/stats.pkl', 'wb'))
    print(cvars)

    from cvar.gridworld.plots.other import plot_cvars
    plot_cvars()


if __name__ == '__main__':
    import pickle
    from cvar.gridworld.plots.grid import InteractivePlotMachine, PlotMachine
    PERFORM_VI = False
    # MAX_ITERS = 40
    MAX_ITERS = 100
    TOLL = 1e-3

    np.random.seed(2)
    if PERFORM_VI:
        world = GridWorld(14, 16, random_action_p=0.05, path='gridworld3.png')
        # # world = AutonomousCarNavigation()
        V = value_iteration(world, max_iters=MAX_ITERS, eps_convergence=TOLL)
        pickle.dump((world, V), open('data/models/vi_test.pkl', mode='wb'))

    # ============================= load
    world, V = pickle.load(open('data/models/vi_test.pkl', 'rb'))
    # indexes = [2, 11, 20]
    alphas = np.concatenate(([0], np.logspace(-2, 0, 20)))
    # alphas = alphas[1:]
    # ============================= RUN
    CvarValues = [np.array([V.V[ix].cvar_alpha(alpha) for ix in np.ndindex(V.V.shape)]).reshape(V.V.shape) for alpha in alphas]
    CvarValues = np.array(CvarValues)
    # CvarValues = np.array([CvarValues[i].flatten(order='F') for i in range(CvarValues.shape[0])])
    matlabValues =  -scipy.io.loadmat('data/models/value.mat')['im']
    # CvarValues[matlabValues == 0] = 0
    matlab_alpha1 = matlabValues[-1, :, :]
    python_alpha1 = CvarValues[-1, :, :]
    # print(np.isclose(matlab_alpha1, python_alpha1, atol=1e-2).all())
    # alphas = [1.0]
    # ============================= PLOT
    for alpha in alphas:
        print(alpha)
        pm = InteractivePlotMachine(world, V, alpha=alpha)
        pm.fig.savefig('data/figures/vi_{}.png'.format(alpha))
        # pm.show()

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
