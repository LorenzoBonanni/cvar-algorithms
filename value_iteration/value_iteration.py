from cliffwalker import *
from util.constants import gamma
from util.util import spaced_atoms
from util import cvar_computation
import numpy as np
import copy
from pulp import *

# atom spacing
NB_ATOMS = 4
LOG = True  # atoms are log-spaced
SPACING = 2

# use LP when computing CVaRs
TAMAR_LP = False

WASSERSTEIN = False
# WASSERSTEIN = True


class ValueFunction:

    def __init__(self, world):
        self.world = world

        self.V = np.empty((world.height, world.width), dtype=object)
        for ix in np.ndindex(self.V.shape):
            self.V[ix] = MarkovState()

    def update(self, y, x):

        vars = []
        cvars = []

        for a in self.world.ACTIONS:
            t = list(self.transitions(y, x, a))

            var_values = self.transition_vars(y, x, a)

            if TAMAR_LP:
                v, yv = self.V[y, x].compute_cvar_by_lp([t_.prob for t_ in t], var_values)
            else:

                if WASSERSTEIN:
                    v, yv = self.V[y, x].compute_wasserstein([t_.prob for t_ in t], var_values)
                else:
                    v, yv = self.V[y, x].compute_cvar_by_sort([t_.prob for t_ in t], var_values)
            vars.append(v)
            cvars.append(yv)

        vars = np.array(vars)
        cvars = np.array(cvars)
        best_args = np.argmax(cvars, axis=0)

        self.V[y, x].var = np.array([vars[best_args[i], i] for i in range(len(self.V[y, x].var))])

    def next_action(self, y, x, alpha):
        assert alpha != 0

        best = (-1e6, 0, 0)
        for a in self.world.ACTIONS:

            if TAMAR_LP:
                cv, xis = self.tamar_lp_single(y, x, a, alpha)
            else:
                _, cv, xis = self.var_cvar_xis(y, x, a, alpha)

            if cv > best[0]:
                best = (cv, xis, a)

        _, xis, a = best

        return a, xis

    def tamar_lp_single(self, y, x, a, alpha):
        """
        Create LP:
        min Sum p_t * I

        0 <= xi <= 1/alpha
        Sum p_t * xi == 1

        I = max{yV}

        return yV[alpha]
        """

        transition_p = [t.prob for t in self.transitions(y, x, a)]
        var_values = self.transition_vars(y, x, a)
        nb_transitions = len(transition_p)

        Xi = [LpVariable('xi_' + str(i)) for i in range(nb_transitions)]
        I = [LpVariable('I_' + str(i)) for i in range(nb_transitions)]

        prob = LpProblem(name='tamar')

        for xi in Xi:
            prob.addConstraint(0 <= xi)
            prob.addConstraint(xi <= 1. / alpha)
        prob.addConstraint(sum([xi * p for xi, p in zip(Xi, transition_p)]) == 1)

        for xi, i, var in zip(Xi, I, var_values):
            last = 0.
            f_params = []
            for ix in range(self.V[y, x].nb_atoms):
                k = var[ix]
                last += k * self.V[y, x].atom_p[ix]
                q = last - k * self.V[y, x].atoms[ix + 1]
                prob.addConstraint(i >= k * xi * alpha + q)
                f_params.append((k, q))

        # opt criterion
        prob.setObjective(sum([i * p for i, p in zip(I, transition_p)]))

        prob.solve()

        return value(prob.objective), [value(xi)*alpha for xi in Xi]

    def var_cvar_xis(self, y, x, a, alpha):
        """
        Compute VaR, CVaR and xi values in O(nlogn)
        :param y:
        :param x:
        :param a:
        :param alpha:
        :return: var, cvar, xis
        """
        # TODO: O(n)

        transitions = list(self.transitions(y, x, a))
        var_values = self.transition_vars(y, x, a)

        info = extract_distribution(transitions, var_values, self.V[y, x].atom_p)

        xis = np.zeros(len(transitions))
        p = 0.
        cv = 0.
        v = 0.

        # TODO: unify with util
        for p_, i, v in info:
            if p + p_ >= alpha:
                xis[i] += alpha-p
                cv += (alpha-p) * v
                break
            else:
                xis[i] += p_
                cv += p_ * v
                p += p_

        return v, cv, xis / np.array([t.prob for t in transitions])

    def y_var(self, y, x, a, var):

        transitions = list(self.transitions(y, x, a))
        var_values = self.transition_vars(y, x, a)

        info = extract_distribution(transitions, var_values, self.V[y, x].atom_p)

        yv = 0.
        p = 0
        for p_, _, v_ in info:
            if v_ >= var:  # TODO: solve for discrete distributions
                break
            else:
                yv += p_ * v_
            p += p_

        return p, yv

    def transitions(self, y, x, a):
        for t in self.world.transitions(State(y, x))[a]:
            yield t

    def transition_vars(self, y, x, a):
        return np.array([t.reward + gamma * self.V[t.state.y, t.state.x].var for t in self.transitions(y, x, a)])


def extract_distribution(transitions, var_values, atom_p):
    """

    :param transitions:
    :param var_values:
    :param atom_p:
    :return: sorted list of tuples (probability, index, var)
    """
    info = np.empty(var_values.shape, dtype=object)
    for i_t, t in enumerate(transitions):
        for i_v, v, p_ in zip(range(len(var_values[i_t])), var_values[i_t], atom_p):
            info[i_t, i_v] = (p_ * t.prob, i_t, v)

    info = list(info.flatten())
    info.sort(key=lambda x: x[-1])
    return info


class MarkovState:

    def __init__(self):
        self.nb_atoms = NB_ATOMS
        self.var = np.zeros(self.nb_atoms)
        self.atoms = spaced_atoms(self.nb_atoms, SPACING, LOG)    # e.g. [0, 0.25, 0.5, 1]
        self.atom_p = self.atoms[1:] - self.atoms[:-1]  # [0.25, 0.25, 0.5]

    def plot(self, show=True):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2)

        # var
        ax[0].step(self.atoms, list(self.var) + [self.var[-1]], 'o-', where='post')

        # yV
        # ax[1].plot(self.atoms, np.insert(np.cumsum(atom_p * sol), 0, 0), 'o-')
        if show:
            plt.show()

    def y_cvar(self, alpha, var=None):

        if var is None:
            var = self.var

        if alpha == 0:
            return min(var)

        p = 0.
        cv = 0.
        for p_, v_ in zip(self.atom_p, var):
            if p + p_ >= alpha:
                cv += (alpha-p) * v_
                break
            else:
                cv += p_ * v_
                p += p_

        return cv

    def y_var(self, var, var_values=None):

        if var_values is None:
            var_values = self.var

        cv = 0.
        for p_, v_ in zip(self.atom_p, var_values):
            if v_ > var:  # TODO: solve for discrete distributions
                break
            else:
                cv += p_ * v_

        return cv

    def expected_value(self):
        return np.dot(self.atom_p, self.var)

    def compute_cvar_by_sort(self, transition_p, var_values):
        return cvar_computation.v_yc_from_transitions_sort(self.atoms, transition_p, var_values)

    def compute_wasserstein(self, transition_p, var_values):
        raise NotImplementedError("Waiting for transfer from lp_compare and fix.")


    def compute_cvar_by_lp(self, transition_p, var_values):
        cvar_computation.v_yc_from_transitions_lp(self.atoms, transition_p, var_values)


def value_update(world, V):

    V_ = copy.deepcopy(V)
    for s in world.states():
        V_.update(s.y, s.x)

    return V_


def converged(V, V_, world):
    eps = 1e-4
    max_val = eps
    for s in world.states():
        # dist = np.max(np.abs(V.V[s.y, s.x].var-V_.V[s.y, s.x].var))
        cvars = np.array([V.V[s.y, s.x].y_cvar(alpha)*alpha for alpha in V.V[s.y, s.x].atoms])
        cvars_ = np.array([V_.V[s.y, s.x].y_cvar(alpha)*alpha for alpha in V_.V[s.y, s.x].atoms])
        dist = np.max(np.abs(cvars - cvars_))
        max_val = max(max_val, dist)
    if max_val > eps:
        print(max_val)
        # print(s)
        # print(V.V[s.y, s.x].var)
        # print(V_.V[s.y, s.x].var)
        return False
    return True


def value_iteration(world, max_iters=1e3):
    V = ValueFunction(world)
    i = 0
    while True:
        V_ = value_update(world, V)
        # if i % 10 == 0:
        #     V_.V[0, 5].plot()
        if (converged(V, V_, world) and i != 0) or i > max_iters:
            print("value fully learned after %d iterations" % (i,))
            break
        V = V_
        i += 1

        print('Value iteration:', i)

    return V


# TODO: control error by adding atoms
# TODO: smart convergence  ---  the cvar converges, not necessarily the distributions (?)
# TODO: V -> Q

if __name__ == '__main__':

    # world = GridWorld(1, 3, random_action_p=0.3)
    world = GridWorld(4, 6, random_action_p=0.1)

    print('ATOMS:', spaced_atoms(NB_ATOMS, SPACING, LOG))

    # =============== PI setup
    alpha = 0.1
    V = value_iteration(world, max_iters=100)
    # V.V[3,0].plot()
    # print(V.V[1,5].var)
    # print(V.V[3,0].y_cvar(1.0))
    # print(V.V[3,0].expected_value())




