from cliffwalker import *
import matplotlib.pyplot as plt
import numpy as np
from visual import show_fixed, PlotMachine
from util import q_to_v_argmax
import time

np.random.seed(1337)
np.set_printoptions(3)


MIN_VALUE = FALL_REWARD-50
MAX_VALUE = 0


def clip(ix):
    new_ix = max(0, min(MAX_VALUE - MIN_VALUE, ix))
    return new_ix


class RandomVariable:

    def __init__(self, p=None, z=None):
        self.z = np.arange(MIN_VALUE, MAX_VALUE+1) if z is None else np.copy(z)
        if p is None:
            self.p = np.zeros_like(self.z)
            zero_ix, = np.where(self.z == 0)
            self.p[zero_ix] = 1.0
        else:
            self.p = np.copy(p)
        assert np.abs(np.sum(self.p) - 1.0) < 0.001

    def expected_value(self):
        return np.dot(self.z, self.p)

    def cvar(self, alpha):
        if alpha > 0.999:
            return self.expected_value()
        p = 0.
        i = 0
        while p < alpha:
            p += self.p[i]
            i += 1
        i -= 1
        p = p - self.p[i]
        p_rest = alpha - p
        cvar = (np.dot(self.p[:i], self.z[:i]) + p_rest * self.z[i]) / alpha

        return cvar

    def var(self, alpha):
        if alpha > 0.999:
            return self.z[-1]
        p = 0.
        i = 0
        while p < alpha:
            p += self.p[i]
            i += 1

        if p == alpha:
            var = self.z[i-1]
        else:
            var = self.z[i]

        return var

    def var_index(self, alpha):
        if alpha > 0.999:
            return len(self.z)-1
        p = 0.
        i = 0
        while p < alpha:
            p += self.p[i]
            i += 1

        return i-1

    def exp_(self, s, alpha=None):
        if alpha is None:
            return np.dot(self.p, np.clip(self.z-s, None, 0))
        return 1./alpha*np.dot(self.p, np.clip(self.z-s, None, 0)) + s

    def __add__(self, r):
        # uses the fact that rewards are all integers
        # correct version: z += r
        assert abs(r) < MAX_VALUE - MIN_VALUE  # increase the range
        if r == 0:
            p = self.p
        elif r > 0:
            p = np.roll(self.p, r)
            p[-1] += np.sum(p[:r])
            p[:r] = 0
        else:
            p = np.roll(self.p, r)
            p[0] += np.sum(p[r:])
            p[r:] = 0

        return RandomVariable(p=p)

    def __mul__(self, gamma):
        assert gamma == 1  # other values not supported
        return RandomVariable(z=gamma*self.z, p=self.p)

    def __str__(self):
        return 'p:{}\nz:{}'.format(self.p, self.z)

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.bar(self.z, self.p, width=0.9, )

        ax.set_ylim([0., 1.1 * np.max(self.p)])
        ax.grid()

        fig.show()

    def plot_cdf(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.step(self.z, np.cumsum(self.p), where='post')
        ax.grid()
        plt.show()


def cvar_from_samples(samples, alpha):
    samples = np.sort(samples)
    alpha_ix = int(np.round(alpha * len(samples)))
    var = samples[alpha_ix - 1]
    cvar = np.mean(samples[:alpha_ix])
    return var, cvar


class Policy:
    """ Abstract class representing different policies. """

    __name__ = 'Policy'

    def next_action(self, t):
        raise NotImplementedError()

    def reset(self):
        pass


class FixedPolicy(Policy):
    __name__ = 'Fixed'

    def __init__(self, P, alpha=None):
        self.P = P

    def next_action(self, t):
        return self.P[t.state.y, t.state.x]


class GreedyPolicy(Policy):
    __name__ = 'Greedy'

    def __init__(self, Q, alpha=None):
        self.Q = Q

    def next_action(self, t):
        s = t.state
        return np.argmax(expected_value(self.Q[:, s.y, s.x]))


class NaiveCvarPolicy(Policy):
    __name__ = 'Naive CVaR'

    def __init__(self, Q, alpha):
        self.Q = Q
        self.alpha = alpha

    def next_action(self, t):
        s = t.state
        action_distributions = self.Q[:, s.y, s.x]
        a = np.argmax([d.cvar(self.alpha) for d in action_distributions])
        return a


class AlphaBasedPolicy(Policy):
    __name__ = 'alpha-based CVaR'

    def __init__(self, Q, alpha):
        self.Q = Q
        self.init_alpha = alpha
        self.alpha = alpha
        self.s_old = None
        self.a_old = None

    def next_action(self, t):
        s = t.state

        action_distributions = self.Q[:, s.y, s.x]
        old_action = np.argmax(expected_value(action_distributions))

        if self.alpha > 0.999:
            return old_action

        if self.s_old is not None:
            self.update_alpha(self.s_old, self.a_old, t)
        a = np.argmax([d.cvar(self.alpha) for d in action_distributions])
        self.s_old = s
        self.a_old = a

        return a

    def update_alpha(self, s, a, t):
        """
        Correctly updates next alpha with discrete variables.
        :param s: state we came from
        :param a: action we took
        :param t: transition we sampled
        """
        s_ = t.state
        s_dist = self.Q[a, s.y, s.x]

        a_ = np.argmax(expected_value(self.Q[:, s_.y, s_.x]))

        s__dist = self.Q[a_, s_.y, s_.x]

        var_ix = s_dist.var_index(self.alpha)

        var__ix = clip(var_ix - t.reward)

        # prob before var
        p_pre = np.sum(s_dist.p[:var_ix])
        # prob at var
        p_var = s_dist.p[var_ix]
        # prob before next var
        p__pre = np.sum(s__dist.p[:var__ix])
        # prob at next var
        p__var = s__dist.p[var__ix]

        # how much does t add to the full var
        p_portion = (t.prob * p__var) / self.p_portion_sum(s, a, var_ix)

        # we care about this portion of var
        p_active = (self.alpha - p_pre) / p_var

        self.alpha = p__pre + p_active * p__var * p_portion

    def p_portion_sum(self, s, a, var_ix):

        p_portion = 0.

        for t_ in transitions(s)[a]:
            action_distributions = self.Q[:, t_.state.y, t_.state.x]
            a_ = np.argmax(expected_value(action_distributions))
            p_portion += t_.prob*action_distributions[a_].p[clip(var_ix - t_.reward)]

        return p_portion

    def reset(self):
        self.alpha = self.init_alpha
        self.s_old = None
        self.a_old = None


class VarBasedPolicy(Policy):
    __name__ = 'VaR-based CVaR'

    def __init__(self, Q, alpha):
        self.Q = Q
        self.alpha = alpha
        self.var = None

    def next_action(self, t):

        action_distributions = self.Q[:, t.state.y, t.state.x]

        if self.var is None:
            cvars = cvar(action_distributions, self.alpha)
            a = np.argmax(cvars)
            self.var = action_distributions[a].var(self.alpha)
        else:
            self.var = np.clip((self.var - t.reward) / gamma, MIN_VALUE, MAX_VALUE)

        a = np.argmax([d.exp_(self.var) for d in action_distributions])

        return a

    def reset(self):
        self.var = None

# ===================== algorithms
def expected_value(rv):
    return rv.expected_value()


def cvar(rv, alpha):
    return rv.cvar(alpha)

expected_value = np.vectorize(expected_value)
cvar = np.vectorize(cvar)


def policy_iteration():
    Q = init_q()
    i = 0
    while True:
        expvals = expected_value(Q)
        Q_ = eval_fixed_policy(np.argmax(expvals, axis=0))

        if converged(Q, Q_) and i != 0:
            print("policy fully learned after %d iterations" % (i,))
            break
        i += 1
        Q = Q_

    return Q


def naive_cvar_policy_iteration(alpha):
    Q = init_q()
    i = 0
    while True:
        cvars = cvar(Q, alpha)
        Q_ = eval_fixed_policy(np.argmax(cvars, axis=0))

        if converged(Q, Q_) and i != 0:
            print("naive cvar policy fully learned after %d iterations" % (i,))
            break
        i += 1
        Q = Q_

    return Q


def init_q():
    Q = np.empty((4, H, W), dtype=object)
    for ix in np.ndindex(Q.shape):

        Q[ix] = RandomVariable()
    return Q


def eval_fixed_policy(P):
    Q = init_q()
    i = 0
    while True:
        Q_ = value_update(Q, P)
        if converged(Q, Q_) and i != 0:
            break
        Q = Q_
        i += 1

    return Q


def value_update(Q, P):
    """
    One value update step.
    :param Q: (A, M, N): current Q-values
    :param P: (M, N): indices of actions to be selected
    :return: (A, M, N): new Q-values
    """

    Q_ = init_q()
    for s in states():
        for a, action_transitions in zip(actions, transitions(s)):

            # transition probabilities
            t_p = np.array([t.prob for t in action_transitions])
            # random variables created by transitioning
            t_q = [Q[P[t.state.y, t.state.x], t.state.y, t.state.x] * gamma + t.reward for t in action_transitions]
            # picked out new probability vectors
            t_p_ = np.array([q.p for q in t_q])
            # weight the prob. vectors by transition probs
            # new_p = np.einsum('i,ij->ij', t_p, t_p_)
            new_p = np.matmul(t_p, t_p_)

            Q_[a, s.y, s.x] = RandomVariable(p=new_p)

    return Q_


def converged(Q, Q_):
    p = np.array([rv.p for rv in Q.flat])
    p_ = np.array([rv.p for rv in Q_.flat])

    return np.linalg.norm(p-p_)/Q.size < 0.001


def several_epochs(arg):
    np.random.seed()
    policy, nb_epochs = arg
    rewards = np.zeros(nb_epochs)

    for i in range(nb_epochs):
        S, A, R = epoch(initial_state, policy)
        policy.reset()
        rewards[i] = np.sum(R)

    return rewards


def policy_stats(policy, alpha, nb_epochs, verbose=True):
    import copy
    import multiprocessing as mp
    threads = 4

    with mp.Pool(threads) as p:
        rewards = p.map(several_epochs, [(copy.deepcopy(policy), int(nb_epochs/threads)) for _ in range(threads)])

    rewards = np.array(rewards).flatten()

    # clip to make the decision distribution more realistic
    rewards = np.clip(rewards, MIN_VALUE, MAX_VALUE)

    var, cvar = cvar_from_samples(rewards, alpha)
    if verbose:
        print('----------------')
        print(policy.__name__)
        print('expected value=', np.mean(rewards))
        print('cvar_{}={}'.format(alpha, cvar))
        # print('var_{}={}'.format(alpha, var))

    return cvar


def exhaustive_stats(*args):
    Q = policy_iteration()

    alphas = np.array([1.0, 0.5, 0.3, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001])

    cvars = np.zeros((len(args), len(alphas)))
    names = []

    for i, policy in enumerate(args):
        names.append(policy.__name__)
        for j, alpha in enumerate(alphas):
            pol = policy(Q, alpha)

            cvars[i, j] = policy_stats(pol, alpha=alpha, nb_epochs=int(1e6), verbose=False)

            print('{}_{} done...'.format(pol.__name__, alpha))

    import pickle
    pickle.dump({'cvars': cvars, 'alphas': alphas, 'names': names}, open('stats.pkl', 'wb'))
    print(cvars)

    from visual import plot_cvars
    plot_cvars()


def epoch(start_state, policy, max_iters=100, plot_machine=None):
    """
    Evaluates a single epoch starting at start_state, using a given policy.
    :param start_state: 
    :param policy: Policy instance
    :param max_iters: end the epoch after this #steps
    :return: States, Actions, Rewards
    """
    s = start_state
    S = [s]
    A = []
    R = []
    i = 0
    r = 0
    t = Transition(s, 0, 0)
    while s not in goal_states and i < max_iters:
        a = policy.next_action(t)

        if plot_machine is not None:
            plot_machine.step(s, a)
            time.sleep(0.5)


        A.append(a)
        trans = transitions(s)[a]
        state_probs = [tran.prob for tran in trans]
        t = trans[np.random.choice(len(trans), p=state_probs)]

        r = t.reward
        s = t.state

        R.append(r)
        S.append(s)
        i += 1

    return S, A, R


def sample_runs(p, z):

    for i in range(2, 7):
        x = np.random.choice(z, size=(int(10**i)), p=p)
        print(len(x),)
        var, cvar = cvar_from_samples(x, alpha)
        print('1e{} - sample var: {}, cvar: {}'.format(i, var, cvar))


if __name__ == '__main__':

    # TODO: investigate why exp/cvar from starting state don't add up with samples
    # TODO: try naive PI
    # TODO: benchmark 1e6 on risky

    # =============== PI setup
    alpha = 0.1
    Q = policy_iteration()

    greedy_policy = GreedyPolicy(Q)
    alpha_policy = AlphaBasedPolicy(Q, alpha=alpha)
    var_policy = VarBasedPolicy(Q, alpha=alpha)
    naive_cvar_policy = NaiveCvarPolicy(Q, alpha=alpha)

    # exhaustive_stats(GreedyPolicy, AlphaBasedPolicy, NaiveCvarPolicy)

    # =============== PI stats
    nb_epochs = 100000
    policy_stats(greedy_policy, alpha, nb_epochs=nb_epochs)
    policy_stats(alpha_policy, alpha, nb_epochs=nb_epochs)
    policy_stats(var_policy, alpha, nb_epochs=nb_epochs)
    policy_stats(naive_cvar_policy, alpha, nb_epochs=nb_epochs)

    # =============== plot fixed
    Q_exp = expected_value(Q)
    V_exp = q_to_v_argmax(Q_exp)
    Q_cvar = cvar(Q, alpha)
    V_cvar = q_to_v_argmax(Q_cvar)
    # show_fixed(initial_state, q_to_v_argmax(Q_exp), np.argmax(Q_exp, axis=0))
    # show_fixed(initial_state, q_to_v_argmax(Q_cvar), np.argmax(Q_cvar, axis=0))

    # =============== plot dynamic
    plot_machine = PlotMachine(V_cvar)
    policy = var_policy
    for i in range(100):
        S, A, R = epoch(initial_state, policy, plot_machine=plot_machine)
        print('{}: {}'.format(i, np.sum(R)))
        policy.reset()

    # ============== other
    # Q_cvar = eval_fixed_policy(np.argmax(Q_cvar, axis=0))
    # interesting = Q_cvar[2, -1, 0]
    # print(interesting.cvar(alpha))
    # interesting.plot()



