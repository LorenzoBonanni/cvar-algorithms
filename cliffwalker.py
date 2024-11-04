import random
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np

# helper data structures:
# a state is given by row and column positions designated (y, x)
# State = namedtuple('State', ['y', 'x'])

class State:
    def __init__(self, y, x, NROW=14, NCOL=16):
        self.y = y
        self.x = x
        self.NROW = NROW
        self.NCOL = NCOL
        self.id = int(np.ravel_multi_index((y, x), (self.NROW, self.NCOL)))

    def __eq__(self, other):
        return self.y == other.y and self.x == other.x

    def __hash__(self):
        return self.id

    def __repr__(self):
        return f"({self.y}, {self.x})"

# encapsulates a transition to state and its probability
Transition = namedtuple('Transition', ['state', 'prob', 'reward'])  # transition to state with probability prob

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

class GridWorld:
    """ Cliffwalker. """

    ACTION_LEFT = 0
    ACTION_RIGHT = 1
    ACTION_UP = 2
    ACTION_DOWN = 3
    ACTIONS = [ACTION_LEFT, ACTION_RIGHT, ACTION_UP, ACTION_DOWN]
    FALL_REWARD = -2
    ACTION_NAMES = {ACTION_LEFT: "Left", ACTION_RIGHT: "Right", ACTION_UP: "Up", ACTION_DOWN: "Down"}

    def __init__(self, height, width, random_action_p=0.1, risky_p_loss=0.15, path=None, goal_pos=(1, 15), start_pos=(12, 15)):
        self.risky_p_loss = risky_p_loss
        self.random_action_p = random_action_p

        # self.risky_goal_states = {State(0, 5)}
        self.risky_goal_states = {}
        im = plt.imread(path)
        im = rgb2gray(im)
        self.height, self.width =  im.shape
        cliff = np.where(im == 0)

        goal_pos = goal_pos
        start_pos = start_pos
        self.initial_state = State(start_pos[0], start_pos[1], self.height, self.width)
        # self.absorbing_state = State(-1, -1)
        self.goal_states = {State(goal_pos[0], goal_pos[1], self.height, self.width)}

        self.cliff_states = set(State(cs[0], cs[1]) for cs in zip(*cliff))
        self.terminal_states = self.goal_states | self.cliff_states
        self.Ns = self.height * self.width

    def states(self):
        """ iterator over all possible states """
        for x in range(self.width):
            for y in range(self.height):
                s = State(y, x, self.height, self.width)
                if s in self.cliff_states:
                    continue
                yield s

    def target_state(self, s, a):
        """ Return the next deterministic state """
        x, y = s.x, s.y
        if a == self.ACTION_LEFT:
            return State(y, max(x - 1, 0), self.height, self.width)
        if a == self.ACTION_RIGHT:
            return State(y, min(x + 1, self.width - 1), self.height, self.width)
        if a == self.ACTION_UP:
            return State(max(y - 1, 0), x, self.height, self.width)
        if a == self.ACTION_DOWN:
            return State(min(y + 1, self.height - 1), x, self.height, self.width)

    def transitions(self, s):
        """
        returns a list of Transitions from the state s for each action, only non zero probabilities are given
        serves the lists for all actions at once
        """
        if s in self.goal_states:
            return [[Transition(state=s, prob=1.0, reward=0)] for a in self.ACTIONS]

        # if s in self.risky_goal_states:
        #     goal = next(iter(self.goal_states))
        #     return [[Transition(state=goal, prob=self.risky_p_loss, reward=-50),
        #              Transition(state=goal, prob=1-self.risky_p_loss, reward=100)] for a in self.ACTIONS]

        transitions_full = []
        for a in self.ACTIONS:
            curr_states_trans = {}

            # over all *random* actions
            for a_ in self.ACTIONS:
                s_ = self.target_state(s, a_)
                if s_ in self.cliff_states:
                    r = self.FALL_REWARD
                else:
                    r = -1
                initial_trans = curr_states_trans.get((s_.y, s_.x), None)
                prob = 1.0 - self.random_action_p if a_ == a else self.random_action_p / 3
                if initial_trans is None:
                    curr_states_trans[(s_.y, s_.x)] = Transition(s_, prob, r)
                else:
                    curr_states_trans[(s_.y, s_.x)] = Transition(s_, prob+initial_trans.prob, r)

            transitions_actions = [tran for state, tran in curr_states_trans.items() if tran.prob != 0]
            transitions_full.append(transitions_actions)

        return transitions_full

    def sample_transition(self, s, a):
        """ Sample a single transition, duh. """
        trans = self.transitions(s)[a]
        state_probs = [tran.prob for tran in trans]
        transition = random.choices(population=trans, weights=state_probs)[0]
        return transition


#
# if __name__ == '__main__':
#     from cvar.gridworld.core.constants import *
#     from cvar.gridworld.plots.grid import grid_plot
#     import matplotlib.pyplot as plt
#
#     world = GridWorld(14, 16, random_action_p=0.05, path='gridworld3.png')
#     grid_plot(world)
#     plt.show()
#     # for i in range(20):
#     #     print('seed=', i)
#     #     np.random.seed(i)
#     #     world = GridWorld(14, 16, random_action_p=0.05, path='gridworld3.png')
#     #     grid_plot(world)
#     #     plt.show()