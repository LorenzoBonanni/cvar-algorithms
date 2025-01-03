import random
from collections import namedtuple
from copy import deepcopy
from functools import partial

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# a state is given by row and column positions designated (y, x)
class State:
    def __init__(self, y, x, NROW=14, NCOL=16, NTYPES=3, type=0):
        self.y = y
        self.x = x
        self.NROW = NROW
        self.NCOL = NCOL
        self.NTYPES = NTYPES
        self.type = type
        self.id = int(np.ravel_multi_index((y, x, type), (NROW, NCOL, NTYPES)))

    def __eq__(self, other):
        return (self.id == other.id) or (self.y == other.y and self.x == other.x)

    def __hash__(self):
        return self.id

    def __repr__(self):
        return f"({self.y}, {self.x}, {self.type})"

    def __copy__(self):
        return State(self.y, self.x, self.NROW, self.NCOL, self.NTYPES, self.type)


Transition = namedtuple('Transition', ['state', 'prob', 'reward'])


class AutonomousCarNavigation:
    ACTION_LEFT = 0
    ACTION_RIGHT = 1
    ACTION_UP = 2
    ACTION_DOWN = 3
    ACTIONS = [ACTION_LEFT, ACTION_RIGHT, ACTION_UP, ACTION_DOWN]
    ACTION_NAMES = ['LEFT', 'RIGHT', 'UP', 'DOWN']
    # Define road types and their corresponding colors
    ROAD_TYPES = {
        'highway': 'black',
        'main': 'red',
        'street': 'blue',
        'lane': 'green'
    }
    GOAL_REWARD = 80
    # Define time taken to travel between nodes
    # fast, medium, slow
    TIME_TAKEN = {
        'highway': [1, 2, 18],
        'main': [2, 4, 13],
        'street': [4, 5, 11],
        'lane': [7, 7, 8]
    }

    # Define the probabilities of transitioning to each road type
    # Sampled from dirichlet([1, 1, 0.4])
    # probabilities = {
    #     'highway': np.array([[0.36755459, 0.63158533, 0.00086008]]),
    #     'main': np.array([[0.42425342, 0.54990484, 0.02584174]]),
    #     'street': np.array([[0.69823654, 0.23496235, 0.06680112]]),
    #     'lane': np.array([[0.15440924, 0.8376856, 0.00790516]])
    # }
    probabilities = {
        'highway': np.array([[1 / 3, 1 / 3, 1 / 3]]),
        'main': np.array([[1 / 3, 1 / 3, 1 / 3]]),
        'street': np.array([[1 / 3, 1 / 3, 1 / 3]]),
        'lane': np.array([[1 / 3, 1 / 3, 1 / 3]])
    }

    def __init__(self):
        self.height = 4
        self.width = 5
        self.ntypes = 3
        self.State = partial(State, NROW=self.height, NCOL=self.width, NTYPES=self.ntypes)
        self.start = self.State(x=0, y=3, type=0)
        self.initial_state = self.start
        self.goal = self.State(x=4, y=0, type=0)
        self.goal_states = {self.State(x=4, y=0, type=t) for t in range(self.ntypes)}
        self.map = self.create_navigation_graph()
        self.Ns  = self.height * self.width * self.ntypes


    def create_navigation_graph(self):
        G = nx.Graph()

        # Add nodes
        for y in range(self.height):
            for x in range(self.width):
                G.add_node((x, y), pos=(x, -y))  # Negative y to flip the grid vertically

        # # Add horizontal edges
        horizontal_edges = [
            ((4, 0), (4, 1), 'street'), ((4, 1), (4, 2), 'main'), ((4, 2), (4, 3), 'highway'),
        ]
        G.add_edges_from((start, end, {'type': road_type}) for start, end, road_type in horizontal_edges)

        # HIGHWAY
        for i in range(4):
            G.add_edge((i, 3), (i + 1, 3), type='highway')

        for i in range(1, 5):
            G.add_edge((i, 2), (i, 3), type='highway')

        # MAIN
        for i in range(4):
            G.add_edge((i, 2), (i + 1, 2), type='main')

        # STREET
        for i in range(4):
            G.add_edge((i, 1), (i + 1, 1), type='street')

        for i in range(2):
            for j in range(1, 4):
                G.add_edge((j, i), (j, i + 1), type='street')

        # LANE
        for i in range(4):
            G.add_edge((i, 0), (i + 1, 0), type='lane')
        for i in range(3):
            G.add_edge((0, i), (0, i + 1), type='lane')

        return G

    def states(self):
        """ iterator over all possible states """
        for y in range(self.height):
            for x in range(self.width):
                if self.State(x=x, y=y) in self.goal_states:
                    continue
                for t in range(self.ntypes):
                    yield self.State(x=x, y=y, type=t)

    def actions(self, s):
        actions = self.ACTIONS.copy()
        if s.x == 0:
            actions.remove(self.ACTION_LEFT)
        elif s.x == self.width - 1:
            actions.remove(self.ACTION_RIGHT)

        if s.y == 0:
            actions.remove(self.ACTION_DOWN)
        elif s.y == self.height - 1:
            actions.remove(self.ACTION_UP)

        return actions

    def target_state(self, s, a):
        """ Return the next deterministic state """
        x = s.x
        y = s.y
        if a == self.ACTION_LEFT:
            return self.State(x=max(x - 1, 0), y=y)
        if a == self.ACTION_RIGHT:
            return self.State(x=min(x + 1, self.width - 1), y=y)
        if a == self.ACTION_UP:
            return self.State(x=x, y=min(y + 1, self.height - 1))
        if a == self.ACTION_DOWN:
            return self.State(x=x, y=max(y - 1, 0))

    def is_terminal(self, s):
        return s in self.goal_states

    def transitions(self, s):
        """
        Return a list of (state, prob, reward) triples, where prob is the
        probability of transitioning to state with reward.
        """
        transitions_full = []
        if s in self.goal_states:
            return [[Transition(self.goal, 1.0, self.GOAL_REWARD)] for _ in self.ACTIONS]

        for a in self.ACTIONS:
            next_state = self.target_state(s, a)
            if next_state in self.goal_states:
                reward = self.GOAL_REWARD
            else:
                reward = 0

            if next_state == s:
                next_states = [self.State(x=next_state.x, y=next_state.y, type=t) for t in range(self.ntypes)]
                transitions_actions = [Transition(next_states[i], 1/3, 0) for i in range(len(next_states))]
            else:
                edge_type = self.map.get_edge_data((s.x, s.y), (next_state.x, next_state.y))
                time_taken = self.TIME_TAKEN[edge_type['type']]
                prob = self.probabilities[edge_type['type']][0]
                next_states = [self.State(x=next_state.x, y=next_state.y, type=t) for t in range(self.ntypes)]
                transitions_actions = [Transition(next_states[i], float(prob[i]), -time_taken[i]+reward) for i in
                                       range(len(time_taken))]
            transitions_full.append(transitions_actions)

        return transitions_full

    def plot_value_function(self, value, suffix):
        plt.figure(figsize=(12, 10))
        reshaped_value = value.reshape(self.height, self.width, self.ntypes)
        plt.imshow(reshaped_value, cmap='viridis')
        for (i, j), val in np.ndenumerate(reshaped_value):
            plt.text(j, i, f'{val:.2f}', ha='center', va='center', color='white')
        plt.colorbar()
        plt.title(f"Value function {suffix}")
        plt.xticks(np.arange(self.width), np.arange(self.width))
        plt.yticks(np.arange(self.height), np.arange(self.height))
        plt.tight_layout()
        plt.savefig(f'../plots/value_function_{suffix}.png')
        plt.close()

    def plot_navigation_graph(self):
        G = self.map
        plt.figure(figsize=(12, 10))
        pos = nx.get_node_attributes(G, 'pos')

        # Draw edges
        for (u, v, data) in G.edges(data=True):
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=self.ROAD_TYPES[data['type']], width=2)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='white', edgecolors='black')

        # Draw labels
        labels = {node: f"({node[0]},{node[1]})" for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)

        # Highlight start and goal
        nx.draw_networkx_nodes(G, pos, nodelist=[(self.start.x, self.start.y)], node_color='yellow', node_size=700,
                               edgecolors='black')
        nx.draw_networkx_nodes(G, pos, nodelist=[(self.goal.x, self.goal.y)], node_color='red', node_size=700,
                               edgecolors='black')

        plt.title("Autonomous Car Navigation Domain")
        plt.axis('off')

        plt.tight_layout()

        plt.legend(handles=[plt.Line2D([0], [0], color=color, lw=4, label=road_type) for road_type, color in
                            self.ROAD_TYPES.items()], loc='center right', bbox_to_anchor=(0.9, 0.5), fontsize='x-large')
        plt.savefig('../plots/navigation_graph.png')

    def plot_trajectory(self, policy, suffix):
        G = self.map
        plt.figure(figsize=(12, 10))
        pos = nx.get_node_attributes(G, 'pos')

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='white', edgecolors='black')
        s0 = self.initial_state
        s = deepcopy(s0)
        while s not in self.goal_states:
            action = policy[s.id]
            u = (s.x, s.y)
            next_state = self.target_state(s, action)
            v = (next_state.x, next_state.y)
            s = next_state
            data = G.get_edge_data(u, v)
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=self.ROAD_TYPES[data['type']], width=2)

        # Draw labels
        labels = {node: f"({node[0]},{node[1]})" for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)

        # Highlight start and goal
        nx.draw_networkx_nodes(G, pos, nodelist=[(self.start.x, self.start.y)], node_color='yellow', node_size=700,
                               edgecolors='black')
        nx.draw_networkx_nodes(G, pos, nodelist=[(self.goal.x, self.goal.y)], node_color='red', node_size=700,
                               edgecolors='black')

        plt.title(f"Autonomous Car Navigation Domain {suffix}")
        plt.axis('off')

        plt.tight_layout()

        plt.legend(handles=[plt.Line2D([0], [0], color=color, lw=4, label=road_type) for road_type, color in
                            self.ROAD_TYPES.items()], loc='center right', bbox_to_anchor=(0.9, 0.5), fontsize='x-large')
        plt.savefig(f'../plots/trajectory_traffic_{suffix}.png')
        plt.close()

    def generate_plots(self, policy, value, suffix):
        self.plot_trajectory(policy, suffix)
        # self.plot_value_function(value, suffix)

    def sample_transition(self, s, a):
        transitions = self.transitions(s)[a]
        probs = [t.prob for t in transitions]
        transition = random.choices(transitions, probs)[0]
        return transition

# # Create and plot the graph
#
# G = env.map
# env.plot_navigation_graph()
#
# # Print some information about the graph
# print(f"Number of nodes: {G.number_of_nodes()}")
# print(f"Number of edges: {G.number_of_edges()}")
# print("\nExample of node neighbors:")
# print(f"Neighbors of node (0, 0): {list(G.neighbors((0, 0)))}")
# print("\nExample of edge data:")
# print(f"Edge between (0, 0) and (1, 0): {G.get_edge_data((0, 0), (1, 0))}")
# print(
#     env.transitions(env.initial_state)
# )
