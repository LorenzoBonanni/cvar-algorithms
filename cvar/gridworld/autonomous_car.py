from collections import namedtuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# a state is given by row and column positions designated (y, x)
State = namedtuple('State', ['x', 'y'])
Transition = namedtuple('Transition', ['state', 'prob', 'reward'])


class AutonomousCarNavigation:
    ACTION_LEFT = 0
    ACTION_RIGHT = 1
    ACTION_UP = 2
    ACTION_DOWN = 3
    ACTIONS = [ACTION_LEFT, ACTION_RIGHT, ACTION_UP, ACTION_DOWN]
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
    probabilities = {
        'highway': np.array([[0.36755459, 0.63158533, 0.00086008]]),
        'main': np.array([[0.42425342, 0.54990484, 0.02584174]]),
        'street': np.array([[0.69823654, 0.23496235, 0.06680112]]),
        'lane': np.array([[0.15440924, 0.8376856, 0.00790516]])
    }


    def __init__(self):
        self.map = self.create_navigation_graph()
        self.start = State(0, 3)
        self.initial_state = self.start
        self.goal = State(4, 0)
        self.goal_states = {self.goal}
        self.height = 4
        self.width = 5

    def create_navigation_graph(self):
        G = nx.Graph()

        # Add nodes
        for y in range(4):
            for x in range(5):
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

    def plot_navigation_graph(self, G):
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
        nx.draw_networkx_nodes(G, pos, nodelist=[self.start], node_color='yellow', node_size=700, edgecolors='black')
        nx.draw_networkx_nodes(G, pos, nodelist=[self.goal], node_color='red', node_size=700, edgecolors='black')

        plt.title("Autonomous Car Navigation Domain")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('navigation_graph.png')

    def states(self):
        """ iterator over all possible states """
        for y in range(4):
            for x in range(5):
                yield State(x=x, y=y)

    def target_state(self, s, a):
        """ Return the next deterministic state """
        x = s.x
        y = s.y
        if a == self.ACTION_LEFT:
            return State(max(x - 1, 0), y)
        if a == self.ACTION_RIGHT:
            return State(min(x + 1, 4), y)
        if a == self.ACTION_UP:
            return State(x, max(y - 1, 0))
        if a == self.ACTION_DOWN:
            return State(x, min(y + 1, 3))

    def transitions(self, s):
        """
        Return a list of (state, prob, reward) triples, where prob is the
        probability of transitioning to state with reward.
        """
        transitions_full = []
        state = State(x=s.x, y=s.y)
        if state == self.goal:
            return [[Transition(self.goal, 1.0, self.GOAL_REWARD)] for _ in self.ACTIONS]
        if state in self.map:
            for a in self.ACTIONS:
                transitions_actions = []
                next_state = self.target_state(state, a)
                if next_state == state:
                    transitions_actions.append(Transition(next_state, 1.0, 0.0))
                else:
                    edge_type = self.map.get_edge_data(state, next_state)
                    time_taken = self.TIME_TAKEN[edge_type['type']]
                    prob = self.probabilities[edge_type['type']][0]
                    transitions_actions = [Transition(next_state, prob[i], -time_taken[i]) for i in range(3)]
                transitions_full.append(transitions_actions)
        return transitions_full


# # Create and plot the graph
# G = create_navigation_graph()
# plot_navigation_graph(G)
#
# # Print some information about the graph
# print(f"Number of nodes: {G.number_of_nodes()}")
# print(f"Number of edges: {G.number_of_edges()}")
# print("\nExample of node neighbors:")
# print(f"Neighbors of node (0, 0): {list(G.neighbors((0, 0)))}")
# print("\nExample of edge data:")
# print(f"Edge between (0, 0) and (1, 0): {G.get_edge_data((0, 0), (1, 0))}")
# AutonomousCarNavigation().transitions(State(y=0, x=4))
