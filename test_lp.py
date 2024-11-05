import pickle

import numpy as np
import pulp
from pulp import PULP_CBC_CMD

from simple_env import SimpleEnv, State

alpha = 0.01
alpha_i =  0.01
alpha_i_next = 1
v_s0_alpha_i_next = 3.7
v_s0_alpha_i = 0

q = []
env = SimpleEnv()
s = State(0)
transitions = env.transitions(s)
value = {
    State(0): [v_s0_alpha_i, v_s0_alpha_i_next],
    State(1): [0, 0],
}
for i in range(20):
    q = []
    for ACTION in range(2):
        t_action = transitions[ACTION]
        r_1 = t_action[0].reward
        p_1 = t_action[0].prob
        r_2 = t_action[1].reward
        p_2 = t_action[1].prob
        problem = pulp.LpProblem("CVaR", pulp.LpMinimize)
        variables = pulp.LpVariable.dicts("xi", range(2), lowBound=0, upBound=1 / alpha, cat='Continuous')

        problem += variables[0]*r_1*p_1 + variables[1]*r_2*p_2

        problem += (variables[0]*p_1 + variables[1]*p_2) == 1
        problem.solve(PULP_CBC_CMD(msg=False))

        if problem.status != 1:
            print("Problem status:", problem.status)
            exit(1)

        # print("Solution:", pulp.value(problem.objective))
        # print("Variables:", [pulp.value(variables[i]) for i in range(2)])
        q.append(pulp.value(problem.objective))

    v_s0_alpha_i = max(q)

hand_values = np.array([
    [-0.6], [v_s0_alpha_i], [v_s0_alpha_i_next]
])
hand_values = np.hstack((hand_values, np.zeros_like(hand_values)))

print(hand_values)
pickle.dump(hand_values, open('cvar_vi_hand.pkl', mode='wb'))