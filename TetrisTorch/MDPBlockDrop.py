import mdpsolver
from itertools import product

# Parameters
m = 6  # max height
n = 5  # number of columns
terminal_state = m**n  # the last index is the terminal state

# Generate all valid (non-terminal) states
states = list(product(range(m), repeat=n))
state_to_idx = {s: i for i, s in enumerate(states)}
idx_to_state = {i: s for s, i in state_to_idx.items()}

# Add terminal state
states.append("TERMINAL")

# Actions: drop block in column 0 or column 1
actions = list(range(n))

# Initialize transition matrix and rewards
num_states = len(states)
num_actions = n
tranMat = [[[0.0 for _ in range(num_states)] for _ in range(num_actions)] for _ in range(num_states)]
rewards = [[0.0 for _ in range(num_actions)] for _ in range(num_states)]

# Build transitions and rewards
for i, state in enumerate(states):
    if state == "TERMINAL":
        for a in actions:
            tranMat[i][a][terminal_state] = 1.0
            rewards[i][a] = 0
        continue

    for a in actions:
        next_state = list(state)
        next_state[a] += 1

        if next_state[a] >= m:
            # Game over
            tranMat[i][a][terminal_state] = 1.0
            rewards[i][a] = -m
        elif all(h == next_state[0] for h in next_state):
            # Line cleared (all equal): subtract 1 from each column
            cleared = tuple(h - 1 for h in next_state)
            j = state_to_idx[cleared]
            tranMat[i][a][j] = 1.0
            rewards[i][a] = 1
        else:
            # Just increase selected column
            j = state_to_idx[tuple(next_state)]
            tranMat[i][a][j] = 1.0
            rewards[i][a] = 0

# Solve the MDP
mdl = mdpsolver.model()
mdl.mdp(discount=0.9, rewards=rewards, tranMatWithZeros=tranMat)
mdl.solve()

# Print optimal policy (column to drop into for each state)
policy = mdl.getPolicy()

# Show result in readable format
for i, a in enumerate(policy):
    state_str = states[i] if states[i] != "TERMINAL" else "TERMINAL"
    print(f"State {state_str}: Drop in column {a}")
