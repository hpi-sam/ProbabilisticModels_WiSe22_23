"""
CTMC transient distribution computation using matrix powers and uniformization.

Usage: Adjust the inputs below and run the script.

Learnings:
- Matrix powers need many iterations to converge.
- Matrix powers are not numerically stable. Numerical instability does *not* mean that theoretical convergence is not guaranteed (as I assumed earlier) but that practical rounding errors are magnified to the point that the result is entirely incorrect. In the example, we needed to use float128s because float64s produced incorrect results (probabilites outside [0, 1]). In practice, we would need to use arbitrary precision floats using sympy to ensure correctness (high memory consumption, low speed).
- Uniformization (discretizing CTMC to DTMC and computing probabilities) is numerically stable and converges faster than matrix powers, but requires a uniformization rate to be specified. There are heuristics to compute the uniformization rate, but according to ChatGPT, manual tuning might be required.
- np.array.__pow__ and np.array.__iadd__ do not do what you might think they do. :-)
"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
np.set_printoptions(precision=4, suppress=True)

# --- Inputs ---
P = np.array([[0, 1], [2, 0]])
t = 3
steps = 50
# Uniformization rate (or None to disable)
uniformization_rate = 5
# --------------

Q = P - np.diag(np.sum(P, axis=1, dtype='float128'))

if uniformization_rate:
    # Compute the matrix exponential using uniformization
    # Formula: dist = row_wise_norm(((Q + uniformization_rate * I)*t)^steps)

    Q_mod = Q + uniformization_rate * np.eye(Q.shape[0])
    uni_dist = np.linalg.matrix_power(Q_mod * t, steps)
    uni_dist = uni_dist / np.sum(uni_dist, axis=1, keepdims=True) # Normalize the rows to sum to 1

    print("Using uniformization", uni_dist, sep="\n")

# Approximate transition probabilities up to 20 units of time
# Formula: dist = sum_{i=1}^steps (Q^i * t^i / i!) (letting zero-matrix power by 0 = 1)
approx_dist_history = []
approx_dist = np.identity(Q.shape[0])
for i in range(1, steps):
    approx_dist = approx_dist + (np.linalg.matrix_power(Q, i)) * float((t ** i) / np.math.factorial(i))
    approx_dist_history.append(approx_dist)
print("Distribution using matrix powers", approx_dist, sep="\n")

fig, ax = plt.subplots(Q.shape[0], Q.shape[1])
for i in range(Q.shape[0]):
    for j in range(Q.shape[1]):
        if uni_dist is not None:
            ax[i, j].axhline(uni_dist[i, j], color="red")

        ax[i, j].plot([dist_i[i, j] for dist_i in approx_dist_history])
        ax[i, j].set_title(f"dist[{i}, {j}] value history")
        ax[i, j].legend(["Matrix powers"])

        if uni_dist is not None:
            ax[i, j].legend(["Uniformization", "Matrix powers"])

        ax[i, j].set_ylim([-0.5, 1.5])

plt.tight_layout()
plt.show()
