# P1 - DTMC (Benito, Christoph, Ivan)

## Instructions

Steps:

> 1. Build Markov Chain for the 18 components and the 3 types of states
> 2. Generate the transitions using the dependency data* 
> 3. Apply a transition log data* to estimate (learn) the transition matrix
> 4. Plot the transition matrix (suggest ways to visualize the convergence to steady state)
> 5. Identify the types of paths

Questions to answer:

> After 10, 20, and 30 length state-traces:
> 1. What is the probability of seeing a failure cascade that affects only component 1?
> 1. What is the probability of seeing a failure cascade that affects component 2?
> 1. What is the probability of seeing a failure masking? P(𝑆𝑢_1;𝑆𝑢_2;𝑆𝑜_1)
> 1. What is the probability of  seeing a systemic degradation?
> 1. What is the probability of normal operation?
> 1. What is the probability of have, 1, 2, or more intermittent failures?
> 1. In the case of an intermittent failure, what is the probability of a failure cascade?
> 1. In the case of a failure cascade, what is the probability of failure masking? 
> 1. Extra - In the case of an intermittent failure, what is the probability of failure masking? 

## Solution

> 1. Build Markov Chain for the 18 components and the 3 types of states

See `load_transitions()`, `add_self_loops()`, and `add_stuck_self_loops()`/`add_supervisory_component()`.

- `load_transitions()` scrapes the dependency data from the Excel sheet.
- `add_self_loops()` makes sure that each component can remain in the current state instead of switching to another one, so the model time corresponds to the real time.
- `add_stuck_self_loops()` and `add_supervisory_component()` are two alternative `completion_strategy`s that can be used to complete the Markov Chain. The first one adds a self-loop to each component so that it can get stuck in the current state. The second one adds a supervisory component that can be used to detect stuck components and reset them to one of the normal states.

> 2. Generate the transitions using the dependency data

See `generate_transitions()`. We define the proportions of switching from one state (operational, degraded, or unresponsive) to another state. We use the normalized Kronecker product to generate the transition matrix for the final DTMC that has one state for each combination of component and state.

> 3. Apply a transition log data to estimate (learn) the transition matrix

See `random_walks()` and `estimate_transition_matrix()`.

- `random_walks()` generates random walks through the DTMC based on the transition probabilities.
- `estimate_transition_matrix()` uses statistical interference to estimate the transition matrix based on the generated random walks.

> 4. Plot the transition matrix (suggest ways to visualize the convergence to steady state)

See `approximate_limiting_distribution()` and `visualize_dtmc()`.

- `approximate_limiting_distribution()` multiplies the transition matrix with itself for each step (it is an approximation only because the iterative approach does not allow for passing an infinite `num_steps` parameter).
  - We also considered a closed-form solution for computing the limiting distribution by solving $\pi \cdot P = \pi \wedge \sum_{s\in S}{\pi(s)} = 1$ to $\pi$ using `sympy`. However, this would require us to reduce the DTMC first and identify its BSCCs, so we discontinued this approach.
- `visualize_dtmc()` visualizes the DTMC using the `graphviz` library. The nodes are colored based on the limiting distribution and the edges are colored based on the transition probabilities.

Additionally, we created an animation that displays the convergence of the DTMC in `calculate_and_visualize_limiting_distribution()` by rendering the DTMC for a series of points in time and combining them into a video using `ffmpeg`.

Furthermore, we created a pie chart that shows the approximated limiting distribution.

> 5. Identify the types of paths

See `answer_questions()`.

For answering the questions, we decided to keep it simple, stupid. Instead of developing an efficient algorithm for each question, we used GitHub Copilot to generate a naive numpy/list comprehension query for each question and manually fixed the corrected the queries as needed.

## Implementation notes

- The interface to our solution is the parametrized `main()` method which is intended to be called from an IPython/Jupyter/pdb context with the desired arguments.
- Depending on the specified arguments, we cache different intermediate results using `pickle` to speed up the feedback loop.