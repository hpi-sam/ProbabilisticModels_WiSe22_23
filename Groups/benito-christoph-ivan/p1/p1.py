from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tqdm import tqdm
import typing


def load_transitions(file_name):
	sheet_name = 'prior_transition_matrix'
	num_components = 18

	transitions = pd.read_excel(file_name, sheet_name=sheet_name, skiprows=1, index_col=0)
	transitions = transitions.iloc[:, 0:num_components]

	transitions.columns = transitions.columns.str.strip()
	transitions.index = transitions.index.str.strip()

	return transitions

def add_self_loops(transitions):
	extended_transitions = transitions.copy()
	for component in extended_transitions.index:
		extended_transitions.loc[component, component] = 1 - extended_transitions.loc[component].sum()
	return extended_transitions

def add_supervisory_component(transitions):
	supervisory_component = "Supervisory Component"
	supervisory_outgoing = [
		"Reputation Service", "Authentication Service",
		"Bid and Buy Service", "Inventory Service",
		"User Management Service",
		"Item Management Service",
		"Supervisory Component"
	]

	extended_transitions = transitions.copy()
	extended_transitions[supervisory_component] = 0
	extended_transitions[supervisory_component] = 1 - extended_transitions.sum(axis=1)
	extended_transitions.loc[supervisory_component] = 0
	extended_transitions.loc[supervisory_component, supervisory_outgoing] = 1 / (len(supervisory_outgoing))
	return extended_transitions

STATES: typing.List[StateType] = ['operational', 'degraded', 'unresponsive']

def generate_transitions(component_transitions):
	proportions = pd.DataFrame(
		[[0.6, 0.15, 0.02], [0.3, 0.5, 0.1], [0.2, 0.2, 0.1]],
		index=STATES,
		columns=STATES
	)
	large_transitions = pd.DataFrame(
		np.kron(component_transitions, proportions),
		index=pd.MultiIndex.from_product([component_transitions.index, proportions.index]),
		columns=pd.MultiIndex.from_product([component_transitions.columns, proportions.index])
	)
	# normalize row-wise
	large_transitions = large_transitions.div(large_transitions.sum(axis=1), axis=0)
	return large_transitions

def random_walk(transitions, num_steps):
	current_state = np.random.choice(transitions.index)
	yield current_state
	for _ in range(num_steps):
		assert transitions.loc[current_state].sum().sum().round(2) == 1

		current_state = np.random.choice(transitions.columns, p=transitions.loc[current_state])
		yield current_state

def random_walks(transitions, num_walks, num_steps):
	for _ in tqdm(range(num_walks), desc='Generating random walks'):
		yield list(random_walk(transitions, num_steps))

def estimate_transition_matrix(transition_logs, compound_states):
	transition_matrix = pd.DataFrame(0, index=compound_states, columns=compound_states)
	for log in tqdm(transition_logs, desc='Estimating transition matrix'):
		for compound, next_compound in zip(log, log[1:]):
			transition_matrix.loc[compound, next_compound] += 1
	transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)
	# remove all unreached components (suspicious, but keeps the DTMC valid)
	transition_matrix = transition_matrix.loc[transition_matrix.sum(axis=1) > 0, transition_matrix.sum(axis=1) > 0]
	return transition_matrix

def calculate_limiting_distribution(transitions):
	# solve $\pi \cdot P = \pi \wedge \sum_{s\in S}{\pi(s)} = 1$ to $\pi$
	# we express this as a linear system of equations and solve it using numpy

	#             p11 p12 p13
	#             p21 p22 p23
	#             p31 p32 p33
	# pi1 pi2 pi3 1   1   1   = pi1 pi2 pi3
	#
	# pi1 p11 + pi2 p21 + pi3 p31 = pi1
	# pi1 p12 + pi2 p22 + pi3 p32 = pi2
	# pi1 p13 + pi2 p23 + pi3 p33 = pi3

	# sympy hello world
	# 0 = 2y+1
	# y = sympy.Symbol('y')
	# sympy.solve(2*y+1, y)

	import sympy
	pis = [sympy.Symbol(f'pi_{i}') for i in range(len(transitions.index))]
	solution = sympy.solve(
		[
			*[
				sum(pis[i] * transitions.loc[[transitions.index[i]], [transitions.index[j]]].sum().sum() for j in range(len(transitions.index))) - pis[i]
				for i in range(len(transitions.index))
			],
			sum(pis) - 1
		],
		pis
	)
	import pdb; pdb.set_trace()

	return pd.Series({
		transitions.index[i]: solution[pis[i]]
		for i in range(len(transitions.index))
	})

def approximate_limiting_distribution_gen(transitions, num_steps):
	distribution = pd.Series(1 / len(transitions), transitions.index)
	for _ in range(num_steps):
		yield (distribution := distribution.dot(transitions))

def approximate_limiting_distribution(transitions, num_steps):
	return list(approximate_limiting_distribution_gen(transitions, num_steps))[-1]

def visualize_dtmc(estimated_transitions, distribution, png_path):
	import graphviz

	dot = graphviz.Digraph()
	for compound in estimated_transitions.index:
		# for the color of the node, use the probability in distribution (0=white, 1=red)
		r = 1
		g = b = 1 - (distribution[compound] / distribution.max())
		color = f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}'
		dot.node(str(compound), style='filled', fillcolor=color)
		# TODO: improve positioning?
	for compound in estimated_transitions.index:
		for next_compound in estimated_transitions.index:
			probability = estimated_transitions.loc[[compound], [next_compound]].sum().sum()
			if probability > 0:
				dot.edge(str(compound), str(next_compound), label=f"{probability:.2f}")

	dot.format = 'png'
	dot.render(png_path)

def main(completion_strategy=add_supervisory_component, anim=False):
	component_transitions = load_transitions('../../../ProjectDescriptions/mRubis_Transition_Matrix.xlsx')

	component_transitions = completion_strategy(component_transitions)

	transitions = generate_transitions(component_transitions)
	logs = list(random_walks(transitions, 20, 100))
	estimated_transitions = estimate_transition_matrix(logs, transitions.index)

	estimated_transitions = estimated_transitions.fillna(0)

	if anim:
		for (i, distribution) in enumerate(tqdm(
			approximate_limiting_distribution_gen(estimated_transitions, 100),
			desc='Rendering limiting distribution',
			total=100)):
			visualize_dtmc(estimated_transitions, distribution, f'./dtmc_frame_{i}')
		# TODO: this is still very slow - took 30 minutes on my machine. run with caution!
		import subprocess
		subprocess.run([
			'ffmpeg',
			'-y',
			'-framerate', '10',
			'-i', './dtmc_frame_%d.png',
			'-c:v', 'libx264',
			'-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
			'-profile:v', 'high',
			'-crf', '20',
			'-pix_fmt', 'yuv420p',
			'./dtmc.mp4'
		])
		# remove intermediate images
		import glob
		for path in glob.glob('./dtmc_frame_*.png'):
			os.remove(path)
	limiting_distribution = approximate_limiting_distribution(estimated_transitions, 100)
	visualize_dtmc(estimated_transitions, limiting_distribution, f'./dtmc')

if __name__ == '__main__':
	import pdb; pdb.set_trace()
	main()

# NEXT: see below

# tell me a joke
# once upon a time there was a programmer who wanted to learn about markov chains
# he wrote a script to generate random walks through a markov chain
# he then used the random walks to estimate the transition matrix of the markov chain
# but suddenly, he realized that he had no idea how to calculate the limiting distribution of the markov chain
# so he went to the internet and found a solution

# todos
# - answer questions
# - chris:
#   > Use this discussion of self-loops and supervisory component to solidify your understanding of the Markov Chain properties of being reducible and irreducible, periodic, which relate to being ergodic...
# - visualize *convergence* to steady state
#   > It could as simply as line charts of the probability of each state or something more elaborate like some cumulative regret of how much each state is, at any given moment, distant from the its steady state probability. This will also show you that certain states take longer to achieve their steady state probability, simply because they are more seldomly visited.
# - do we want to reconsider the analytical solution?

# next meeting: tue 13:00 - 18:00
