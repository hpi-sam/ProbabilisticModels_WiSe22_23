#!/usr/bin/env python3

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

def add_self_loops(transitions, probability=0.5):
	extended_transitions = transitions.copy()
	for component in extended_transitions.index:
		extended_transitions.loc[component, component] = probability / (1 - probability)  # denormalized
	# normalize row-wise
	extended_transitions = extended_transitions.div(extended_transitions.sum(axis=1), axis=0)
	return extended_transitions

def add_stuck_self_loops(transitions):
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
		[[0.6, 0.15, 0.02], [0.3, 0.5, 0.5], [0.2, 0.2, 0.1]],
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

def visualize_dtmc(estimated_transitions, distribution, max_probability, png_path):
	import graphviz

	dot = graphviz.Digraph()
	for compound in estimated_transitions.index:
		r = 1
		g = b = 1 - (distribution[compound] / max_probability)
		color = f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}'
		dot.node(str(compound), style='filled', fillcolor=color)
		# TODO: improve positioning? (optional)
	for compound in estimated_transitions.index:
		for next_compound in estimated_transitions.index:
			probability = estimated_transitions.loc[[compound], [next_compound]].sum().sum()
			if probability > 0:
				dot.edge(str(compound), str(next_compound), label=f"{probability:.2f}")

	dot.format = 'png'
	dot.render(png_path)

def answer_questions(transitions, distributions):
	# Types of traces:
	# Normal operation:
	#   c1/operational -> c2/operational
	# Intermittent failure:
	#   c1/operational -> c1/degraded -> c1/operational
	# Systemic degradation:
	#   c1/degraded -> c2/degraded
	# Failure masking:
	#   c1/unresponsive -> c2/unresponsive -> c1/operational
	# Failure cascade:
	#   c1/degraded -> c1/unresponsive -> c2/unresponsive
	#   c1/degraded -> c2/degraded -> c2/unresponsive
	import pdb; pdb.set_trace()

	for n in [10, 20, 30]:
		print(f"After {n} steps:")

		print("What is the probability of seeing a failure cascade that affects only component 1 (Bid and Buy Service)?")
		result = sum(
			distributions[i][(component, 'degraded')] * transitions.loc[(component, 'degraded'), ('Bid and Buy Service', 'degraded')] * transitions.loc[('Bid and Buy Service', 'degraded'), ('Bid and Buy Service', 'unresponsive')]
			for component in transitions.index.get_level_values(0).unique().difference(['Bid and Buy Service'])
			for i in range(0, n + 1 - 2)
		) / (n + 1 - 2)
		print(result)

		print("What is the probability of seeing a failure cascade that affects component 2 (Item Management Service)?")
		result = sum(
			distributions[i][(component, 'degraded')] * (
				transitions.loc[(component, 'degraded'), (component, 'unresponsive')] * transitions.loc[(component, 'unresponsive'), ('Item Management Service', 'unresponsive')]
				+
				transitions.loc[(component, 'degraded'), ('Item Management Service', 'degraded')] * transitions.loc[('Item Management Service', 'degraded'), ('Item Management Service', 'unresponsive')]
			for component in transitions.index.get_level_values(0).unique().difference(['Item Management Service'])
			for i in range(0, n + 1 - 2)
		) / (n + 1 - 2)
		print(result)

		print("What is the probability of seeing a failure masking?")
		result = sum(
			distributions[i][(component_1, 'unresponsive')] * transitions.loc[(component_1, 'unresponsive'), (component_2, 'unresponsive')] * transitions.loc[(component_2, 'unresponsive'), (component_1, 'operational')]
			for component_1 in transitions.index.get_level_values(0).unique()
			for component_2 in transitions.index.get_level_values(0).unique().difference([component_1])
			for i in range(0, n + 1 - 2)
		) / (n + 1 - 2)
		print(result)

		print("What is the probability of seeing a systemic degradation?")
		result = sum(
			distributions[i][(component_1, 'degraded')] * transitions.loc[(component_1, 'degraded'), (component_2, 'degraded')]
			for component_1 in transitions.index.get_level_values(0).unique()
			for component_2 in transitions.index.get_level_values(0).unique().difference([component_1])
			for i in range(0, n + 1 - 1)
		) / (n + 1 - 1)
		print(result)

		print("What is the probability of normal operation?")
		result = sum(
			distributions[i][(component, 'operational')]
			for component in transitions.index.get_level_values(0).unique()
			for i in range(0, n + 1)
		) / (n + 1)
		print(result)

		print("What is the probability of having 1, 2, or more intermittent failures?")
		result = sum(
			distributions[i][(component, 'operational')] * transitions.loc[(component, 'operational'), (component, 'degraded')] * transitions.loc[(component, 'degraded'), (component, 'operational')]
			for component in transitions.index.get_level_values(0).unique()
			for i in range(0, n + 1 - 2)
		) / (n + 1 - 2)
		print(result)

		print("In the case of an intermittent failure, what is the probability of a failure cascade?")
		# TODO: Does "in the case" mean "immediately after", "at any time after", or that the failure cascade takes place in the middle of the intermittent failure?

		print("In the case of a failure cascade, what is the probability of failure masking?")

		print("Extra - In the case of an intermittent failure, what is the probability of failure masking?")
		# TODO

def calculate_and_visualize_limiting_distribution(transitions, visualize):
	limiting_distributions = list(approximate_limiting_distribution_gen(transitions, 100))
	limiting_distribution = limiting_distributions[-1]

	if visualize == 'anim':
		for (i, distribution) in enumerate(tqdm(
			limiting_distributions,
			desc='Rendering limiting distribution',
			total=100)):
			visualize_dtmc(transitions, distribution, max(dist.max() for dist in limiting_distributions), f'./dtmc_frame_{i}')
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
	elif visualize == 'static':
		visualize_dtmc(transitions, limiting_distribution, limiting_distribution.max(), f'./dtmc')

	return limiting_distribution, limiting_distributions

def create_pie_charts(limiting_distribution):
	import matplotlib.pyplot as plt
	limiting_distribution.groupby(level=0).sum().sort_values(ascending=False).plot.pie(
		autopct='%1.1f%%',
		startangle=90,
		counterclock=False,
		pctdistance=0.8,
		labeldistance=1.1,
		rotatelabels=True,
	)
	plt.savefig('./limiting_distribution.png')

	plt.clf()

	limiting_distribution.groupby(level=1).sum().sort_values(ascending=False).plot.pie(
		autopct='%1.1f%%',
		startangle=90,
		counterclock=False,
		pctdistance=0.8, # pctdistance is the distance from the center of the pie to the start of the text generated by autopct
		labeldistance=1.1
	)
	plt.savefig('./limiting_distribution_states.png')

def main(completion_strategy=add_supervisory_component, visualize=None):
	component_transitions = load_transitions('../../../ProjectDescriptions/mRubis_Transition_Matrix.xlsx')

	component_transitions = completion_strategy(component_transitions)
	component_transitions = add_self_loops(component_transitions)

	transitions = generate_transitions(component_transitions)
	logs = list(random_walks(transitions, 100, 100)) # TODO: cache and more
	estimated_transitions = estimate_transition_matrix(logs, transitions.index)

	estimated_transitions = estimated_transitions.fillna(0)

	limiting_distribution, limiting_distributions = calculate_and_visualize_limiting_distribution(estimated_transitions, visualize)
	create_pie_charts(limiting_distribution)

	answer_questions(estimated_transitions, limiting_distributions)

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
# - normalize animation frames
# - answer questions
# - defer for presentation? chris:
#   > Use this discussion of self-loops and supervisory component to solidify your understanding of the Markov Chain properties of being reducible and irreducible, periodic, which relate to being ergodic...
# - do we want to reconsider the analytical solution?

# next meeting: tue 13:00 - 18:00
