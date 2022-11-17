from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np
import typing
from tqdm import tqdm


def load_transitions():
	file_name = '../../ProjectDescriptions/mRubis_Transition_Matrix.xlsx'
	sheet_name = 'prior_transition_matrix'
	num_components = 18

	transitions = pd.read_excel(file_name, sheet_name=sheet_name, skiprows=1, index_col=0)
	transitions = transitions.iloc[:, 0:num_components]

	transitions.columns = transitions.columns.str.strip()
	transitions.index = transitions.index.str.strip()

	return transitions


StateType = str
ComponentType = str
CompoundType = typing.Tuple[ComponentType, StateType]
STATES: typing.List[StateType] = ['operational', 'degraded', 'unresponsive']


@dataclass
class TracePatternBuilder:
	transitions: pd.DataFrame
	compound: CompoundType

	@property
	def component(self):
		return self.compound[0]

	@property
	def state(self):
		return self.compound[1]

	def next(self, component: ComponentType, state: StateType) -> CompoundType:
		self.compound = (component, state)
		return self.compound

	def next_component(self) -> ComponentType:
		return self.next(
			np.random.choice(self.transitions.index, p=self.transitions.loc[self.component]),
			self.state
		)

	def alternate_state(self, states: typing.List[StateType]) -> ComponentType:
		return lambda: self.next(
			self.component,
			states[(states.index(self.state) + 1) % len(states)]
		)

	def next_state(self, state: StateType) -> ComponentType:
		return lambda: self.next(
			self.component,
			state
		)

	def any_pattern(self, patterns: typing.List[TracePattern]) -> typing.List[CompoundType]:
		name, pattern = TracePattern.random_applicable(patterns, self.compound)
		return lambda: [
			self.next(*compound)
			for compound in pattern.apply(self.compound, self.transitions)
		][-1]


@dataclass
class TracePattern:
	applicable_states: typing.Optional[typing.List[StateType]]
	transitions: typing.Callable[[TracePatternBuilder], typing.List[CompoundType]]

	def can_apply(self, compound: CompoundType) -> bool:
		if self.applicable_states is None:
			return True
		return compound[1] in self.applicable_states

	def apply(self, compound: CompoundType, transitions) -> typing.List[CompoundType]:
		return [
			function()
			for function in self.transitions(TracePatternBuilder(transitions, compound))
		]

	@staticmethod
	def random_applicable(trace_patterns: typing.Dict[str, TracePattern], compound: CompoundType) -> typing.Tuple[str, TracePattern]:
		applicable_patterns = [(name, pattern) for (name, pattern) in trace_patterns.items() if pattern.can_apply(compound)]
		return applicable_patterns[np.random.choice(range(len(applicable_patterns)))]


INTERMITTENT_STATES = ['operational', 'degraded']
TRACE_PATTERNS = {
	'normal operation': TracePattern(
		applicable_states={'operational'},
		transitions=lambda pattern: [pattern.next_component]),
	'intermittent failure': TracePattern(
		applicable_states=INTERMITTENT_STATES,
		transitions=lambda pattern: [pattern.alternate_state(INTERMITTENT_STATES)]),
	'systemic degradation': TracePattern(
		applicable_states={'degraded'},
		transitions=lambda pattern: [pattern.next_component]),
	'failure masking': TracePattern(
		applicable_states={'unresponsive'},
		transitions=lambda pattern: [pattern.next_component, pattern.next_state('operational')]),
	'failure cascade': TracePattern(
		applicable_states={'degraded'},
		transitions=lambda pattern: [pattern.any_pattern({
			"1": TracePattern(
				applicable_states=None,
				transitions=lambda pattern: [pattern.next_state('unresponsive'), pattern.next_component]),
			"2": TracePattern(
				applicable_states=None,
				transitions=lambda pattern: [pattern.next_component, pattern.next_state('unresponsive')])})])
}


def random_walk(transitions, num_steps):
	current_component = np.random.choice(transitions.index)
	current_state = np.random.choice(STATES)
	yield (current_component, current_state)
	for _ in range(num_steps):
		if transitions.loc[current_component].sum() == 0:
			# break
			yield (current_component, current_state)
			continue

		name, trace_pattern = TracePattern.random_applicable(TRACE_PATTERNS, (current_component, current_state))
		trace = trace_pattern.apply((current_component, current_state), transitions)
		yield from trace
		(current_component, current_state) = trace[-1]

def random_walks(transitions, num_walks, num_steps):
	for _ in tqdm(range(num_walks), desc='Generating random walks'):
		yield list(random_walk(transitions, num_steps))

def estimate_transition_matrix(transition_logs, components):
	compound_states = [(component, state) for component in components for state in STATES]
	transition_matrix = pd.DataFrame(0, index=compound_states, columns=compound_states)
	for log in tqdm(transition_logs, desc='Estimating transition matrix'):
		for compound, next_compound in zip(log, log[1:]):
			transition_matrix.loc[[compound], [next_compound]] += 1
	transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)
	return transition_matrix

def visualize_dtmc(estimated_transition_matrix, distribution, png_path):
	import graphviz
	dot = graphviz.Digraph()
	for compound in estimated_transition_matrix.index:
		# for the color of the node, use the probability in distribution (1=red, 0=transparent)
		color = f'#{int(255 * distribution[compound]):02x}0000'
		dot.node(str(compound), style='filled', fillcolor=color)
	for compound in estimated_transition_matrix.index:
		for next_compound in estimated_transition_matrix.index:
			probability = estimated_transition_matrix.loc[[compound], [next_compound]].sum().sum()
			if probability > 0:
				dot.edge(str(compound), str(next_compound), label=f"{probability:.2f}")
	dot.render(png_path)
	print(f"Markov chain visualization saved to {png_path}")

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

def main():
	transitions = load_transitions()
	logs = list(random_walks(transitions, 20, 100))
	estimated_transition_matrix = estimate_transition_matrix(logs, transitions.index)

	# histogram of the ten likeliest values in the matrix
	import matplotlib.pyplot as plt
	plt.hist(estimated_transition_matrix.values.flatten(), bins=100)
	plt.savefig('transition_matrix_histogram.png')

	estimated_transition_matrix = estimated_transition_matrix.fillna(0)

	import pdb; pdb.set_trace()
	limiting_distribution = calculate_limiting_distribution(estimated_transition_matrix)

	visualize_dtmc(estimated_transition_matrix, limiting_distribution, 'markov_chain.pdf')

	# count nans in estimated_transition_matrix
	number_of_nans = estimated_transition_matrix.isnull().sum().sum()
	number_of_zeros = (estimated_transition_matrix == 0).sum().sum()
	import pdb; pdb.set_trace()

if __name__ == '__main__':
	main()

# NEXT: Learn transition matrix from logs

# tell me a joke
# once upon a time there was a programmer who wanted to learn about markov chains
# he wrote a script to generate random walks through a markov chain
# he then used the random walks to estimate the transition matrix of the markov chain
# but suddenly, he realized that he had no idea how to calculate the limiting distribution of the markov chain
# so he went to the internet and found a solution

# what is a matrix determinant
# in simple words, it's the area of a parallelogram formed by two vectors
# to explain it to a 5-year-old, you can say that it's the area of a square formed by two vectors
# det M :=

# todos
# - visualize 1-loops
# - why are sums of probabilities not 1?
# - answer questions
# - should we use supervisory component? do we need limiting distributions?
# mo treffen wieder?
