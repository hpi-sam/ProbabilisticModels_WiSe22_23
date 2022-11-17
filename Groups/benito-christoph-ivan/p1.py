from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np
import typing


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
			break

		name, trace_pattern = TracePattern.random_applicable(TRACE_PATTERNS, (current_component, current_state))
		trace = trace_pattern.apply((current_component, current_state), transitions)
		yield from trace
		(current_component, current_state) = trace[-1]

def random_walks(transitions, num_walks, num_steps):
	for _ in range(num_walks):
		yield list(random_walk(transitions, num_steps))

def main():
	import pdb; pdb.set_trace()
	transitions = load_transitions()
	logs = list(random_walks(transitions, 20, 100))
	import pdb; pdb.set_trace()

if __name__ == '__main__':
	main()

# NEXT: Learn transition matrix from logs