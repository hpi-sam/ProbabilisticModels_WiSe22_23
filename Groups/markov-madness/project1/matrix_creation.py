import csv

from sympy import Matrix, eye, ones
from sympy.physics.quantum import TensorProduct

reader = csv.reader(open("transitions.csv", "r"))
csv_content = list(reader)
components = [component.lstrip() for component in csv_content[0]]
transition_matrix = csv_content[1:]

component_transitions = Matrix(transition_matrix)  # Initial component transition probabilities
num_components = len(transition_matrix)

states = ("operational", "degrading", "unresponsive")
inner_state_transitions = Matrix([[0.15, 0.05, 0.00],
                                  [0.50, 0.25, 0.10],
                                  [0.25, 0.00, 0.50]])

self_loops = TensorProduct(eye(num_components), inner_state_transitions)

transitions = TensorProduct(eye(len(states)), component_transitions)

failure_masking_identity = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
transitions_failure_masking = TensorProduct(component_transitions.T, failure_masking_identity)

full_transition_matrix = self_loops + transitions + transitions_failure_masking

print(self_loops * ones(self_loops.shape[1], 1))
