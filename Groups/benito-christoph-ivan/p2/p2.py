#!/usr/bin/env python3

# import from sibling directory - IT WORKS, DON'T TOUCH IT
import sys; import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from p1 import random_walks, calculate_and_visualize_limiting_distribution
# ---

from hmmlearn import hmm
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

np.random.seed(42)


def generate_emissions(components, states):
	emissions = pd.DataFrame(0, index=pd.MultiIndex.from_product([states, components]), columns=pd.MultiIndex.from_product([states, components]))

	for component in components:
		for state in states:
			emissions.loc[(state, component), (state, component)] = 1

	# add noise (Gaussian distribution, mean=0, std=42 / n)
	# TODO discuss choice of std
	# DEBUG: disabled noise
	emissions = emissions + np.random.normal(0, 0 / emissions.size, emissions.shape).clip(min=0)
	emissions = emissions.div(emissions.sum(axis=1), axis=0)

	return emissions

def generate_observations(log, emissions):
	for (component, state) in log:
		yield np.random.choice(emissions.columns, p=emissions.loc[(state, component)].values)

def quantize_observations(observations, numeric_emissions):
	# hmmlearn expects the observations to contain numeric values. We don't have numeric values, so we have to convert the observations. We do this by assigning a unique number to each unique observation.
	#numeric_states = list(set(
	#	state
	#	for observation in observations
	#	for state in observation
	#))
	numeric_observations = [
		[numeric_emissions.index(emission) for emission in observation]
		for observation in observations
	]
	#return states, numeric_observations
	return numeric_observations

def unquantize_states(numeric_states, numeric_emissions):
	return [
		numeric_emissions[numeric_state][::-1]
		for numeric_state in numeric_states
	]

def learn_hmm(observations, n_components, n_iter=10, n_attempts=50):
	# swap observations randomly
	observations = np.array(observations)
	observations = observations[np.random.permutation(len(observations))]
	obs_train = observations[:len(observations) // 2]
	obs_test = observations[len(observations) // 2:]
	print(obs_train)

	best_score = -np.inf
	best_model = None

	# To find the best model, we try to fit the model multiple times with different random seeds. We then choose the model with the highest score.
	# Unfortunately, training the model is very slow. Another strategy could be performing a BFS over the space of possible models (by training each model incrementally and ordering the models by their score), but hmmlearn seems not to support this at the moment.

	for attempt in tqdm(range(n_attempts), desc='Learning HMM'):
		model = hmm.CategoricalHMM(n_components=n_components, n_iter=n_iter, random_state=attempt, verbose=True)
		model.fit(np.concatenate(obs_train).reshape(-1, 1), lengths=[len(observation) for observation in obs_train])
		score = model.score(np.concatenate(obs_test).reshape(-1, 1), lengths=[len(observation) for observation in obs_test])

		print(f"Attempt {attempt + 1}/{n_attempts}: score={score}")

		if score > best_score:
			best_score = score
			best_model = model

	return model

argmax = lambda seq: max(seq, key=lambda x: x[1])

def answer_questions(model, transitions, distributions, numeric_emissions):
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
	components = transitions.index.get_level_values(1).unique()

	def predict_states(observation):
		numeric_observations = quantize_observations([observation], numeric_emissions)
		states = model.predict(np.concatenate(numeric_observations).reshape(-1, 1), lengths=[len(observation)])
		return unquantize_states(states, numeric_emissions)

	for n in [10]:
		print(f"After {n} steps:")

		print("  What is the probability of seeing an intermittent failure in component 1 (Bid and Buy Service)?")
		result = sum(
			distributions[i][('operational', 'Bid and Buy Service')] * transitions.loc[('operational', 'Bid and Buy Service'), ('degraded', 'Bid and Buy Service')] * transitions.loc[('degraded', 'Bid and Buy Service'), ('operational', 'Bid and Buy Service')]
			for i in range(0, n + 1 - 2)
		) / (n + 1 - 2)
		print(f"    {result:.5f}") if result > 1e-5 else print(f"    {result}")
		print("  What is the most probable sequence of states for this observation?")
		states = [
			('Bid and Buy Service', 'operational'),
			('Bid and Buy Service', 'degraded'),
			('Bid and Buy Service', 'operational')
		]
		print(f"    {states}")

		print("  What is the probability of seeing an intermittent failure in component 2 (Item Management Service)?")
		result = sum(
			distributions[i][('operational', 'Item Management Service')] * transitions.loc[('operational', 'Item Management Service'), ('degraded', 'Item Management Service')] * transitions.loc[('degraded', 'Item Management Service'), ('operational', 'Item Management Service')]
			for i in range(0, n + 1 - 2)
		) / (n + 1 - 2)
		print(f"    {result:.5f}") if result > 1e-5 else print(f"    {result}")
		print("  What is the most probable sequence of states for this observation?")
		states = [
			('Item Management Service', 'operational'),
			('Item Management Service', 'degraded'),
			('Item Management Service', 'operational')
		]
		print(f"    {states}")

		print("  What is the probability of seeing a failure cascade of the type O1d→O2d→O2u?")
		result = sum(
			distributions[i][('degraded', component_1)] * transitions.loc[('degraded', component_1), ('degraded', component_2)] * transitions.loc[('degraded', component_2), ('unresponsive', component_2)]
			for component_1 in components
			for component_2 in components.difference([component_1])
			for i in range(0, n + 1 - 2)
		) / (n + 1 - 2)
		print(f"    {result:.5f}") if result > 1e-5 else print(f"    {result}")
		print("  What is the most probable instance for this observation?")
		observation, probability = argmax(
			(
				[('degraded', component_1), ('degraded', component_2), ('unresponsive', component_2)],
				sum(
					distributions[i][('degraded', component_1)] * transitions.loc[('degraded', component_1), ('degraded', component_2)] * transitions.loc[('degraded', component_2), ('unresponsive', component_2)]
					for i in range(0, n + 1 - 2)
				)
			)
			for component_1 in components
			for component_2 in components.difference([component_1])
		)
		print(f"    {observation}")
		print(f"    (p={probability:.5f})") if probability > 1e-5 else print(f"    (p={probability})")
		print("  What is the most probable sequence of states for this observation?")
		states = predict_states(observation)
		print(f"    {states}")

		print("  What is the probability of seeing a failure cascade of the type O1d→O1u→O2u?")
		result = sum(
			distributions[i][('degraded', component_1)] * transitions.loc[('degraded', component_1), ('unresponsive', component_1)] * transitions.loc[('unresponsive', component_1), ('unresponsive', component_2)]
			for component_1 in components
			for component_2 in components.difference([component_1])
			for i in range(0, n + 1 - 2)
		) / (n + 1 - 2)
		print(f"    {result:.5f}") if result > 1e-5 else print(f"    {result}")
		print("  What is the most probable instance for this observation?")
		observation, probability = argmax(
			(
				[('degraded', component_1), ('unresponsive', component_1), ('unresponsive', component_2)],
				sum(
					distributions[i][('degraded', component_1)] * transitions.loc[('degraded', component_1), ('unresponsive', component_1)] * transitions.loc[('unresponsive', component_1), ('unresponsive', component_2)]
					for i in range(0, n + 1 - 2)
				)
			)
			for component_1 in components
			for component_2 in components.difference([component_1])
		)
		print(f"    {observation}")
		print(f"    (p={probability:.5f})") if probability > 1e-5 else print(f"    (p={probability})")
		print("  What is the most probable sequence of states for this observation?")
		states = predict_states(observation)
		print(f"    {states}")

		print("  What is the probability of seeing failure masking of the type O1u→O2u→O1o?")
		result = sum(
			distributions[i][('unresponsive', component_1)] * transitions.loc[('unresponsive', component_1), ('unresponsive', component_2)] * transitions.loc[('unresponsive', component_2), ('operational', component_1)]
			for component_1 in components
			for component_2 in components.difference([component_1])
			for i in range(0, n + 1 - 2)
		) / (n + 1 - 2)
		print(f"    {result:.5f}") if result > 1e-5 else print(f"    {result}")
		print("  What is the most probable instance for this observation?")
		observation, probability = argmax(
			(
				[('unresponsive', component_1), ('unresponsive', component_2), ('operational', component_1)],
				sum(
					distributions[i][('unresponsive', component_1)] * transitions.loc[('unresponsive', component_1), ('unresponsive', component_2)] * transitions.loc[('unresponsive', component_2), ('operational', component_1)]
					for i in range(0, n + 1 - 2)
				)
			)
			for component_1 in components
			for component_2 in components.difference([component_1])
		)
		print(f"    {observation}")
		print(f"    (p={probability:.5f})") if probability > 1e-5 else print(f"    (p={probability})")
		print("  What is the most probable sequence of states for this observation?")
		states = predict_states(observation)
		print(f"    {states}")

def main(logs_cache_flags=['load', 'store'], observations_cache_flags=['load', 'store']):
	print("Loading transitions from cache...", end='')
	with open('./original_transitions.pickle', 'rb') as f:
		transitions = pickle.load(f)
	print('done')

	emissions = generate_emissions(*transitions.index.levels)

	logs = None
	if 'load' in logs_cache_flags:
		print("Loading logs from cache...", end='')
		try:
			with open('./logs.pickle', 'rb') as f:
				logs = pickle.load(f)
			print('done')
		except FileNotFoundError:
			print("no cache found")
	if logs is None:
		logs = list(random_walks(transitions, 1000, 1000))
	if 'store' in logs_cache_flags:
		print("Storing logs in cache...", end='')
		with open('./logs.pickle', 'wb') as f:
			pickle.dump(logs, f)
		print("done")

	observations = None
	if 'load' in observations_cache_flags:
		print("Loading observations from cache...", end='')
		try:
			with open('./observations.pickle', 'rb') as f:
				observations = pickle.load(f)
			print('done')
		except FileNotFoundError:
			print("no cache found")
	if observations is None:
		observations = [list(generate_observations(log, emissions)) for log in tqdm(logs, desc="Generating observations")]
	if 'store' in observations_cache_flags:
		print("Storing observations in cache...", end='')
		with open('./observations.pickle', 'wb') as f:
			pickle.dump(observations, f)
		print("done")
	numeric_emissions = list(emissions.index.values)
	numeric_observations = quantize_observations(observations, numeric_emissions)

	estimated_model = learn_hmm(numeric_observations, n_components=transitions.shape[0])

	# DEBUGging ...
	#predicted_numeric_states = list(predict_states(estimated_model, numeric_observations))
	##numeric_states = list(predict_states(estimated_model, observations))
	#import pdb; pdb.set_trace()
	#predicted_states = unquantize_states(predicted_numeric_states, numeric_states)

	estimated_transitions = estimated_model.transmat_
	estimated_transitions = pd.DataFrame(estimated_transitions, index=emissions.index, columns=emissions.columns)
	if False:  # DEBUG with fake transitions
		estimated_transitions = transitions.swaplevel(0, 1).swaplevel(0, 1, axis=1)
	_, limiting_distributions = calculate_and_visualize_limiting_distribution(estimated_transitions, None)
	answer_questions(estimated_model, estimated_transitions, limiting_distributions, numeric_emissions)

if __name__ == '__main__':
	import pdb; pdb.set_trace()
	main()

# next meeting: ???

# TODOs:
# - model quality is terrible - not usable
#   symptoms: 1) sample observations are not valid according to original data, 2) answered questions have tiny probabilities, 3) predict_states outputs very unplausible sequences
#   Are we making any systematic error? Are the data/num_iter/num_attempts too small? Does the resulting score support the low model quality?
# - resolve DEBUG flags
# - we should sum all probabilities in answer_questions()
