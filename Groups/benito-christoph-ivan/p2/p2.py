#!/usr/bin/env python3

# IT WORKS, DON'T TOUCH IT
import sys; import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from p1 import random_walks
# ---

from hmmlearn import hmm
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm


def generate_emissions(components, states):
	emissions = pd.DataFrame(0, index=pd.MultiIndex.from_product([states, components]), columns=pd.MultiIndex.from_product([states, components]))

	for component in components:
		for state in states:
			emissions.loc[(state, component), (state, component)] = 1

	# add noise (Gaussian distribution, mean=0, std=42 / n)
	# TODO discuss choice of std
	emissions = emissions + np.random.normal(0, 42 / emissions.size, emissions.shape).clip(min=0)
	emissions = emissions.div(emissions.sum(axis=1), axis=0)

	return emissions

def generate_observations(log, emissions):
	for (component, state) in log:
		yield np.random.choice(emissions.columns, p=emissions.loc[(state, component)].values)

def quantize_observations(observations):
	# hmmlearn expects the observations to contain numeric values because it uses a Gaussian distribution
	# to model the emissions. We don't have numeric values, so we have to convert the observations to
	# numeric values. We do this by assigning a unique number to each unique observation.
	states = list(set(
		state
		for observation in observations
		for state in observation
	))
	numeric_observations = [
		[[states.index(state)] for state in observation]
		for observation in observations
	]
	return states, numeric_observations

def unquantize_states(states, numeric_states):
	return [
		[numeric_states[state[0]] for state in observation]
		for observation in states
	]

def learn_hmm(observations, n_components, n_iter=1000):
	model = hmm.MultinomialHMM(n_components=n_components, n_iter=n_iter, verbose=True)

	concatenated_observations = np.concatenate(observations)
	model.fit(concatenated_observations, lengths=[len(observation) for observation in observations])

	return model

def predict_states(model, observations):
	concatenated_observations = np.concatenate(observations)
	return model.predict(concatenated_observations, lengths=[len(observation) for observation in observations])

def main(logs_cache_flags=['load', 'store'], observations_cache_flags=['load', 'store']):
	print("Loading transitions from cache...", end='')
	try:
		with open('./transitions.pickle', 'rb') as f:
			transitions = pickle.load(f)
		print('done')
	except FileNotFoundError:
		print("no cache found")

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
		logs = list(random_walks(transitions, 1000, 100))
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
	numeric_states, numeric_observations = quantize_observations(observations)

	estimated_model = learn_hmm(numeric_observations, n_components=transitions.shape[0])

	predicted_numeric_states = list(predict_states(estimated_model, numeric_observations))
	import pdb; pdb.set_trace()
	predicted_states = unquantize_states(predicted_numeric_states, numeric_states)

if __name__ == '__main__':
	import pdb; pdb.set_trace()
	main()

# next meeting: thu 9.30 and 17.30
# ct: ask questions about multinomial & terminology
