from hmmlearn import hmm
import numpy as np

np.random.seed(15)

observations = (
	[[0] * 100]
	+ [[1] * 100] * 2
	+ [[2] * 100]
) * 500
observations = np.array(observations)
observations = observations[np.random.permutation(len(observations))]

observations_train = observations[:len(observations) // 2]
observations_test = observations[len(observations) // 2:]

best_score = best_model = None
n_fits = 5
for idx in range(n_fits):
	model = hmm.CategoricalHMM(n_components=3, random_state=idx, verbose=True)

	model.fit(np.concatenate(observations_train).reshape(-1, 1), lengths=[100] * len(observations_train))
	score = model.score(np.concatenate(observations_test).reshape(-1, 1), lengths=[100] * len(observations_test))

	print("Model", idx, "score:", score)
	if best_score is None or score > best_score:
		best_score = score
		best_model = model

model = best_model
print("startprob:", model.startprob_.round(2), sep='\n')
print("transmat:", model.transmat_.round(2), sep='\n')
print("emissionprob:", model.emissionprob_.round(2), sep='\n')

# QUESTION: What are we doing wrong here? Our goal is to learn the following parameters:
# startprob: [0.25, 0.5, 0.25] (or any permutation of these)
# transmat: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] (or any permutation of these)
# emissionprob: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] (or any permutation of these)
# ANSWER: Our error was that we only fitted the model once instead of fitting it multiple times with different random_states and choosing the best model.
