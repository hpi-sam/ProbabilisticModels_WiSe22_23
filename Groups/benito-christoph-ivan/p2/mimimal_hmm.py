from hmmlearn import hmm
import numpy as np

observations = (
	[[0] * 100]
	+ [[1] * 100] * 2
	+ [[2] * 100]
) * 500
observations = np.array(observations)
observations = observations[np.random.permutation(len(observations))]

model = hmm.CategoricalHMM(n_components=3)

model.fit(np.concatenate(observations).reshape(-1, 1), lengths=[100] * len(observations))

print("startprob:", model.startprob_.round(2), sep='\n')
print("transmat:", model.transmat_.round(2), sep='\n')
print("emissionprob:", model.emissionprob_.round(2), sep='\n')

# QUESTION: What are we doing wrong here? Our goal is to learn the following parameters:
# startprob: [0.25, 0.5, 0.25] (or any permutation of these)
# transmat: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] (or any permutation of these)
# emissionprob: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] (or any permutation of these)
