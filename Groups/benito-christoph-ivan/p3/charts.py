# %%
import p3
# %%
p3.main(['qlearning', '-q'], 'scores_qlearning.csv')
p3.main(['sarsa', '-q'], 'scores_sarsa.csv')
# %%
import numpy as np
import matplotlib.pyplot as plt

def load_scores(filename):
	"""Load scores from a csv file"""
	scores = np.loadtxt(filename, delimiter=',')
	scores = scores[1:]
	return scores

def smooth(scores, window_size=30):
	"""Smooth scores by averaging over a window"""
	smoothed = np.zeros(len(scores))
	for i in range(len(scores)):
		smoothed[i] = np.mean(scores[max(0, i - window_size):i + 1])
	return smoothed

def plot_scores(scores_q, scores_s, title=None):
	plt.plot(scores_q)
	plt.plot(scores_s, color='red')
	plt.legend(['Q-learning', 'Sarsa'])
	plt.xlabel('Episode')
	plt.ylabel(f'Score ({title})' if title else 'Score')
# %%
scores_q = smooth(load_scores('scores_qlearning.csv'))
scores_s = smooth(load_scores('scores_sarsa.csv'))
plot_scores(scores_q, scores_s)

# %%
p3.main(['qlearning', '-q', '--new-obstacle', '--shift'], 'scores_qlearning.csv')
p3.main(['sarsa', '-q', '--new-obstacle', '--shift'], 'scores_sarsa.csv')

# %%
scores_qs = smooth(load_scores('scores_qlearning.csv_shifted'))
scores_ss = smooth(load_scores('scores_sarsa.csv_shifted'))
plot_scores(scores_qs, scores_ss, 'shifted')
# %%
scores_qs = smooth(load_scores('scores_qlearning_shifted.csv'))
scores_ss = smooth(load_scores('scores_sarsa_shifted.csv'))
plot_scores(scores_qs, scores_ss, 'shifted')
# %%
scores_qn = smooth(load_scores('scores_qlearning.csv_newobstacle'))
scores_sn = smooth(load_scores('scores_sarsa.csv_newobstacle'))
plot_scores(scores_qn, scores_sn, 'new obstacle')

# %%
