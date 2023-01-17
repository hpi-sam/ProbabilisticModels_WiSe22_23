"""
This is a simple example of how to Reproduce the famous Sutton&Barton cliff maze.

Charts can be found in charts.py.

Insights:
- If epsilon-decay is disabled, Sarsa will always take the safe path as compared to Q-learning which is more optimistic and will take the optimal cliff path, so Sarsa will converge to a higher score.
- If epsilon-decay is enabled, both algorithms will take the cliff path and converge with an equal speed to the optimal score.
- It must be noted that these algorithms do not support transfer learning. They will relearn the optimal policy for each cell in the grid individually, even though there are only three different policies in the optimal solution. This is a great example of inefficient brute-force ML solutions. If all unknown cells were initialized with the same policy as their neighbors/average cells, the algorithms would converge much faster.
- If after deploying the agent, the cliff is moved to the upper right corner, both algorithms will adapt with the same speed by average. If after deploying the agent, a new obstacle is added to the grid, Sarsa will initially adapt slightly faster than Q-learning, but will be less stable than Q-learning in the long run. The latter can only partially be explained by the fact that optimistic Q-learning already has explored the entire grid more thoroughly than Sarsa, so it will move more souverainly through the new cells around the obstacle. In general, if no fine-tuning or decaying for the greediness of the algorithm is possible, Q-learning should be chosen preferably in an environment that is subjected to possible adversarial laws or concept shifts. TODO find sound explanation for this. is this just a matter of choosing the right hyperparameter for epsilon?
"""

from collections import defaultdict
from contextlib import contextmanager
import gymnasium as gym
import numpy as np
import os
import pandas as pd
import sys
from tqdm import tqdm


def forever():
    i = 0
    while True:
        yield i
        i += 1


class Cliff(gym.Env):
    width = 12
    height = 4
    goal = (width - 1, height - 1)
    cliff = ((1, height - 1), (width - 2, height - 1))

    action_space = gym.spaces.Discrete(4)

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.state = (0, self.height - 1)
        return self.state

    def step(self, action):
        next_state = list(self.state)
        if action == 0:  # left
            next_state[0] -= 1
        elif action == 1:  # down
            next_state[1] += 1
        elif action == 2:  # right
            next_state[0] += 1
        elif action == 3:  # up
            next_state[1] -= 1

        # keep the agent in the grid
        if next_state[0] < 0:
            next_state[0] = 0
        if next_state[0] >= self.width:
            next_state[0] = self.width - 1
        if next_state[1] < 0:
            next_state[1] = 0
        if next_state[1] >= self.height:
            next_state[1] = self.height - 1

        # compute rewards
        done = False
        reward = 0
        if self.is_goal(next_state):
            done = True
        elif self.is_cliff(next_state):
            reward = -100
            next_state = (0, self.height - 1)
        else:
            reward = -1

        self.state = tuple(next_state)
        return self.state, reward, done, {}

    def render(self):
        print("Cliff Maze")
        print("  " + "".join(str(x).ljust(2) for x in range(0, self.width, 2)))
        for y in range(self.height):
            line = ""
            for x in range(self.width):
                if (x, y) == self.state:
                    line += "X"
                elif self.is_cliff((x, y)):
                    line += "█"
                else:
                    line += "░"
            print(y, line, sep=" ")

    def is_cliff(self, state):
        return all(self.cliff[0][d] <= state[d] <= self.cliff[1][d] for d in range(len(self.cliff[0])))

    def is_goal(self, state):
        return tuple(state) == self.goal


class Agent:
    def __init__(self, env):
        self.env = env

    def reset(self):
        self.env.reset()

    def copy(self):
        return self.__class__(self.env)


class UserAgent(Agent):
    def get_action(self, state):
        action = input("Action (wasd, r[eset], q[uit], d[ebug]): ")
        if action == "debug":
            import pdb; pdb.set_trace()
            return self.get_action(state)
        elif action in ("reset", "r"):
            self.reset()
            return None
        elif action in ("quit", "q"):
            return sys.exit(0)
        elif action[0].isalpha():
            action = "asdw".index(action[0])
        else:
            action = int(action)
        return action

    def update(self, state, action, reward, next_state):
        pass

    def reset(self):
        super().reset()
        print()
        print("🚀 New game!")
        print()


class LearningAgent(Agent):
    """Inspired by https://gymnasium.farama.org/tutorials/blackjack_tutorial"""

    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, epsilon=0.1, epsilon_decay=0.999):
        super().__init__(env)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

    def copy(self):
        copy = self.__class__(self.env, self.learning_rate, self.discount_factor, self.epsilon, self.epsilon_decay)
        copy.q_values = self.q_values.copy()
        return copy

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # explore
            return self.env.action_space.sample()
        else:
            # exploit
            return self.q_values[state].argmax()

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay


class QLearningAgent(LearningAgent):
    def update(self, state, action, reward, next_state):
        next_q_value = self.q_values[next_state].max()
        temporal_difference = reward + self.discount_factor * next_q_value - self.q_values[state][action]
        self.q_values[state][action] += self.learning_rate * temporal_difference

        self.decay_epsilon()


class SarsaAgent(LearningAgent):
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, epsilon=0.1, epsilon_decay=0.999):
        super().__init__(env, learning_rate, discount_factor, epsilon, epsilon_decay)
        self.next_action = None

    def copy(self):
        copy = super().copy()
        copy.next_action = self.next_action
        return copy

    def reset(self):
        super().reset()
        self.next_action = self._get_action(self.env.state)

    def get_action(self, state):
        return self.next_action

    def _get_action(self, state):
        return super().get_action(state)

    def update(self, state, action, reward, next_state):
        self.next_action = self._get_action(next_state)
        next_q_value = self.q_values[next_state][self.next_action]
        temporal_difference = reward + self.discount_factor * next_q_value - self.q_values[state][action]
        self.q_values[state][action] += self.learning_rate * temporal_difference

        self.decay_epsilon()


class Game:
    def __init__(self, env, agent, display_flags=[]):
        self.env = env
        self.agent = agent
        self.display_flags = display_flags
        self.scores = []

    score = 0

    def play(self):
        self.score = 0
        self.agent.reset()
        self.state = self.env.state
        scores = [self.score]

        if 'out' in self.display_flags:
            print("Episode", len(self.scores) + 1)
            self.render()

        done = False
        while not done:
            action = self.agent.get_action(self.state)
            if action is None:
                if 'in' in self.display_flags:
                    print("break")
                break

            if 'in' in self.display_flags:
                print("Action:", action, f"({'◀🔽▶🔼'[action]})")

            next_state, reward, done, _ = self.env.step(action)
            self.agent.update(self.state, action, reward, next_state)
            self.score += reward
            self.state = next_state
            scores.append(self.score)

            if 'out' in self.display_flags:
                print("#", self.state, reward, done)
                self.render()

        self.scores.append(scores)
        if 'out' in self.display_flags:
            if done:
                print("🎉 Game over! 🎉")
            print()

    @contextmanager
    def freeze(self):
        old_agent = self.agent
        dummy_agent = self.agent.copy()
        self.agent = dummy_agent
        try:
            yield
        finally:
            self.agent = old_agent
            self.scores.pop()

    def play_nongreedy(self):
        with self.freeze():
            self.agent.epsilon = 0
            self.play()

    def render(self):
        self.env.render()
        print("Score:", self.score)
        print()


def train(game, args):
    scores = []
    for episode in (bar := tqdm(range(20000)) if args[0] != "user" else forever()):
        game.play()

        if episode % 1 == 0:
            game.play_nongreedy()
            scores.append(game.score)
        if bar is not None and episode % 10 == 0:
            bar.set_description(f"Score: {np.mean(scores[-10:])}")

    if args[0] != "user" and all(x not in args for x in ('-q', '--quiet')):
        print()
        print()
        print("🎉 Play without exploration! 🎉")
        game.play_nongreedy()

    return scores


def output_scores(scores, scores_file_or_fd):
    if scores_file_or_fd is None:
        return
    if isinstance(scores_file_or_fd, int):
        try:
            f = os.fdopen(scores_file_or_fd, 'w')
        except OSError:
            return
    else:
        f = open(scores_file_or_fd, 'w')

    with f:
        df = pd.DataFrame(scores)
        df.to_csv(f, index=False, header=False)


def main(args, scores_file_or_fd=3):
    if len(args) < 1:
        print("Usage: python p3.py [ user | qlearning | sarsa ]")
        sys.exit(1)

    env = Cliff()

    if args[0] == "user":
        agent = UserAgent(env)
    elif args[0] == "qlearning":
        agent = QLearningAgent(env, epsilon=0.1, learning_rate=0.01, discount_factor=0.95)
    elif args[0] == "sarsa":
        agent = SarsaAgent(env, epsilon=0.1, learning_rate=0.01, discount_factor=0.95)
    else:
        print("Unknown algorithm:", args[0])
        sys.exit(1)

    agent.epsilon_decay = 1
    if all(x not in args for x in ('-q', '--quiet')):
        display_flags = ['out']
        if args[0] != "user" or any(x in args for x in ('-v', '--verbose')):
            display_flags += ['in']
    else:
        display_flags = []

    game = Game(env, agent, display_flags)
    scores = train(game, args)

    output_scores(scores, scores_file_or_fd)

    if any(x in args for x in ('--new-obstacle',)):
        print("😈 New obstacle! 😈")
        old_is_cliff = env.is_cliff
        env.is_cliff = lambda state: old_is_cliff(state) or tuple(state) == (env.width - 3, env.height - 2) or tuple(state) == (env.width - 3, env.height - 3)
        with game.freeze():
            scores = train(game, args)
        output_scores(scores, scores_file_or_fd + '_newobstacle' if isinstance(scores_file_or_fd, str) else scores_file_or_fd + 1)
        env.is_cliff = old_is_cliff

    if any(x in args for x in ('--shift',)):
        print("😈 Shift the goal! 😈")
        old_goal = env.goal
        env.goal = (env.width - 1, 0)
        with game.freeze():
            scores = train(game, args)
        output_scores(scores, scores_file_or_fd + '_shifted' if isinstance(scores_file_or_fd, str) else scores_file_or_fd + 2)
        env.goal = old_goal


if __name__ == "__main__":
    main(sys.argv[1:])


# TODOs:
# - why does convergence look differently than on the slides? play again with hyperparameters
# - improve discussion of insights, see TODO above
