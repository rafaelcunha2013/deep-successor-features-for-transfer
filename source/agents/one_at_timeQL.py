# -*- coding: UTF-8 -*-
from collections import defaultdict
import numpy as np

from agents.ql import QL
from agents.ql_multiagent import MultiagentQL


class one_at_timeQL(MultiagentQL):
    """
    Creates a new tabular successor feature able to deal with multiagents
    """

    def __init__(self, *args, **kwargs):
        super(one_at_timeQL, self).__init__(*args, **kwargs)
        # self.alpha = learning_rate

    def train_agent(self, s, s_enc, a, r, s1, s1_enc, gamma):
        for i in range(self.n_agents):
            if i > 0:
                s = (s, tuple(a[:i]))
                s1 = (s1, tuple(a[:i]))
            target = r + gamma * np.max(self.Q[i][s1])
            error = target - self.Q[i][s][a[i]]
            self.Q[i][s][a[i]] += self.alpha * error

    def next_sample(self, viewer=None, n_view_ev=None):
        """
        Updates the agent by performing one interaction with the current training environment.
        This function performs all interactions with the environment, data and storage manipulations,
        training the agent, and updating all history.

        Parameters
        ----------
        viewer : object
            a viewer that displays the agent's exploration behavior on the task based on its update() method
            (defaults to None)
        n_view_ev : integer
            how often (in training episodes) to invoke the viewer to display agent's learned behavior
            (defaults to None)
        """

        # start a new episode
        if self.new_episode:
            self.s = self.active_task.initialize()
            self.s_enc = self.encoding(self.s)
            self.new_episode = False
            self.episode += 1
            self.steps_since_last_episode = 0
            self.episode_reward = self.reward_since_last_episode
            self.reward_since_last_episode = 0.
            if viewer is not None:
                viewer.initialize()
            if self.episode > 1:
                self.episode_reward_hist.append(self.episode_reward)
                self.episode_mean_reward_hist.append(np.mean(self.episode_reward_hist[-100:]))

        a = []
        for i in range(self.n_agents):
            if i > 0:
                s = (self.s, tuple(a[:i]))
            else:
                s = self.s
            # compute the Q-values in the current state
            q = self.get_Q_values(s, self.s_enc, i)

            # choose an action using the epsilon-greedy policy
            a.append(self._epsilon_greedy(q))

        a = tuple(a)
        # take action a and observe reward r and next state s'
        # print(self.s)
        # print(a)
        s1, r, terminal = self.active_task.transition(a)
        # viewer.update(s1[0])   # Render the environment
        # print(s1)
        # print(terminal)
        s1_enc = self.encoding(s1)
        if terminal:
            gamma = 0.
            self.new_episode = True
        else:
            gamma = self.gamma

        # train the agent
        self.train_agent(self.s, self.s_enc, a, r, s1, s1_enc, gamma)

        # update counters
        self.s, self.s_enc = s1, s1_enc
        self.steps += 1
        self.reward += r
        self.steps_since_last_episode += 1
        self.reward_since_last_episode += r
        self.cum_reward += r

        if self.steps_since_last_episode >= self.T:
            self.new_episode = True

        if self.steps % self.save_ev == 0:
            self.reward_hist.append(self.reward)
            self.cum_reward_hist.append(self.cum_reward)

        # viewing
        # if viewer is not None and self.episode % n_view_ev == 0 and self.n_tasks > 38:
        if viewer is not None and self.episode % n_view_ev == 0:
            viewer.update(s1[0])

        # printing
        if self.steps % self.print_ev == 0:
            print('\t'.join(self.get_progress_strings()))
