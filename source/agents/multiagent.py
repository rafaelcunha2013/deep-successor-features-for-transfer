# -*- coding: UTF-8 -*-
import random
import numpy as np

from agents.agent import Agent


class multiagent(Agent):

    def __init__(self, *args, **kwargs):
        super(multiagent, self).__init__(*args, **kwargs)

    def _epsilon_greedy(self, q):
        assert q.size == self.n_actions
        # n_agents = len(self.s[0])
        # n_action_per_agent = int(self.n_actions/n_agents)
        # sample from a Bernoulli distribution with parameter epsilon
        if random.random() <= self.epsilon:
            # a = tuple([random.randrange(n_action_per_agent) for _ in range(n_agents)])
            a = random.randrange(self.n_actions)
        else:
            # a = tuple([np.argmax(q[0][i * n_action_per_agent:(i+1)*n_action_per_agent]) for i in range(n_agents)])
            a = np.argmax(q)

        # decrease the exploration gradually
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return a

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
            # compute the Q-values in the current state
            q = self.get_Q_values(self.s, self.s_enc, i)

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
        # if self.steps % self.print_ev == 0:
        #     print('\t'.join(self.get_progress_strings()))

    def add_training_task(self, task):
        super(multiagent, self).add_training_task(task)
        self.n_agents = task.agent_count()

    def train_on_task(self, train_task, n_samples, viewer=None, n_view_ev=None):
        """
        Trains the agent on the current task.

        Parameters
        ----------
        train_task : Task
            the training task instance
        n_samples : integer
            how many samples should be generated and used to train the agent
        viewer : object
            a viewer that displays the agent's exploration behavior on the task based on its update() method
            (defaults to None)
        n_view_ev : integer
            how often (in training episodes) to invoke the viewer to display agent's learned behavior
            (defaults to None)
        """
        self.add_training_task(train_task)
        self.set_active_training_task(self.n_tasks - 1)
        for _ in range(n_samples):
            self.next_sample(viewer, n_view_ev)
