# -*- coding: UTF-8 -*-
import numpy as np

from agents.sfql import SFQL
from agents.multiagent import multiagent


class MultiagentSFQL(multiagent, SFQL):
    """
    Creates a new tabular successor feature able to deal with multiagents
    """

    def __init__(self, *args, **kwargs):
        super(MultiagentSFQL, self).__init__(*args, **kwargs)

    def get_Q_values(self, s, s_enc, agent):
        q, self.c = self.sf.GPI(s_enc, self.task_index, agent, update_counters=self.use_gpi)
        if not self.use_gpi:
            self.c = self.task_index
        return q[:, self.c,:]

    def train_agent(self, s, s_enc, a, r, s1, s1_enc, gamma):

        # update w
        t = self.task_index
        # How the method below can have phi as name, but call "features" in gridworld
        # self.phis.append(task.features) --> It appends a method from the task class
        # self.phi = self.phis[index]
        phi = self.phi(s, a, s1)
        self.sf.update_reward(phi, r, t)
        for agent in range(self.n_agents):
            # update SF for the current task t
            if self.use_gpi:
                q1, _ = self.sf.GPI(s1_enc, t, agent)
                q1 = np.max(q1[0, :, :], axis=0)
            else:
                q1 = self.sf.GPE(s1_enc, t, t, agent)[0, :]
            next_action = np.argmax(q1)
            transitions = [(s_enc, a[agent], phi, s1_enc, next_action, gamma)]
            self.sf.update_successor(transitions, t, agent)

            # update SF for source task c
            if self.c != t:
                q1 = self.sf.GPE(s1_enc, self.c, self.c, agent)
                next_action = np.argmax(q1)
                transitions = [(s_enc, a[agent], phi, s1_enc, next_action, gamma)]
                self.sf.update_successor(transitions, self.c, agent)

    def add_training_task(self, task):
        super(MultiagentSFQL, self).add_training_task(task)
        self.sf.add_training_task(task, -1)

    # def set_active_training_task(self, index):
    #     """
    #     Sets the task at the requested index as the current task the agent will train on.
    #     The index is based on the order in which the training task was added to the agent.
    #     """
    #
    #     # set the task
    #     self.task_index = index
    #     self.active_task = self.tasks[index]
    #     self.phi = self.phis[index]
    #
    #     # reset task-dependent counters
    #     self.s = self.s_enc = None
    #     self.new_episode = True
    #     self.episode, self.episode_reward = 0, 0.
    #     self.steps_since_last_episode, self.reward_since_last_episode = 0, 0.
    #     self.steps, self.reward = 0, 0.
    #     self.epsilon = self.epsilon_init
    #     self.episode_reward_hist = []
