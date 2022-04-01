# -*- coding: UTF-8 -*-
from collections import defaultdict
from copy import deepcopy
import numpy as np

from features.successor import SF


class multiSF(SF):
    def __init__(self, *args, **kwargs):
        super(multiSF, self).__init__(*args, **kwargs)

    def add_training_task(self, task, source=None):
        """
        Adds a successor feature representation for the specified task.

        Parameters
        ----------
        task : Task
            a new MDP environment for which to learn successor features
        source : integer
            if specified and not None, the parameters of the successor features for the task at the source
            index should be copied to the new successor features, as suggested in [1]
        """

        # add successor features to the library
        n_agents = task.agent_count()
        if len(self.psi) == 0:
            self.psi = [[], []]
        for i in range(n_agents):
            self.psi[i].append(self.build_successor(task, i, source))

        ##################################
        # self.psi0.append(self.build_successor(task, source, 0))
        # self.psi1.append(self.build_successor(task, source, 1))
        # self.psi = [self.psi0, self.psi1]
        ###################################
        self.n_tasks = len(self.psi[0])

        # build new reward function
        true_w = task.get_w()
        self.true_w.append(true_w)
        if self.use_true_reward:
            fit_w = true_w
        else:
            n_features = task.feature_dim()
            fit_w = np.random.uniform(low=-0.01, high=0.01, size=(n_features, 1))
        self.fit_w.append(fit_w)

        # add statistics
        for i in range(len(self.gpi_counters)):
            self.gpi_counters[i] = np.append(self.gpi_counters[i], 0)
        self.gpi_counters.append(np.zeros((self.n_tasks,), dtype=int))

    def GPE_w(self, state, policy_index, w, agent):
        """
        Implements generalized policy evaluation according to [1]. In summary, this uses the
        learned reward parameters of one task and successor features of a policy to estimate the Q-values of
        the policy if it were executed in that task.

        Parameters
        ----------
        state : object
            a state or collection of states of the MDP
        policy_index : integer
            the index of the task whose policy to evaluate
        w : numpy array
            reward parameters of the task in which to evaluate the policy

        Returns
        -------
        np.ndarray : the estimated Q-values of shape [n_batch, n_actions], where
            n_batch is the number of states in the state argument
            n_actions is the number of actions in the MDP
        """
        psi = self.get_successor(state, policy_index, agent)
        q = psi @ w  # shape (n_batch, n_actions)
        return q

    def GPE(self, state, policy_index, task_index, agent):
        """
        Implements generalized policy evaluation according to [1]. In summary, this uses the
        learned reward parameters of one task and successor features of a policy to estimate the Q-values of
        the policy if it were executed in that task.

        Parameters
        ----------
        state : object
            a state or collection of states of the MDP
        policy_index : integer
            the index of the task whose policy to evaluate
        task_index : integer
            the index of the task (e.g. reward) to use to evaluate the policy

        Returns
        -------
        np.ndarray : the estimated Q-values of shpae [n_batch, n_actions], where
            n_batch is the number of states in the state argument
            n_actions is the number of actions in the MDP
        """
        return self.GPE_w(state, policy_index, self.fit_w[task_index], agent)

    def GPI(self, state, task_index, agent, update_counters=False):
        """
        Implements generalized policy improvement according to [1].

        Parameters
        ----------
        state : object
            a state or collection of states of the MDP
        task_index : integer
            the index of the task in which the GPI action will be used
        update_counters : boolean
            whether or not to keep track of which policies are active in GPI

        Returns
        -------
        np.ndarray : the maximum Q-values computed by GPI for selecting actions
        of shape [n_batch, n_tasks, n_actions], where:
            n_batch is the number of states in the state argument
            n_tasks is the number of tasks
            n_actions is the number of actions in the MDP
        np.ndarray : the tasks that are active in each state of state_batch in GPi
        """
        q, task = self.GPI_w(state, self.fit_w[task_index], agent)
        if update_counters:
            self.gpi_counters[task_index][task] += 1
        return q, task

    def GPI_w(self, state, w, agent):
        """
        Implements generalized policy improvement according to [1].

        Parameters
        ----------
        state : object
            a state or collection of states of the MDP
        w : numpy array
            the reward parameters of the task to control

        Returns
        -------
        np.ndarray : the maximum Q-values computed by GPI for selecting actions
        of shape [n_batch, n_tasks, n_actions], where:
            n_batch is the number of states in the state argument
            n_tasks is the number of tasks
            n_actions is the number of actions in the MDP
        np.ndarray : the tasks that are active in each state of state_batch in GPi
        """
        psi = self.get_successors(state, agent)
        q = (psi @ w)[:, :, :, 0]  # shape (n_batch, n_tasks, n_actions)
        task = np.squeeze(np.argmax(np.max(q, axis=2), axis=1))  # shape (n_batch,)
        return q, task