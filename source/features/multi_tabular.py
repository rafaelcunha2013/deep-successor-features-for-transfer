# -*- coding: UTF-8 -*-
from collections import defaultdict
from copy import deepcopy
import numpy as np

from features.tabular import TabularSF
from features.successor_multiagent import multiSF


class multi_TabularSF(multiSF, TabularSF):
    def __init__(self, *args, **kwargs):
        super(multi_TabularSF, self).__init__(*args, **kwargs)

    def build_successor(self, task, agent, source=None):
        if source is None or len(self.psi[agent]) == 0:
        # if source is None or len(self.psi) == 0:
            n_actions = task.action_count()
            n_features = task.feature_dim()
            self.n_agent = task.agent_count()
            return defaultdict(lambda: self.noise_init((n_actions, n_features)))
        else:
            return deepcopy(self.psi[agent][source])

    def get_successors(self, state, agent):
        # print(agent)
        # aux = []
        # for psi in self.psi[agent]:
        #     # if np.shape(psi) != ():
        #     #     psi = psi[0]
        #         # print(1)
        #     aux.append(psi[state])
        # return np.expand_dims(np.array(aux), axis=0)
        return np.expand_dims(np.array([psi[state] for psi in self.psi[agent]]), axis=0)

    def update_successor(self, transitions, policy_index, agent):
        for state, action, phi, next_state, next_action, gamma in transitions:
            psi = self.psi[agent][policy_index]
            # if np.shape(psi) != ():
            #     psi = psi[0]
            targets = phi.flatten() + gamma * psi[next_state][next_action,:]
            errors = targets - psi[state][action,:]
            psi[state][action,:] = psi[state][action,:] + self.alpha * errors

    def get_successor(self, state, policy_index, agent):
        return np.expand_dims(self.psi[agent][policy_index][state], axis=0)
