# -*- coding: UTF-8 -*-
from collections import defaultdict
import numpy as np

from agents.ql import QL
from agents.multiagent import multiagent


class MultiagentQL(multiagent, QL):
    """
    Creates a new tabular successor feature able to deal with multiagents
    """

    def __init__(self, *args, **kwargs):
        super(MultiagentQL, self).__init__(*args, **kwargs)
        # self.alpha = learning_rate

    def get_Q_values(self, s, s_enc, agent):
        return self.Q[agent][s]

    def train_agent(self, s, s_enc, a, r, s1, s1_enc, gamma):
        for i in range(self.n_agents):
            target = r + gamma * np.max(self.Q[i][s1])
            error = target - self.Q[i][s][a[i]]
            self.Q[i][s][a[i]] += self.alpha * error

    def set_active_training_task(self, index):
        super(QL, self).set_active_training_task(index)
        self.Q = []
        for _ in range(self.n_agents):
            self.Q.append(defaultdict(lambda: np.random.uniform(low=-0.01, high=0.01, size=(self.n_actions,))))

