# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pickle

from agents.sfql import SFQL
from agents.sfql_multiagent import MultiagentSFQL
from agents.ql import QL
from agents.ql_multiagent import MultiagentQL
from features.tabular import TabularSF
from features.multi_tabular import multi_TabularSF
# from tasks.gridworld import Shapes
from tasks.gridworld_multagent import MultiagentShapes
from tasks.render import Render
from utils.config import parse_config_file
from utils.stats import OnlineMeanVariance
from utils.tools import save_file
from utils.tools import load_file
from utils.tools import my_plot
from utils.tools import simple_plot

# general training params
config_params = parse_config_file('gridworld.cfg')
gen_params = config_params['GENERAL']
task_params = config_params['TASK']
agent_params = config_params['AGENT']
sfql_params = config_params['SFQL']
ql_params = config_params['QL']

my_reward = [[0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5], [1.0, 0., 0.], [0., 1.0, 0.], [0., 0., 1.0]]


# tasks
def generate_task(this_reward):
    rewards = dict(zip(['1', '2', '3'], list(np.random.uniform(low=-1.0, high=1.0, size=3))))
    # rewards = dict(zip(['1', '2', '3'], this_reward))
    # print('\n \n')
    # print(rewards)
    return MultiagentShapes(maze=np.array(task_params['maze-multi']), shape_rewards=rewards)
    # return MultiagentShapes(maze=np.array(task_params['maze']), shape_rewards=rewards)
 

# agents
# sfql = SFQL(TabularSF(**sfql_params), **agent_params)
# ql = QL(**agent_params, **ql_params)
sfql = MultiagentSFQL(multi_TabularSF(**sfql_params), **agent_params)
ql = MultiagentQL(**agent_params, **ql_params)
agents = [sfql, ql]
names = ['SFQL', 'QLearning']
# agents = [sfql]
# names = ['SFQL']
# agents = [ql]
# names = ['Single-QL']
# names = ['QLearning']
# maze_type = 'maze-multi'

# Visualization of the environment
# my_grid = Render(maze=np.array(task_params['maze-multi']))
# n_view_ev = 10000

# train
data_task_return = [OnlineMeanVariance() for _ in agents]
n_trials = 2  #gen_params['n_trials']
n_samples = 2000000  # gen_params['n_samples']
n_tasks = 2  # gen_params['n_tasks']

NAME = str(names) + '_' + str(my_reward[0]) + str(n_samples)

for trial in range(n_trials):
    
    # train each agent on a set of tasks
    for agent in agents:
        agent.reset()
    for t in range(n_tasks):
        task = generate_task(my_reward[t])
        save_file(data_task_return, NAME)
        # print(task)
        for agent, name in zip(agents, names):
            print('\ntrial {}, solving with {}'.format(trial, name))
            # agent.train_on_task(task, n_samples, viewer=my_grid, n_view_ev=n_view_ev)
            agent.train_on_task(task, n_samples)
             
    # update performance statistics
    for i, agent in enumerate(agents):
        data_task_return[i].update(agent.reward_hist)
        rew_hist = agent.episode_reward_hist
        mean_rew_hist = agent.episode_mean_reward_hist

my_reward_data = [rew_hist, mean_rew_hist]
# Save parameters
# NAME = str(names) + '_' + str(my_reward[0]) + str(n_samples)
save_file(data_task_return, NAME)
save_file(my_reward, NAME + 'rw_per_episode')
my_plot(NAME, names, n_samples, n_tasks)
simple_plot(rew_hist, mean_rew_hist, NAME, 2)
simple_plot(rew_hist[-1000:], mean_rew_hist[-1000:], NAME + 'last1000', 3)


