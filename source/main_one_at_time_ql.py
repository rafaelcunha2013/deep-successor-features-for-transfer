# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pickle

from agents.sfql import SFQL
from agents.sfql_multiagent import MultiagentSFQL
from agents.ql import QL
from agents.ql_multiagent import MultiagentQL
from agents.one_at_timeQL import one_at_timeQL
from features.tabular import TabularSF
from features.multi_tabular import multi_TabularSF
# from tasks.gridworld import Shapes
from tasks.gridworld_multagent import MultiagentShapes
from tasks.render import Render
from utils.config import parse_config_file
from utils.stats import OnlineMeanVariance

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
    # rewards = dict(zip(['1', '2', '3'], list(np.random.uniform(low=-1.0, high=1.0, size=3))))
    rewards = dict(zip(['1', '2', '3'], this_reward))
    print('\n \n')
    print(rewards)
    return MultiagentShapes(maze=np.array(task_params['maze-multi']), shape_rewards=rewards)
    # return MultiagentShapes(maze=np.array(task_params['maze']), shape_rewards=rewards)


def save_file(file, name):
    """
    Save file
    input: file and file_name
    """
    with open('data/' + name, "wb") as fp:  # Pickling
        pickle.dump(file, fp)


def load_file(file_name):
    with open('data/' + file_name, "rb") as fp:  # Unpickling
        return pickle.load(fp)


def my_plot(NAME):
    data_task_return = load_file(NAME)
    # plot the task return
    ticksize = 14
    textsize = 18
    figsize = (20, 10)

    plt.rc('font', size=textsize)  # controls default text sizes
    plt.rc('axes', titlesize=textsize)  # fontsize of the axes title
    plt.rc('axes', labelsize=textsize)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=ticksize)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=ticksize)  # fontsize of the tick labels
    plt.rc('legend', fontsize=ticksize)  # legend fontsize

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    for i, name in enumerate(names):
        mean = data_task_return[i].mean
        n_sample_per_tick = n_samples * n_tasks // mean.size
        x = np.arange(mean.size) * n_sample_per_tick
        se = data_task_return[i].calculate_standard_error()
        plt.plot(x, mean, label=name)
        ax.fill_between(x, mean - se, mean + se, alpha=0.3)
    plt.xlabel('sample')
    plt.ylabel('cumulative reward')
    plt.title('Cumulative Training Reward Per Task')
    plt.tight_layout()
    plt.legend(ncol=2, frameon=False)
    plt.savefig('figures/' + NAME + '.png')


# agents
# sfql = SFQL(TabularSF(**sfql_params), **agent_params)
# ql = QL(**agent_params, **ql_params)
# sfql = MultiagentSFQL(multi_TabularSF(**sfql_params), **agent_params)
ql = one_at_timeQL(**agent_params, **ql_params)
# agents = [sfql, ql]
# names = ['SFQL', 'QLearning']
# agents = [sfql]
# names = ['SFQL']
agents = [ql]
# names = ['Single-QL']
# names = ['QLearning']
names = ['one_at_timeQL']
# maze_type = 'maze-multi'

# Visualization of the environment
my_grid = Render(maze=np.array(task_params['maze-multi']))
n_view_ev = 1000

# train
data_task_return = [OnlineMeanVariance() for _ in agents]
n_trials = 1  # gen_params['n_trials']
n_samples = gen_params['n_samples']
n_tasks = 1  # gen_params['n_tasks']

for trial in range(n_trials):

    # train each agent on a set of tasks
    for agent in agents:
        agent.reset()
    for t in range(n_tasks):
        task = generate_task(my_reward[t])
        # print(task)
        for agent, name in zip(agents, names):
            print('\ntrial {}, solving with {}'.format(trial, name))
            agent.train_on_task(task, n_samples, viewer=my_grid, n_view_ev=n_view_ev)
            #agent.train_on_task(task, n_samples)

    # update performance statistics
    for i, agent in enumerate(agents):
        data_task_return[i].update(agent.reward_hist)
        rew_hist = agent.episode_reward_hist
        mean_rew_hist = agent.episode_mean_reward_hist

plt.plot(rew_hist)
plt.plot(mean_rew_hist)
plt.ylim([-1, 2])
plt.show()

plt.plot(rew_hist[-1000:])
plt.plot(mean_rew_hist[-1000:])
plt.ylim([-1, 2])
plt.show()

# Save parameters
NAME = str(names) + '_' + str(my_reward[0]) + str(n_samples) + 'test01'
save_file(data_task_return, NAME)
my_plot(NAME)

