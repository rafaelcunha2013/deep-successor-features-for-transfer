# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pickle


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


def my_plot(NAME, *args):
    data_task_return = load_file(NAME)
    names = args[0]
    n_samples = args[1]
    n_tasks = args[2]
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


def simple_plot(rew_hist, mean_rew_hist, NAME, n_fig):
    # plot the task return
    ticksize = 14
    textsize = 18
    # figsize = (20, 10)

    plt.rc('font', size=textsize)  # controls default text sizes
    plt.rc('axes', titlesize=textsize)  # fontsize of the axes title
    plt.rc('axes', labelsize=textsize)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=ticksize)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=ticksize)  # fontsize of the tick labels
    plt.rc('legend', fontsize=ticksize)  # legend fontsize

    plt.figure(n_fig, figsize=(12, 6))
    # ax = plt.gca()
    plt.plot(rew_hist)
    plt.plot(mean_rew_hist)
    # plt.ylim([-1, 2])

    plt.xlabel('sample')
    plt.ylabel('cumulative reward')
    plt.title(NAME + '_per_episode')
    plt.tight_layout()
    # plt.legend(ncol=2, frameon=False)
    plt.savefig('figures/' + NAME + '_per_episode' + '.png')