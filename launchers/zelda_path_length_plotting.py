"""In this file we compute the hamming distance https://arxiv.org/pdf/1910.01603.pdf"""

import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from launchers.hamming_distance_analysis import get_valid_lvls
from launchers.zelda_duplicated_lvls_evaluation import evaluate


import numpy as np 


def run(data_root,
        to_stats):
    left_stick = 1000
    right_stick = 0
    for key in to_stats:
        cur_valid_lvls = get_valid_lvls(os.path.join(data_root, to_stats[key]['folder']))
        costs = []
        for lvl in cur_valid_lvls:
            _, cost = evaluate(lvl)
            costs.append(cost)
            if cost > right_stick:
                right_stick = cost
            if cost < left_stick:
                left_stick = cost
        costs = np.array(costs).astype(int)
        to_stats[key]['costs'] = costs
    binwidth = 1
    bins = np.arange(left_stick, right_stick + binwidth, binwidth)

    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=20)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for key in to_stats:
        plt.hist(to_stats[key]['costs'], color=to_stats[key]['color'], edgecolor=to_stats[key]['edgecolor'],
                 alpha=0.7, label=key, density=True, bins=bins)

    ax.set_ylabel('Count', fontsize=20)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel('Path Length from Key to Door', fontsize=20)
    ax.legend(loc='best', frameon=False, fontsize=16, ncol=1)

    fig.tight_layout()
    plt.savefig('average_cost_analysis.pdf')



if __name__ == '__main__':
    data_root = dataroot = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda')
    to_stats = {'Human': {'color': '#5699c5', 'edgecolor': '#2d7fb8', 'folder': 'Human_Json'},
                'GAN': {'color': '#c88576', 'edgecolor': '#a26657', 'folder': 'zelda_better_gan_no_fix'},
                'GAN+MIP': {'color': '#60b761', 'edgecolor': '#3aa539', 'folder': 'zelda_better_gan'}}
    run(data_root,
        to_stats)
