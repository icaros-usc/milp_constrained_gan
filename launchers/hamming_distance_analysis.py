"""In this file we compute the hamming distance https://arxiv.org/pdf/1910.01603.pdf"""

import json
import os
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from launchers.gan_evaluation import evaluate


def get_valid_lvls(data_root):
    res = []
    for lvl in os.listdir(data_root):
        with open(os.path.join(data_root, lvl), 'r') as f:
            lvlJson = json.load(f)
            if_valid = evaluate(lvlJson)

            if if_valid:
                res.append(lvlJson)
    return res


def compute_tile_difference(lvl, other_lvl):
    diff = 0
    for row in range(len(lvl)):
        for col in range(len(lvl[0])):
            if lvl[row][col] != other_lvl[row][col]:
                diff += 1
    return diff


def compute_hamming_distance(lvl, lvls):
    total_difference = 0

    for other_lvl in lvls:
        total_difference += compute_tile_difference(lvl, other_lvl)

    average_diff = int(total_difference / len(lvls))  # round to integer
    return average_diff


def do_stats(valid_lvls):
    # pre specified the axes
    langs = list(range(15, 44))
    counts = [0 for lang in langs]
    for i, lvl in enumerate(valid_lvls):
        copied_all_valid_lvls = deepcopy(valid_lvls)
        copied_all_valid_lvls.pop(i)
        hamming_dist = compute_hamming_distance(lvl, copied_all_valid_lvls)
        counts[langs.index(hamming_dist)] += 1
    for i, count in enumerate(counts):
        counts[i] /= len(valid_lvls)
    return langs, counts


if __name__ == '__main__':
    gan_generated_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda', 'fake')
    milp_gan_two_stage_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda',
                                           'fake_milp_gan_obj')
    human_generated_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda', 'Human_Json')

    # first get all valid levels generated
    all_valid_gan_lvls = get_valid_lvls(gan_generated_root)
    all_two_stage_lvls = get_valid_lvls(milp_gan_two_stage_root)
    all_human_lvls = get_valid_lvls(human_generated_root)

    gan_langs, gan_counts = do_stats(all_valid_gan_lvls)
    two_stage_langs, two_stage_counts = do_stats(all_two_stage_lvls)
    human_langs, human_counts = do_stats(all_human_lvls)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    # ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(gan_langs, gan_counts, color='#ff9f4a', edgecolor='#ff871d', alpha=0.7, width=1, label='GAN')
    ax.bar(human_langs, human_counts, color='#5699c5', edgecolor='#2d7fb8', alpha=0.7, width=1, label='Human')
    ax.bar(two_stage_langs, two_stage_counts, color='#60b761', edgecolor='#3aa539', alpha=0.7, width=1,
           label='GAN+MILP (Two-stage)')
    ax.set_ylabel('Count')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel('Average Hamming Distance')
    ax.set_xticks([15, 20, 25, 30, 35, 40])
    ax.legend()
    fig.tight_layout()
    plt.savefig('hamming_distance_analysis.pdf')