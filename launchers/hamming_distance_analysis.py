"""In this file we compute the hamming distance https://arxiv.org/pdf/1910.01603.pdf"""

import json
import os
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from launchers.zelda_duplicated_lvls_evaluation import evaluate


def get_valid_lvls(data_root):
    res = []
    for lvl in os.listdir(data_root):
        with open(os.path.join(data_root, lvl), 'r') as f:
            lvlJson = json.load(f)
            if_valid = evaluate(lvlJson)

            if if_valid:
                res.append(lvlJson)
    return res


def get_valid_and_unique_lvls(data_root):
    res = []
    for lvl in os.listdir(data_root):
        with open(os.path.join(data_root, lvl), 'r') as f:
            lvlJson = json.load(f)
            if_valid = evaluate(lvlJson)

            if if_valid:
                if_dup = False
                for already_lvl in res:
                    if already_lvl == lvlJson:
                        if_dup = True
                        break
                if not if_dup:
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


def run(data_root,
        to_stats):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for key in to_stats:
        cur_path = os.path.join(data_root, to_stats[key]['path'])
        cur_valid_lvls = get_valid_lvls(cur_path)
        cur_langs, cur_counts = do_stats(cur_valid_lvls)

        ax.bar(cur_langs, cur_counts, color=to_stats[key]['color'], edgecolor=to_stats[key]['edgecolor'],
               alpha=0.7, width=1, label=key)

    ax.set_ylabel('Count')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel('Average Hamming Distance')
    ax.set_xticks([15, 20, 25, 30, 35, 40])
    ax.legend()
    fig.tight_layout()
    plt.savefig('hamming_distance_analysis.pdf')


if __name__ == '__main__':
    data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda')
    to_stats = {
        'GAN': {'path': 'zelda_better_gan_no_fix', 'color': '#ff9f4a', 'edgecolor': '#ff871d'},
        'Human': {'path': 'Human_Json', 'color': '#5699c5', 'edgecolor': '#2d7fb8'},
        'GAN+MILP (Two-stage)': {'path': 'zelda_better_gan', 'color': '#60b761', 'edgecolor': '#3aa539'}
    }
    run(data_root,
        to_stats)
