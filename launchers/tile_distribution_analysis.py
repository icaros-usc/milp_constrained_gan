"""In this file we compute the distribution of the tiles https://arxiv.org/pdf/1910.01603.pdf"""


import json
import os
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt

from launchers.gan_evaluation import evaluate
from launchers.hamming_distance_analysis import get_valid_lvls

gan_generated_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda', 'fake')
milp_gan_two_stage_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda', 'fake_milp_gan_obj')
human_generated_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda', 'Human_Json')

# first get all valid levels generated
all_valid_gan_lvls = get_valid_lvls(gan_generated_root)
all_two_stage_lvls = get_valid_lvls(milp_gan_two_stage_root)
all_human_lvls = get_valid_lvls(human_generated_root)


def compute_tile_number(lvl, tile_type):
    num = 0
    for row in range(len(lvl)):
        for col in range(len(lvl[row])):
            if lvl[row][col] == tile_type:
                num += 1
    return num


def do_stats(valid_lvls, tile_type):
    stats = {}
    for i, lvl in enumerate(valid_lvls):
        copied_all_valid_lvls = deepcopy(valid_lvls)
        copied_all_valid_lvls.pop(i)
        hamming_dist = compute_hamming_distance(lvl, copied_all_valid_lvls)
        if hamming_dist not in stats:
            stats[hamming_dist] = 1
        else:
            stats[hamming_dist] += 1
    for hamming_dist in stats:
        stats[hamming_dist] /= len(valid_lvls)
    return stats


gan_stats = do_stats(all_valid_gan_lvls)
two_stage_stats = do_stats(all_two_stage_lvls)
human_stats = do_stats(all_human_lvls)


def plot(stats, color):
    for key in stats:
        plt.scatter(key, stats[key], c=color)


plot(gan_stats, 'r')
plot(two_stage_stats, 'g')
plot(human_stats, 'b')

plt.show()