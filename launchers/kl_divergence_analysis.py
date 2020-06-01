"""We want to compute the difference between the learned levels and the human designed levels https://dl.acm.org/doi/pdf/10.1145/3321707.3321781"""

import copy
import os

import numpy as np

from launchers.hamming_distance_analysis import get_valid_lvls


def count_pattern_distribution_in_lvl(np_lvl, f_h, f_w, all_patterns):
    for i in range(np_lvl.shape[0] - f_h + 1):
        for j in range(np_lvl.shape[1] - f_w + 1):
            pattern = np_lvl[i:i + f_h, j:j + f_w]
            flat_pattern = pattern.flatten()
            str_flat_pattern = np.array2string(flat_pattern)
            if str_flat_pattern not in all_patterns:
                all_patterns[str_flat_pattern] = 1
            else:
                all_patterns[str_flat_pattern] += 1


def count_pattern_distribution(lvls, f_h, f_w):
    all_patterns = {}
    for lvl in lvls:
        np_lvl = np.array(lvl)
        count_pattern_distribution_in_lvl(np_lvl, f_h, f_w, all_patterns)
    return all_patterns


def count2dist(count_patterns):
    count_dist = copy.deepcopy(count_patterns)
    num_sum = 0
    for pattern in count_dist:
        num_sum += count_dist[pattern]
    for pattern in count_dist:
        count_dist[pattern] = count_dist[pattern] / num_sum
    return count_dist


def compute_kl(dist1, dist2):
    assert len(dist1) == len(dist2)

    kl = 0
    for x in dist1:
        kl += dist1[x] * np.log(dist1[x] / dist2[x])
    return kl


def run(pattern_h,
        pattern_w,
        data_root,
        to_stats,
        human_root):
    # first count all appearred tile patterns in human designed levels
    human_data_path = os.path.join(data_root, human_root)
    human_valid_lvls = get_valid_lvls(human_data_path)
    human_patterns = count_pattern_distribution(human_valid_lvls, pattern_h, pattern_w)
    human_pattern_dist = count2dist(human_patterns)

    for key in to_stats:
        cur_data_path = os.path.join(data_root, to_stats[key][0])
        cur_valid_lvls = get_valid_lvls(cur_data_path)
        cur_patterns = count_pattern_distribution(cur_valid_lvls, pattern_h, pattern_w)
        cur_human_like_patterns = {}
        for pattern in human_patterns:
            if pattern in cur_patterns:
                cur_human_like_patterns[pattern] = cur_patterns[pattern]
            else:
                cur_human_like_patterns[pattern] = 1e-5
        cur_pattern_dist = count2dist(cur_human_like_patterns)
        kl = compute_kl(human_pattern_dist, cur_pattern_dist)
        print('{}: {}'.format(key, kl))


if __name__ == '__main__':
    pattern_h = 2
    pattern_w = 2
    data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda')

    to_stats = {'two-stage': ['zelda_better_gan_3299'], 'plain gan': ['zelda_better_gan_no_fix'],
                'end-2-end_fix': ['zelda_better_end2end_3299'], 'end-2-end': ['zelda_better_end2end_3299_no_fix']}
    human_root = 'Human_Json'
    run(pattern_h,
        pattern_w,
        data_root,
        to_stats,
        human_root)
