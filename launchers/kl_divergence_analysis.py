"""We want to compute the difference between the learned levels and the human designed levels https://dl.acm.org/doi/pdf/10.1145/3321707.3321781"""

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
    num_sum = 0
    for pattern in count_patterns:
        num_sum += count_patterns[pattern]
    for pattern in count_patterns:
        count_patterns[pattern] = count_patterns[pattern] / num_sum


def compute_kl(dist1, dist2):
    assert len(dist1) == len(dist2)

    kl = 0
    for x in dist1:
        kl += dist1[x] * np.log(dist1[x] / dist2[x])
    return kl


def run():
    # first count all appearred tile patterns in human designed levels
    human_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda', 'Human_Json')
    milp_gan_two_stage_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda',
                                           'fake_milp_gan_obj')
    gan_generated_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda', 'fake')

    human_lvls = get_valid_lvls(human_root)
    two_stage_lvls = get_valid_lvls(milp_gan_two_stage_root)
    valid_gan_lvls = get_valid_lvls(gan_generated_root)

    f_h = 2
    f_w = 2
    all_human_patterns = count_pattern_distribution(human_lvls, f_h, f_w)
    all_two_stage_patterns = count_pattern_distribution(two_stage_lvls, f_h, f_w)
    all_valid_gan_patterns = count_pattern_distribution(valid_gan_lvls, f_h, f_w)

    new_two_stage_patterns = {}
    new_valid_gan_patterns = {}
    for pattern in all_human_patterns:
        if pattern in all_two_stage_patterns:
            new_two_stage_patterns[pattern] = all_two_stage_patterns[pattern]
        else:
            new_two_stage_patterns[pattern] = 1e-5
        if pattern in all_valid_gan_patterns:
            new_valid_gan_patterns[pattern] = all_valid_gan_patterns[pattern]
        else:
            new_valid_gan_patterns[pattern] = 1e-5

    count2dist(new_two_stage_patterns)
    count2dist(new_valid_gan_patterns)
    count2dist(all_human_patterns)

    kl_h_t = compute_kl(all_human_patterns, new_two_stage_patterns)
    kl_h_g = compute_kl(all_human_patterns, new_valid_gan_patterns)

    print('KL divergence between human and two stage: {}'.format(kl_h_t))
    print('KL divergence between human and gan: {}'.format(kl_h_g))


if __name__ == '__main__':
    run()
