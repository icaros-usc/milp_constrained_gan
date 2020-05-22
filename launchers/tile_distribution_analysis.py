"""In this file we compute the distribution of the tiles https://arxiv.org/pdf/1910.01603.pdf"""


import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from launchers.hamming_distance_analysis import get_valid_lvls


def compute_tile_number(lvl, tile_type):
    num = 0
    for row in range(len(lvl)):
        for col in range(len(lvl[row])):
            if lvl[row][col] in tile_type:  # there could be multi types
                num += 1
    return num


def do_stats(valid_lvls, tile_type):
    stats = {}
    min_num = 10000
    max_num = 0
    for i, lvl in enumerate(valid_lvls):
        num_tile = compute_tile_number(lvl, tile_type)

        if num_tile not in stats:
            stats[num_tile] = 1
        else:
            stats[num_tile] += 1
        if num_tile > max_num:
            max_num = num_tile
        if num_tile < min_num:
            min_num = num_tile
    langs = list(range(min_num - 3, max_num + 3))
    counts = [0 for lang in langs]
    for num_tile in stats:
        stats[num_tile] /= len(valid_lvls)
        counts[langs.index(num_tile)] = stats[num_tile]
    return langs, counts


def run():
    gan_generated_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda', 'fake')
    milp_gan_two_stage_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda',
                                           'fake_milp_gan_obj')
    human_generated_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda', 'Human_Json')

    # first get all valid levels generated
    all_valid_gan_lvls = get_valid_lvls(gan_generated_root)
    all_two_stage_lvls = get_valid_lvls(milp_gan_two_stage_root)
    all_human_lvls = get_valid_lvls(human_generated_root)

    plot_type = 2  # 1: empty space; 2: wall; 3: enemy

    if plot_type == 1:
        tile_type = [1]
        title = 'Empty Distribution'
        output = 'em_dist.pdf'
    elif plot_type == 2:
        tile_type = [0]
        title = 'Walls Distribution'
        output = 'walls_dist.pdf'
    elif plot_type == 3:
        tile_type = [4, 5, 6]
        title = 'Enemy Distribution'
        output = 'enemy_dist.pdf'
    else:
        raise NotImplementedError('The input type {} is not valid'.format(plot_type))

    gan_langs, gan_counts = do_stats(all_valid_gan_lvls, tile_type)
    two_stage_langs, two_stage_counts = do_stats(all_two_stage_lvls, tile_type)
    human_langs, human_counts = do_stats(all_human_lvls, tile_type)

    left_tick = min(gan_langs[0], two_stage_langs[0], human_langs[0])
    right_tick = max(gan_langs[-1], two_stage_langs[-1], human_langs[-1])

    fig, ax = plt.subplots(figsize=(6, 4.5))
    # ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(gan_langs, gan_counts, color='#ff9f4a', edgecolor='#ff871d', alpha=0.7, width=1, label='GAN')
    ax.bar(human_langs, human_counts, color='#5699c5', edgecolor='#2d7fb8', alpha=0.7, width=1, label='Human')
    ax.bar(two_stage_langs, two_stage_counts, color='#60b761', edgecolor='#3aa539', alpha=0.7, width=1,
           label='GAN+MILP (Two-stage)')
    ax.set_ylabel('Count')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel(title)
    xticks = list(range(left_tick, right_tick, ((right_tick - left_tick) // 6)))[1:-1]
    if plot_type == 3:
        xticks = [0, 2, 4, 6, 8]
    ax.set_xticks(xticks)
    ax.legend()
    fig.tight_layout()
    plt.savefig(output)


if __name__ == '__main__':
    run()
