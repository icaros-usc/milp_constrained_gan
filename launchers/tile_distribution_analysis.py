"""In this file we compute the distribution of the tiles https://arxiv.org/pdf/1910.01603.pdf"""


import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from launchers.hamming_distance_analysis import get_valid_lvls, get_valid_and_unique_lvls


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


def run(data_root,
        human_root,
        to_stats):
    tile_types = [[1], [0], [4, 5, 6]]
    plot_titles = ['Empty Distribution', 'Walls Distribution', 'Enemy Distribution']
    plot_outputs = ['em_dist.pdf', 'walls_dist.pdf', 'enemy_dist.pdf']
    for tile_type, plot_title, plot_output in zip(tile_types, plot_titles, plot_outputs):
        human_data_path = os.path.join(data_root, human_root)
        human_valid_lvls = get_valid_lvls(human_data_path)
        human_langs, human_counts = do_stats(human_valid_lvls, tile_type)

        left_tick = 1000
        right_tick = 0
        for key in to_stats:
            cur_data_path = os.path.join(data_root, to_stats[key]['path'])
            cur_valid_lvls = get_valid_lvls(cur_data_path)
            cur_langs, cur_counts = do_stats(cur_valid_lvls, tile_type)
            if cur_langs[-1] > right_tick:
                right_tick = cur_langs[-1]
            if cur_langs[0] < left_tick:
                left_tick = cur_langs[0]
            to_stats[key]['langs'] = cur_langs
            to_stats[key]['counts'] = cur_counts

        # now visualize
        fig, ax = plt.subplots(figsize=(6, 4.5))
        # first visualize human data
        ax.bar(human_langs, human_counts, color='#5699c5', edgecolor='#2d7fb8', alpha=0.7, width=1, label='Human')
        # then visualize each to plot
        for key in to_stats:
            ax.bar(to_stats[key]['langs'], to_stats[key]['counts'],
                   color=to_stats[key]['color'], edgecolor=to_stats[key]['edgecolor'],
                   alpha=0.7, width=1, label=key)
        ax.set_ylabel('Count')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_xlabel(plot_title)
        xticks = list(range(left_tick, right_tick, ((right_tick - left_tick) // 6)))[1:-1]
        if 4 in tile_type:  # plotting enemies
            xticks = [0, 2, 4, 6, 8]
        ax.set_xticks(xticks)
        ax.legend()
        fig.tight_layout()
        plt.savefig(plot_output)


if __name__ == '__main__':
    data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda')
    human_root = 'Human_Json'
    to_stats = {'GAN': {'path': 'zelda_better_gan_no_fix', 'color': '#ff9f4a', 'edgecolor': '#ff871d'},
                'GAN + MILP (Two-Stage)': {'path': 'zelda_better_gan', 'color': '#60b761', 'edgecolor': '#3aa539'},
                'GAN + MILP (End-to-End)': {'path': 'zelda_better_end2end_3299', 'color': '#d5a49b', 'edgecolor': '#d8a99f'}}

    run(data_root,
        human_root,
        to_stats)
