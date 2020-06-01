"""We want to see if we can design some objective function to use gan's output as a guide for milp solver"""

import json
import os
import random
import time

import numpy as np
import torch
import tqdm

import algos.torch.dcgan.dcgan as dcgan

from algos.milp.zelda.milp_utils import fix_zelda_level
from utils.TrainLevelHelper import get_lvls, get_integer_lvl
from launchers.hamming_distance_analysis import get_valid_lvls


def compute_tile_number(lvl, tile_type):
    num = 0
    for row in range(len(lvl)):
        for col in range(len(lvl[row])):
            if lvl[row][col] == tile_type:  # there could be multi types
                num += 1
    return num


def do_stats(valid_lvls):
    stats = np.zeros([8])
    for i, lvl in enumerate(valid_lvls):
        for tile_type in range(0, 8):
            num_tile = compute_tile_number(lvl, tile_type)
            stats[tile_type] += num_tile
    return stats


def gan_output_to_txt(input):
    grid = [[]]
    for i in range(9):
        for j in range(13):
            if input[i][j] == 0:
                grid[i].append('w')
            elif input[i][j] == 1:
                grid[i].append('.')
            elif input[i][j] == 2:
                grid[i].append('+')
            elif input[i][j] == 3:
                grid[i].append('g')
            elif input[i][j] == 4:
                grid[i].append('1')
            elif input[i][j] == 5:
                grid[i].append('2')
            elif input[i][j] == 6:
                grid[i].append('3')
            elif input[i][j] == 7:
                grid[i].append('A')
            else:
                raise NotImplementedError('unknown character')
        if i < 8:
            grid.append([])

    return grid


def run(data_root,
        human_root):
    human_data_path = os.path.join(data_root, human_root)

    index2strJson = json.load(open('zelda_index2str.json', 'r'))
    str2index = {}
    for key in index2strJson:
        str2index[index2strJson[key]] = key

    seed = 999
    random.seed(seed)
    torch.manual_seed(seed)

    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda', 'zelda_mip_rand_obj')
    os.makedirs(output_path, exist_ok=True)

    human_lvls = get_valid_lvls(human_data_path)
    human_counts = do_stats(human_lvls)
    human_counts = human_counts / sum(human_counts)

    num_iter = 1000

    for i in tqdm.tqdm(range(num_iter)):
        im = np.zeros([9, 13])
        for ii in range(0, 9):
            for jj in range(0, 13):
                im[ii, jj] = np.argmax(np.random.multinomial(n=1, pvals=human_counts))

        level = gan_output_to_txt(im)

        new_level = fix_zelda_level(level)

        numpyLvl = get_integer_lvl(new_level[0], str2index)
        with open(os.path.join(output_path, '{}.json'.format(i)), 'w') as f:
            f.write(json.dumps(numpyLvl.tolist()))


if __name__ == '__main__':
    data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda')
    human_root = 'Human_Json'
    run(data_root,
        human_root)
