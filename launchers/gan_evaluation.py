"""Use this file to evaluate generated levels."""

import json
import os

import numpy as np

from algos.astar import search


def evaluate(lvl):
    """Use this function to evaluate the level"""
    num_player = 0
    num_key = 0
    num_door = 0
    num_enemy1 = 0
    num_enemy2 = 0
    num_enemy3 = 0
    num_empty = 0

    grid = [[0 for _ in range(len(lvl[0]))] for _ in range(len(lvl))]
    heuristic_key = [[0 for _ in range(len(lvl[0]))] for _ in range(len(lvl))]
    heuristic_door = [[0 for _ in range(len(lvl[0]))] for _ in range(len(lvl))]
    goal_key = [0, 0]
    goal_door = [0, 0]
    init = [0, 0]
    for row in range(len(lvl)):
        for col in range(len(lvl[row])):
            if lvl[row][col] == 2:
                num_key += 1
                goal_key[0] = row
                goal_key[1] = col
            elif lvl[row][col] == 3:
                num_door += 1
                goal_door[0] = row
                goal_door[1] = col
            elif lvl[row][col] == 7:
                num_player += 1
                init[0] = row
                init[1] = col
            elif lvl[row][col] == 4:
                num_enemy1 += 1
            elif lvl[row][col] == 5:
                num_enemy2 += 1
            elif lvl[row][col] == 6:
                num_enemy3 += 1
            elif lvl[row][col] == 1:
                num_empty += 1
            elif lvl[row][col] == 0:
                grid[row][col] = 1

    # enemies cover less than 60% of the empty space
    if (num_enemy1 + num_enemy2 + num_enemy3) / (num_enemy1 + num_enemy2 + num_enemy3 + num_empty) > 0.6:
        return False

    # border wall
    np_lvl = np.array(lvl)
    if not (np.sum(np_lvl, axis=-1)[0] == 0 and np.sum(np_lvl, axis=-1)[-1] == 0 and np.sum(np_lvl, axis=0)[0] == 0 and \
            np.sum(np_lvl, axis=0)[-1] == 0):
        return False

    # there must be only one key, one player and one door
    if not (num_key == 1 and num_player == 1 and num_door == 1):
        return False

    # the player should be able to reach the key
    delta = [[-1, 0],
             [0, -1],
             [1, 0],
             [0, 1]]
    cost = 1
    for row in range(len(heuristic_key)):
        for col in range(len(heuristic_key[row])):
            heuristic_key[row][col] = abs(row - goal_key[0]) + abs(col - goal_key[1])
            heuristic_door[row][col] = abs(row - goal_door[0]) + abs(col - goal_door[1])
    result_key, _ = search(grid, init, goal_key, cost, delta, heuristic_key)
    if result_key == 'fail':
        return False

    # the player should be able to reach the door
    result_door, _ = search(grid, init, goal_door, cost, delta, heuristic_door)
    if result_door == 'fail':
        return False

    # the level at least has one enemy
    # if num_enemy1 + num_enemy2 + num_enemy3 == 0:
    #    return False

    return True


def compute_duplicated_lvls(lvl_lst):
    num_duplicated = 0
    unique_lvls = []
    for lvl in lvl_lst:
        if_duplicated = False
        for unique_lvl in unique_lvls:
            if unique_lvl == lvl:
                if_duplicated = True
                break
        if if_duplicated:
            num_duplicated += 1
        else:
            unique_lvls.append(lvl)
    return num_duplicated, unique_lvls


def run():
    dataroot = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda', 'fake')

    num_levels = 0
    num_valid_levels = 0
    all_valid_lvs = []
    for lvl in os.listdir(dataroot):
        num_levels += 1
        with open(os.path.join(dataroot, lvl), 'r') as f:
            lvlJson = json.load(f)
            if_valid = evaluate(lvlJson)

            if if_valid:
                all_valid_lvs.append(lvlJson)
                print(lvl)
                num_valid_levels += 1
    num_duplicated, unique_lvls = compute_duplicated_lvls(all_valid_lvs)

    print('Valid levels: {} / {}'.format(num_valid_levels, num_levels))
    print('Duplicated valid levels: {} / {}'.format(num_duplicated, num_valid_levels))
    print('Unique and playable levles: {} / {}'.format(len(unique_lvls), num_levels))


if __name__ == '__main__':
    run()
