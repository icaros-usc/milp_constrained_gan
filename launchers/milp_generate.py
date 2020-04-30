import json
import os
import random

from tqdm import tqdm

from algos.milp.program import Program

output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda', 'fake_milp')
os.makedirs(output_path, exist_ok=True)

program = Program()


def vars2grid(si):
    grid = [[1 for _ in range(13)] for _ in range(9)]
    for i in range(9):
        for j in range(13):
            ind = i * 13 + j
            if si.get_value(program.W[ind]) == 1:
                grid[i][j] = 0
            elif si.get_value(program.K[ind]) == 1:
                grid[i][j] = 2
            elif si.get_value(program.G[ind]) == 1:
                grid[i][j] = 3
            elif si.get_value(program.E1[ind]) == 1:
                grid[i][j] = 4
            elif si.get_value(program.E2[ind]) == 1:
                grid[i][j] = 5
            elif si.get_value(program.E3[ind]) == 1:
                grid[i][j] = 6
            elif si.get_value(program.P[ind]) == 1:
                grid[i][j] = 7
    return grid


for i in tqdm(range(10)):
    random_seed = random.randint(0, 1000000000)
    program.set_randomseed(random_seed)
    si = program.solve()
    new_grid = vars2grid(si)
    with open(os.path.join(output_path, '{}.json'.format(i)), 'w') as f:
        f.write(json.dumps(new_grid))
