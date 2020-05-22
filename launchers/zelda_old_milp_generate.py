import json
import os
import random

from tqdm import tqdm

from algos.milp.zelda.program import Program
from launchers.generate_utils import zelda_milp_vars2grid


def run(output_path,
        seed,
        num_gen):
    os.makedirs(output_path, exist_ok=True)
    program = Program()

    for i in tqdm(range(num_gen)):
        program.set_randomseed(seed)
        si = program.solve()
        new_grid = zelda_milp_vars2grid(si, program)
        with open(os.path.join(output_path, '{}.json'.format(i)), 'w') as f:
            f.write(json.dumps(new_grid))


if __name__ == '__main__':
    seed = random.randint(0, 1000000000)
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda', 'old_milp')
    num_gen = 100
    run(output_path,
        seed,
        num_gen)
