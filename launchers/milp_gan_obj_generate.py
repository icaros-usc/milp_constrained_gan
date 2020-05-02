"""We want to see if we can design some objective function to use gan's output as a guide for milp solver"""

import json
import os
import random
import time

import numpy as np
import torch
import tqdm

from algos.milp.program import Program
import algos.torch.dcgan.dcgan as dcgan

seed = 999
random.seed(seed)
torch.manual_seed(seed)

output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda', 'fake_milp_gan_obj')
os.makedirs(output_path, exist_ok=True)

program = Program()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

netG = dcgan.DCGAN_G(32, 32, 8, 64, 1, 0)

netG_checkpoint = os.path.join(os.path.dirname(__file__), 'samples', 'netG_epoch_23999_999.pth')
netG.load_state_dict(torch.load(netG_checkpoint))
netG.to(device)


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


total_time = 0
num_iter = 100
with torch.no_grad():
    netG.eval()
    for i in tqdm.tqdm(range(num_iter)):
        # first we use gan to generate the level
        fixed_noise = torch.FloatTensor(1, 32, 1, 1).normal_(0, 1).to(device)
        output = netG(fixed_noise)
        im = output[:, :, :9, :13].cpu().numpy()
        im = np.argmax(im, axis=1)

        # then we use milp to fix the level
        # We first need to form the solution
        Wc, Pc, Kc, Gc, E1c, E2c, E3c, Emc = program.get_objective_params_from_gan_output(im[0].tolist())
        start_time = time.time()
        program.set_objective(Wc, Pc, Kc, Gc, E1c, E2c, E3c, Emc)
        end_time = time.time()
        total_time += end_time - start_time
        si = program.solve()
        new_grid = vars2grid(si)

        with open(os.path.join(output_path, '{}.json'.format(i)), 'w') as f:
            f.write(json.dumps(new_grid))

average_time = total_time / num_iter
print('average time for running the solver: {}'.format(average_time))
