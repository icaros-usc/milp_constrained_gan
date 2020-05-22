"""We want to see if we can design some objective function to use gan's output as a guide for milp solver"""

import json
import os
import random
import time

import numpy as np
import torch
import tqdm

from algos.milp.zelda.program import Program
import algos.torch.dcgan.dcgan as dcgan
from launchers.generate_utils import zelda_milp_vars2grid


def run(output_path,
        gan_path,
        num_gen):
    os.makedirs(output_path, exist_ok=True)
    program = Program()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    netG = dcgan.DCGAN_G(32, 32, 8, 64, 1, 0)
    netG.load_state_dict(torch.load(gan_path))
    netG.to(device)

    with torch.no_grad():
        netG.eval()
        for i in tqdm.tqdm(range(num_gen)):
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
            si = program.solve()
            new_grid = zelda_milp_vars2grid(si, program)
            with open(os.path.join(output_path, '{}.json'.format(i)), 'w') as f:
                f.write(json.dumps(new_grid))


if __name__ == '__main__':
    seed = 999
    random.seed(seed)
    torch.manual_seed(seed)

    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda', 'old_gan_milp_two_stage')
    gan_path = os.path.join(os.path.dirname(__file__), 'samples', 'netG_epoch_23999_999.pth')
    num_gen = 100

    run(output_path,
        gan_path,
        num_gen)

