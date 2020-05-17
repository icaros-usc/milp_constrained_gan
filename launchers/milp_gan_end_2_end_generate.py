"""We want to see if we can design some objective function to use gan's output as a guide for milp solver"""

import json
import os
import random
import time
import shutil

import numpy as np
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

from algos.milp.zelda.program import Program
from mipaal.utils import cplex_utils, experiment_utils
from mipaal.mip_solvers import MIPFunction
import algos.torch.dcgan.dcgan as dcgan
from algos.milp.zelda.utils import mip_sol_to_gan_out, gan_out_2_coefs

seed = 999
random.seed(seed)
torch.manual_seed(seed)

output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda', 'fake_milp_gan_end_2_end')
os.makedirs(output_path, exist_ok=True)

program = Program()
# now we need to setup mipfunction
experiment_dir = os.path.join(os.path.dirname(__file__), 'milp_gan_end_2_end_experiment')
param_file = os.path.join(experiment_dir, 'params.json')
params = experiment_utils.Params(param_file)
shutil.rmtree(os.path.join(experiment_dir, 'runs'), ignore_errors=True)
writer = SummaryWriter(log_dir=os.path.join(experiment_dir, 'runs'))
models_dir = os.path.join(experiment_dir, 'models')
os.makedirs(models_dir, exist_ok=True)
# Step 1. We translate our cplex model to matrices
milp_program = Program()
cpx = milp_program.get_cplex_prob()  # get the cplex problem from the docplex model
cpx.cleanup(epsilon=0.0001)
c, G, h, A, b, var_type = cplex_utils.cplex_to_matrices(cpx)
# _, inds = sympy.Matrix(A).T.rref()
# A = A[np.array(inds)]
# b = b[np.array(inds)]

G = torch.from_numpy(G)
h = torch.from_numpy(h)
A = torch.from_numpy(A)
b = torch.from_numpy(b)
Q = 1e-5 * torch.eye(A.shape[1])
Q = Q.type_as(G)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

netG = dcgan.DCGAN_G(32, 32, 8, 64, 1, 0)

netG_checkpoint = os.path.join(os.path.dirname(__file__), 'samples', 'netG_epoch_179_999.pth')
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
num_iter = 5
with torch.no_grad():
    netG.eval()
    for i in tqdm.tqdm(range(num_iter)):
        # first we use gan to generate the level
        fixed_noise = torch.FloatTensor(1, 32, 1, 1).normal_(0, 1).to(device)
        output = netG(fixed_noise)
        pred_coefs = gan_out_2_coefs(output, c.size)
        mip_function = MIPFunction(var_type, G, h, A, b, verbose=0,
                                   input_mps=os.path.join(experiment_dir, 'gomory_prob.mps'),
                                   gomory_limit=params.gomory_limit,
                                   test_timing=params.test_timing,
                                   test_integrality=params.test_integrality,
                                   test_cuts_generated=params.test_cuts_generated)
        x = mip_function(Q, pred_coefs, G, h, A, b)
        mip_sol_to_gan_out(output, x)
        mip_function.release()
        im = output[:, :, :9, :13].cpu().numpy()
        im = np.argmax(im, axis=1)

        # then we use milp to fix the level
        # We first need to form the solution
        # Wc, Pc, Kc, Gc, E1c, E2c, E3c, Emc = program.get_objective_params_from_gan_output(im[0].tolist())
        # start_time = time.time()
        # program.set_objective(Wc, Pc, Kc, Gc, E1c, E2c, E3c, Emc)
        # si = program.solve()
        # end_time = time.time()
        # total_time += end_time - start_time
        # new_grid = vars2grid(si)

        with open(os.path.join(output_path, '{}.json'.format(i)), 'w') as f:
            f.write(json.dumps(im[0].tolist()))

average_time = total_time / num_iter
print('average time for running the solver: {}'.format(average_time))
