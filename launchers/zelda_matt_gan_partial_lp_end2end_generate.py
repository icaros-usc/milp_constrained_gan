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
from mipaal.mip_solvers import LPFunction
import algos.torch.dcgan.dcgan as dcgan
from algos.milp.zelda.utils import mip_sol_to_gan_out, gan_out_2_coefs
from algos.milp.zelda.milp_utils import fix_zelda_level
from mipaal.qpthlocal.qp import QPSolvers
from launchers.generate_utils import zelda_gan_output_to_txt
from utils.TrainLevelHelper import get_integer_lvl


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
                raise NotImplementedError
        if i < 8:
            grid.append([])
    return grid


def txt_to_gan_output(input):
    grid = [[]]
    for i in range(9):
        for j in range(13):
            if input[i][j] == 'w':
                grid[i].append(0)
            elif input[i][j] == '.':
                grid[i].append(1)
            elif input[i][j] == '+':
                grid[i].append(2)
            elif input[i][j] == 'g':
                grid[i].append(3)
            elif input[i][j] == '1':
                grid[i].append(4)
            elif input[i][j] == '2':
                grid[i].append(5)
            elif input[i][j] == '3':
                grid[i].append(6)
            elif input[i][j] == 'A':
                grid[i].append(7)
            else:
                raise NotImplementedError
        if i < 8:
            grid.append([])
    return grid


def run(output_path,
        gan_experiment,
        cuda,
        gpu_id,
        if_fix):
    os.makedirs(output_path, exist_ok=True)
    index2strJson = json.load(open('zelda_index2str.json', 'r'))
    str2index = {}
    for key in index2strJson:
        str2index[index2strJson[key]] = key

    # now we need to setup lp function
    # Step 1. We translate our cplex model to matrices
    milp_program = Program(partial=True)
    cpx = milp_program.get_cplex_prob()  # get the cplex problem from the docplex model
    cpx.cleanup(epsilon=0.0001)
    c, G, h, A, b, var_type = cplex_utils.cplex_to_matrices(cpx)
    # _, inds = sympy.Matrix(A).T.rref()
    # A = A[np.array(inds)]
    # b = b[np.array(inds)]

    if cuda:
        G = torch.from_numpy(G).float().cuda(gpu_id)
        h = torch.from_numpy(h).float().cuda(gpu_id)
        A = torch.from_numpy(A).float().cuda(gpu_id)
        b = torch.from_numpy(b).float().cuda(gpu_id)
        Q = 2e-6 * torch.eye(A.shape[1]).cuda(gpu_id)
        Q = Q.type_as(G).cuda(gpu_id)
    else:
        G = torch.from_numpy(G)
        h = torch.from_numpy(h)
        A = torch.from_numpy(A)
        b = torch.from_numpy(b)
        Q = 2e-6 * torch.eye(A.shape[1])
        Q = Q.type_as(G)

    lp_function = LPFunction(QPSolvers.GUROBI, verbose=False)

    # netG = dcgan.DCGAN_G(32, 32, 8, 64, 1, 0)
    netG = dcgan.DCGAN_G(32, 32, 8, 64, 1, 0)
    netG_checkpoint = os.path.join(gan_experiment, 'netG_epoch_5499_999.pth')
    netG.load_state_dict(torch.load(netG_checkpoint))
    if cuda:
        netG.cuda(gpu_id)

    num_iter = 1000
    with torch.no_grad():
        netG.eval()
        for i in tqdm.tqdm(range(num_iter)):
            # first we use gan to generate the level
            noise = torch.FloatTensor(1, 32, 1, 1).normal_(0, 1)
            if cuda:
                noise = noise.cuda(gpu_id)
            output = netG(noise)
            pred_coefs = gan_out_2_coefs(output, c.size)
            x = lp_function(Q, pred_coefs, G, h, A, b)
            output2 = mip_sol_to_gan_out(output, x)
            im = output2.data[:, :, :9, :13].cpu().numpy()
            numpyLvl = np.argmax(im, axis=1)
            if if_fix:
                level = zelda_gan_output_to_txt(numpyLvl[0])
                new_level = fix_zelda_level(level)
                numpyLvl = get_integer_lvl(new_level, str2index)

            with open(os.path.join(output_path, '{}.json'.format(i)), 'w') as f:
                f.write(json.dumps(numpyLvl.tolist()))


if __name__ == '__main__':
    if_fix = False
    seed = 999
    random.seed(seed)
    torch.manual_seed(seed)

    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda', 'zelda_better_end2end_3299_no_fix')
    gan_experiment = '/home/hejia/universal_grammar/milp_constrained_gan/launchers/zelda_better_end2end'
    run(output_path,
        gan_experiment,
        False,
        0,
        if_fix)
