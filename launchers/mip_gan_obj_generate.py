"""We want to see if we can design some objective function to use gan's output as a guide for milp solver"""

import json
import os
import random
import time

import numpy as np
import torch
import tqdm

import algos.torch.dcgan.dcgan as dcgan

from mip import fix_zelda_level
from utils.TrainLevelHelper import get_lvls, get_integer_lvl


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
                sys.error('unknown character')
        if i < 8:
            grid.append([])

    return grid


index2strJson = json.load(open('index2str.json', 'r'))
str2index = {}
for key in index2strJson:
    str2index[index2strJson[key]] = key


seed = 999
random.seed(seed)
torch.manual_seed(seed)

output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda', 'mip_gan_obj')
os.makedirs(output_path, exist_ok=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

netG = dcgan.DCGAN_G(32, 32, 8, 64, 1, 0)

netG_checkpoint = os.path.join(os.path.dirname(__file__), 'samples', 'netG_epoch_23999_999.pth')
netG.load_state_dict(torch.load(netG_checkpoint))
netG.to(device)



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

        level = gan_output_to_txt(im[0])
 
        new_level = fix_zelda_level(level)
        for line in new_level:
          print(line)

        #from IPython import embed
        #embed()
        # then we use milp to fix the level
        # We first need to form the solution
        #Wc, Pc, Kc, Gc, E1c, E2c, E3c, Emc = program.get_objective_params_from_gan_output(im[0].tolist())
        #start_time = time.time()
        #program.set_objective(Wc, Pc, Kc, Gc, E1c, E2c, E3c, Emc)
        #end_time = time.time()
        #total_time += end_time - start_time
        #si = program.solve()
        #from IPython import embed
        #embed()
        numpyLvl = get_integer_lvl(new_level, str2index)
        with open(os.path.join(output_path, '{}.json'.format(i)), 'w') as f:
            f.write(json.dumps(numpyLvl.tolist()))

#average_time = total_time / num_iter
#print('average time for running the solver: {}'.format(average_time))
