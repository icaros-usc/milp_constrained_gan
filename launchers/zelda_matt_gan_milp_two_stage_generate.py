"""We want to see if we can design some objective function to use gan's output as a guide for milp solver"""

import json
import os
import random

import numpy as np
import torch
import tqdm

import algos.torch.dcgan.dcgan as dcgan

from algos.milp.zelda.matt_milp_utils import fix_zelda_level
from utils.TrainLevelHelper import get_integer_lvl
from launchers.generate_utils import zelda_gan_output_to_txt


def run(output_path,
        gan_path,
        num_gen):
    index2strJson = json.load(open('zelda_index2str.json', 'r'))
    str2index = {}
    for key in index2strJson:
        str2index[index2strJson[key]] = key

    os.makedirs(output_path, exist_ok=True)  # create the output directory if it doesn't exist.

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
            level = zelda_gan_output_to_txt(im[0])
            new_level = fix_zelda_level(level)
            numpyLvl = get_integer_lvl(new_level, str2index)
            with open(os.path.join(output_path, '{}.json'.format(i)), 'w') as f:
                f.write(json.dumps(numpyLvl.tolist()))


if __name__ == '__main__':
    # first set random seed
    seed = 999
    random.seed(seed)
    torch.manual_seed(seed)

    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda', 'matt_gan_milp_two_stage')
    gan_path = os.path.join(os.path.dirname(__file__), 'samples', 'netG_epoch_23999_999.pth')
    num_gen = 100  # the number of maps to gennerate

    run(output_path,
        gan_path,
        num_gen)
