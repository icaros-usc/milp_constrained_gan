import json
import os
import random

import numpy as np
import torch
import tqdm

import algos.torch.dcgan.dcgan as dcgan
from launchers.generate_utils import pacman_gan_output_to_txt
from algos.milp.pacman.milp_utils import fix_pacman_level
from utils.TrainLevelHelper import get_integer_lvl


def run(output_path,
        gan_path,
        num_gen,
        if_fix):
    os.makedirs(output_path, exist_ok=True)
    index2strJson = json.load(open('pacman_index2str.json', 'r'))
    str2index = {}
    for key in index2strJson:
        str2index[index2strJson[key]] = key

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    netG = dcgan.DCGAN_G(32, 32, 10, 64, 1, 0)

    netG.load_state_dict(torch.load(gan_path))
    netG.to(device)
    with torch.no_grad():
        netG.eval()
        for i in tqdm.tqdm(range(num_gen)):
            fixed_noise = torch.FloatTensor(1, 32, 1, 1).normal_(0, 1).to(device)
            output = netG(fixed_noise)
            im = output[:, :, :31, :28].cpu().numpy()
            numpyLvl = np.argmax(im, axis=1)[0]
            if if_fix:
                level = pacman_gan_output_to_txt(numpyLvl)
                new_level = fix_pacman_level(level)
                numpyLvl = get_integer_lvl(new_level, str2index)

            with open(os.path.join(output_path, '{}.json'.format(i)), 'w') as f:
                f.write(json.dumps(numpyLvl.tolist()))


if __name__ == '__main__':
    if_fix = True
    seed = 999
    random.seed(seed)
    torch.manual_seed(seed)

    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'pacman', 'pacman_gan')
    gan_path = os.path.join(os.path.dirname(__file__), 'default_pacman_samples', 'netG_epoch_24999_999.pth')

    num_gen = 10

    run(output_path,
        gan_path,
        num_gen,
        if_fix)
