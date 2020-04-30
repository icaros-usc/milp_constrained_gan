import json
import os
import random

import numpy as np
import torch
import tqdm

from algos.milp.program import Program
import algos.torch.dcgan.dcgan as dcgan

seed = 999
random.seed(seed)
torch.manual_seed(seed)

output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda', 'fake_milp_gan_no_obj')
os.makedirs(output_path, exist_ok=True)

program = Program()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

netG = dcgan.DCGAN_G(32, 32, 8, 64, 1, 0)

netG_checkpoint = os.path.join(os.path.dirname(__file__), 'samples', 'netG_epoch_23999_999.pth')
netG.load_state_dict(torch.load(netG_checkpoint))
netG.to(device)

with torch.no_grad():
    netG.eval()
    for i in tqdm.tqdm(range(15000)):
        # first we use gan to generate the level
        fixed_noise = torch.FloatTensor(1, 32, 1, 1).normal_(0, 1).to(device)
        output = netG(fixed_noise)
        im = output[:, :, :9, :13].cpu().numpy()
        im = np.argmax(im, axis=1)

        # then we use milp to fix the level
        

        with open(os.path.join(output_path, '{}.json'.format(i)), 'w') as f:
            f.write(json.dumps(im[0].tolist()))