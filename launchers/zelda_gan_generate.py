import json
import os
import random

import numpy as np
import torch
import tqdm

import algos.torch.dcgan.dcgan as dcgan


def run(output_path,
        gan_path,
        num_gen):
    os.makedirs(output_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    netG = dcgan.DCGAN_G(32, 32, 8, 64, 1, 0)

    netG.load_state_dict(torch.load(gan_path))
    netG.to(device)

    with torch.no_grad():
        netG.eval()
        for i in tqdm.tqdm(range(num_gen)):
            fixed_noise = torch.FloatTensor(1, 32, 1, 1).normal_(0, 1).to(device)
            output = netG(fixed_noise)
            im = output[:, :, :9, :13].cpu().numpy()
            im = np.argmax(im, axis=1)

            with open(os.path.join(output_path, '{}.json'.format(i)), 'w') as f:
                f.write(json.dumps(im[0].tolist()))


if __name__ == '__main__':
    seed = 999
    random.seed(seed)
    torch.manual_seed(seed)

    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda', 'gan')
    gan_path = os.path.join(os.path.dirname(__file__), 'zelda_gan_samples', 'netG_epoch_24999_999.pth')

    num_gen = 1000

    run(output_path,
        gan_path,
        num_gen)
