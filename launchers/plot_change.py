"""Plot the kl divergence, average path and unique&valid levels percentage of generated levels with the model at different epoches."""

import os

import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt

import algos.torch.dcgan.dcgan as dcgan
from launchers.hamming_distance_analysis import get_valid_lvls
from launchers.kl_divergence_analysis import count_pattern_distribution, count2dist, compute_kl
from launchers.zelda_duplicated_lvls_evaluation import evaluate, compute_duplicated_lvls


def run(plot_range):
    f_h = 2
    f_w = 2
    human_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda', 'Human_Json')
    human_lvls = get_valid_lvls(human_root)
    human_patterns = count_pattern_distribution(human_lvls, f_h, f_w)
    human_pattern_dist = count2dist(human_patterns)

    kl_lst = []
    average_path = []
    human_path = []
    duplicate_percentage = []
    valid_percentage = []
    valid_unique_percentage = []
    average_path_ratio = []
    for epoch_num in tqdm.tqdm(plot_range):
        gan_path = os.path.join(os.path.dirname(__file__),
                                'zelda_better_gan',
                                'netG_epoch_{}_999.pth'.format(epoch_num))

        # first generate 1000 levels
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        netG = dcgan.DCGAN_G(32, 32, 8, 64, 1, 0)
        netG.load_state_dict(torch.load(gan_path))
        netG.to(device)

        num_iter = 1000

        with torch.no_grad():
            netG.eval()

            noise = torch.FloatTensor(num_iter, 32, 1, 1).normal_(0, 1).to(device)
            output = netG(noise)

            im = output[:, :, :9, :13].cpu().numpy()
            im = np.argmax(im, axis=1).astype(int).tolist()

        valid_generated_lvls = []
        costs = []
        # first only keep valid levels
        for lvl in im:
            ret = evaluate(lvl)
            if ret:
                valid_generated_lvls.append(lvl)
                costs.append(ret[1])

        # now it's time to compute kl divergence
        gan_patterns = count_pattern_distribution(valid_generated_lvls, f_h, f_w)
        new_gan_patterns = {}
        for pattern in human_patterns:
            if pattern in gan_patterns:
                new_gan_patterns[pattern] = gan_patterns[pattern]
            else:
                new_gan_patterns[pattern] = 1e-5
        gan_pattern_dist = count2dist(new_gan_patterns)
        kl = compute_kl(human_pattern_dist, gan_pattern_dist)
        kl_lst.append(kl)

        average_cost = np.average(costs)
        average_path.append(average_cost)
        human_path.append(12.52)  # computed offline
        average_path_ratio.append(average_cost / 12.52)

        valid_percentage.append(len(valid_generated_lvls) / num_iter)
        num_duplicated, unique_lvls = compute_duplicated_lvls(valid_generated_lvls)
        valid_unique_percentage.append(len(unique_lvls) / num_iter)
        if len(valid_generated_lvls) > 0:
            duplicate_percentage.append(num_duplicated / len(valid_generated_lvls))
            # duplicate_percentage_x.append(epoch_num)
        else:
            duplicate_percentage.append(np.nan)

    fig = plt.figure()

    plt.plot(list(plot_range), kl_lst, label='KL Divergence')
    # plt.plot(list(plot_range), average_path, label='Average Path Length (GAN)')
    # plt.plot(list(plot_range), human_path, label='Average Path Length (Human)')
    plt.plot(list(plot_range), average_path_ratio, label='Average Path Length Ratio')
    plt.plot(list(plot_range), duplicate_percentage, label='Duplicated Levels')
    plt.plot(list(plot_range), valid_percentage, label='Valid Levels')
    plt.plot(list(plot_range), valid_unique_percentage, label='Valid and Unique Levels')
    # plt.axis('equal')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    plot_range = range(9, 50000, 100)
    run(plot_range)
