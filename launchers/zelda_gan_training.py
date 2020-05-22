"""This file is for pure gan experiment."""

import argparse
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

import algos.torch.dcgan.dcgan as dcgan
import os
import json
from utils.TrainLevelHelper import get_lvls, get_integer_lvl


def run(nz,
        ngf,
        ndf,
        batch_size,
        niter,
        lrD,
        lrG,
        beta1,
        cuda,
        ngpu,
        gpu_id,
        path_netG,
        path_netD,
        clamp_lower,
        clamp_upper,
        D_iters,
        n_extra_layers,
        gan_experiment,
        adam,
        seed,
        lvl_data):
    os.chdir(".")
    print(os.getcwd())

    if not os.path.exists(gan_experiment):
        os.system('mkdir {0}'.format(gan_experiment))

    print("Manual Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    map_size = 32

    index2strJson = json.load(open('zelda_index2str.json', 'r'))
    str2index = {}
    for key in index2strJson:
        str2index[index2strJson[key]] = key
    dataroot = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda', 'Human')

    # get all levels and store them in one numpy array
    np_lvls = []
    lvls = get_lvls(dataroot)
    for lvl in lvls:
        numpyLvl = get_integer_lvl(lvl, str2index)
        np_lvls.append(numpyLvl)

    X = np.array(np_lvls)
    z_dims = len(index2strJson)

    num_batches = X.shape[0] / batch_size

    print("Batch size is " + str(batch_size))

    print("SHAPE ", X.shape)
    X_onehot = np.eye(z_dims, dtype='uint8')[X]

    X_onehot = np.rollaxis(X_onehot, 3, 1)
    # print("SHAPE ", X_onehot.shape)

    X_train = np.zeros((X.shape[0], z_dims, map_size, map_size))

    X_train[:, 1, :, :] = 1.0  # Fill with empty space

    # Pad part of level so its a square
    X_train[:X.shape[0], :, :X.shape[1], :X.shape[2]] = X_onehot

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    netG = dcgan.DCGAN_G(map_size, nz, z_dims, ngf, ngpu, n_extra_layers)
    netG.apply(weights_init)
    if path_netG != '':  # load checkpoint if needed
        netG.load_state_dict(torch.load(path_netG))

    netD = dcgan.DCGAN_D(map_size, nz, z_dims, ndf, ngpu, n_extra_layers)
    netD.apply(weights_init)
    if path_netD != '':
        netD.load_state_dict(torch.load(path_netD))

    input = torch.FloatTensor(batch_size, z_dims, map_size, map_size)
    noise = torch.FloatTensor(batch_size, nz, 1, 1)
    fixed_noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1)
    one = torch.FloatTensor([1])
    mone = one * -1

    if cuda:
        netD.cuda(gpu_id)
        netG.cuda(gpu_id)
        input = input.cuda(gpu_id)
        one, mone = one.cuda(gpu_id), mone.cuda(gpu_id)
        noise, fixed_noise = noise.cuda(gpu_id), fixed_noise.cuda(gpu_id)

    # setup optimizer
    if adam:
        optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(beta1, 0.999))
        print("Using ADAM")
    else:
        optimizerD = optim.RMSprop(netD.parameters(), lr=lrD)
        optimizerG = optim.RMSprop(netG.parameters(), lr=lrG)

    gen_iterations = 0
    for epoch in range(niter):

        # ! data_iter = iter(dataloader)

        X_train = X_train[torch.randperm(len(X_train))]

        i = 0
        while i < num_batches:  # len(dataloader):
            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            # train the discriminator Diters times
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                Diters = 100
            else:
                Diters = D_iters
            j = 0
            while j < Diters and i < num_batches:  # len(dataloader):
                j += 1

                # clamp parameters to a cube
                for p in netD.parameters():
                    p.data.clamp_(clamp_lower, clamp_upper)

                data = X_train[i * batch_size:(i + 1) * batch_size]

                i += 1

                real_cpu = torch.FloatTensor(data)

                netD.zero_grad()
                # batch_size = num_samples #real_cpu.size(0)

                if cuda:
                    real_cpu = real_cpu.cuda(gpu_id)

                input.resize_as_(real_cpu).copy_(real_cpu)
                inputv = Variable(input)

                errD_real = netD(inputv)
                errD_real.backward(one)

                # train with fake
                noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
                noisev = Variable(noise, volatile=True)  # totally freeze netG
                fake = Variable(netG(noisev).data)
                inputv = fake
                errD_fake = netD(inputv)
                errD_fake.backward(mone)
                errD = errD_real - errD_fake
                optimizerD.step()

            ############################
            # (2) Update G network
            ###########################
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            netG.zero_grad()
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev)
            errG = netD(fake)
            errG.backward(one)
            optimizerG.step()
            gen_iterations += 1

            print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                  % (epoch, niter, i, num_batches, gen_iterations,
                     errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))

        if epoch % 1000 == 999 or epoch == niter - 1:  # was 500
            fake = netG(Variable(fixed_noise, volatile=True))
            im = fake.data[:, :, :9, :13].cpu().numpy()
            im = np.argmax(im, axis=1)

            f = open('{0}/fake_level_epoch_{1}_{2}.json'.format(gan_experiment, epoch, seed), "w")
            f.write(json.dumps(im[0].tolist()))
            f.close()

            torch.save(netG.state_dict(), '{0}/netG_epoch_{1}_{2}.pth'.format(gan_experiment, epoch, seed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nz', type=int, default=32, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=50, help='input batch size')
    parser.add_argument('--niter', type=int, default=25000, help='number of epochs to train for')
    parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda', default=True)
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--gpu_id', type=int, default=0, help='the id of the gpu to use')
    parser.add_argument('--path_netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--path_netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--D_iters', type=int, default=5, help='number of D iters per each G iter')
    parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
    parser.add_argument('--gan_experiment', help='Where to store samples and models')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
    parser.add_argument('--seed', type=int, default=999, help='random seed for reproducibility')
    parser.add_argument('--lvl_data', help='Path to the human designed levels.')
    opt = parser.parse_args()

    run(opt.nz,
        opt.ngf,
        opt.ndf,
        opt.batch_size,
        opt.niter,
        opt.lrD,
        opt.lrG,
        opt.beta1,
        opt.cuda,
        opt.ngpu,
        opt.gpu_id,
        opt.path_netG,
        opt.path_netD,
        opt.clamp_lower,
        opt.clamp_upper,
        opt.D_iters,
        opt.n_extra_layers,
        opt.gan_experiment,
        opt.adam,
        opt.seed,
        opt.lvl_data)
