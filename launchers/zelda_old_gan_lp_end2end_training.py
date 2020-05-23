"""This file is for gan and lp relaxation experiment."""

import argparse
import random
import shutil
import time
import os
import json

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torch

import algos.torch.dcgan.dcgan as dcgan
from torch.utils.tensorboard import SummaryWriter
from utils.TrainLevelHelper import get_lvls, get_integer_lvl
from algos.milp.zelda.program import Program
from mipaal.utils import cplex_utils
from mipaal.mip_solvers import LPFunction
from algos.milp.zelda.utils import mip_sol_to_gan_out, gan_out_2_coefs


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
        path_netG,
        path_netD,
        clamp_lower,
        clamp_upper,
        D_iters,
        n_extra_layers,
        gan_experiment,
        mipaal_experiment,
        adam,
        seed,
        lvl_data):
    os.chdir(".")
    print(os.getcwd())

    # now we need to setup lpfunction
    shutil.rmtree(os.path.join(mipaal_experiment, 'runs'), ignore_errors=True)
    writer = SummaryWriter(log_dir=os.path.join(mipaal_experiment, 'runs'))
    models_dir = os.path.join(mipaal_experiment, 'models')
    os.makedirs(models_dir, exist_ok=True)
    # Step 1. We translate our cplex model to matrices
    milp_program = Program()
    cpx = milp_program.get_cplex_prob()  # get the cplex problem from the docplex model
    cpx.cleanup(epsilon=0.0001)
    c, G, h, A, b, var_type = cplex_utils.cplex_to_matrices(cpx)
    # _, inds = sympy.Matrix(A).T.rref()
    # A = A[np.array(inds)]
    # b = b[np.array(inds)]

    if cuda:
        G = torch.from_numpy(G).float().cuda()
        h = torch.from_numpy(h).float().cuda()
        A = torch.from_numpy(A).float().cuda()
        b = torch.from_numpy(b).float().cuda()
        Q = 1e-5 * torch.eye(A.shape[1]).cuda()
        Q = Q.type_as(G).cuda()
    else:
        G = torch.from_numpy(G).float()
        h = torch.from_numpy(h).float()
        A = torch.from_numpy(A).float()
        b = torch.from_numpy(b).float()
        Q = 1e-5 * torch.eye(A.shape[1])
        Q = Q.type_as(G)

    os.makedirs(gan_experiment, exist_ok=True)

    random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = True  # enable cudnn auto-tuner for finding the optimial set of algorithms.

    if torch.cuda.is_available() and not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    map_size = 32

    index2strJson = json.load(open('zelda_index2str.json', 'r'))
    str2index = {}
    for key in index2strJson:
        str2index[index2strJson[key]] = key

    # get all levels and store them in one numpy array
    np_lvls = []
    lvls = get_lvls(lvl_data)
    for lvl in lvls:
        numpyLvl = get_integer_lvl(lvl, str2index)
        np_lvls.append(numpyLvl)

    X = np.array(np_lvls)
    z_dims = len(index2strJson)

    num_batches = X.shape[0] / batch_size

    X_onehot = np.eye(z_dims, dtype='uint8')[X]
    X_onehot = np.rollaxis(X_onehot, 3, 1)
    X_train = np.zeros((X.shape[0], z_dims, map_size, map_size))
    X_train[:, 1, :, :] = 1.0  # Fill with empty space
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
        netD.cuda()
        netG.cuda()
        input = input.cuda()
        one, mone = one.cuda(), mone.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

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
        X_train = X_train[torch.randperm(len(X_train))]  # shuffle the training data

        i = 0
        total_time = 0
        num_running = 0
        while i < num_batches:
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
                    real_cpu = real_cpu.cuda()

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
            # here we will plug in the mipfunction
            # Step 2. construct coefficients from the output of the netG
            pred_coefs = gan_out_2_coefs(fake, c.size, cuda)

            lp_function = LPFunction(var_type, G, h, A, b, input_mps=os.path.join(mipaal_experiment, 'gomory_prob.mps'))
            start_time = time.time()
            x = lp_function(Q, pred_coefs, G, h, A, b)
            end_time = time.time()
            total_time += end_time - start_time
            num_running += 1
            mip_sol_to_gan_out(fake, x)

            errG = netD(fake)
            errG.backward(one)
            # mip_function.release()
            lp_function.release()
            optimizerG.step()
            gen_iterations += 1

            print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                  % (epoch, niter, i, num_batches, gen_iterations,
                     errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))

        print('average time for running the lp_function: {}'.format(total_time / num_running))
        if epoch % 10 == 9 or epoch == niter - 1:  # was 500
            fake = netG(Variable(fixed_noise, volatile=True))
            pred_coefs = gan_out_2_coefs(fake, c.size, cuda)
            lp_function = LPFunction(var_type, G, h, A, b,
                                     input_mps=os.path.join(mipaal_experiment, 'gomory_prob.mps'))
            x = lp_function(Q, pred_coefs, G, h, A, b)
            mip_sol_to_gan_out(fake, x)
            lp_function.release()
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
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
    parser.add_argument('--niter', type=int, default=25000, help='number of epochs to train for')
    parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda', default=True)
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--path_netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--path_netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--D_iters', type=int, default=5, help='number of D iters per each G iter')
    parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
    parser.add_argument('--gan_experiment', help='Where to store samples and models')
    parser.add_argument('--mipaal_experiment', help='Where to store mipaal parameters file')
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
        opt.path_netG,
        opt.path_netD,
        opt.clamp_lower,
        opt.clamp_upper,
        opt.D_iters,
        opt.n_extra_layers,
        opt.gan_experiment,
        opt.mipaal_experiment,
        opt.adam,
        opt.seed,
        opt.lvl_data)
