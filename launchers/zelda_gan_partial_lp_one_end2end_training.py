"""This file is for gan and lp relaxation experiment."""

import argparse
import random
import shutil
import os
import json
import time

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch
from torchviz import make_dot

import algos.torch.dcgan.dcgan as dcgan
from torch.utils.tensorboard import SummaryWriter
from utils.TrainLevelHelper import get_lvls, get_integer_lvl
from mipaal.utils import cplex_utils
from mipaal.mip_solvers import LPFunction
from algos.milp.zelda.utils import mip_sol_to_gan_out, gan_out_2_coefs
from algos.milp.zelda.milp_utils import get_mip_program
from mipaal.qpthlocal.qp import QPSolvers
from algos.milp.zelda.program import Program


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
        n_extra_layers,
        gan_experiment,
        mipaal_experiment,
        adam,
        seed,
        lvl_data):
    # Now we need to setup lpfunction
    shutil.rmtree(os.path.join(mipaal_experiment, 'runs'), ignore_errors=True)
    writer = SummaryWriter(log_dir=os.path.join(mipaal_experiment, 'runs'))
    models_dir = os.path.join(mipaal_experiment, 'models')
    os.makedirs(models_dir, exist_ok=True)
    # Step 1. We translate our cplex model to matrices
    milp_program = Program(partial=True)
    cpx = milp_program.get_cplex_prob()  # get the cplex problem from the docplex model
    cpx.cleanup(epsilon=0.0001)
    c, G, h, A, b, var_type = cplex_utils.cplex_to_matrices(cpx)
    # _, inds = sympy.Matrix(A).T.rref()
    # A = A[np.array(inds)]
    # b = b[np.array(inds)]
    #
    # cpx = get_mip_program()
    # cpx.cleanup(epsilon=0.0001)
    # c, G, h, A, b, var_type = cplex_utils.cplex_to_matrices(cpx)
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
    if cuda:
        lp_function = LPFunction(QPSolvers.PDIPM_BATCHED, verbose=False)
    else:
        lp_function = LPFunction(QPSolvers.GUROBI, verbose=False)


    os.makedirs(gan_experiment, exist_ok=True)

    random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = True  # enable cudnn auto-tuner for finding the optimial set of algorithms.

    if torch.cuda.is_available() and not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    map_size = 32

    index2strJson = json.load(open(os.path.join(os.path.dirname(__file__), 'zelda_index2str.json'), 'r'))
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
    X_train[:, 1, :, :] = 1.0
    X_train[:X.shape[0], :, :X.shape[1], :X.shape[2]] = X_onehot

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    netG = dcgan.DCGAN_G(map_size, nz, z_dims, ngf, ngpu, n_extra_layers)
    netG.apply(weights_init)
    if path_netG != '':
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

    for epoch in range(niter):
        start_time = time.time()
        X_train = X_train[torch.randperm(len(X_train))]  # shuffle the training data

        ############################
        # (1) Update D network
        ###########################
        i = 0
        total_errD_fake = 0
        total_errD_real = 0
        total_errG = 0
        while i < num_batches:
            netD.zero_grad()
            # clamp parameters to a cube
            for p in netD.parameters():
                p.data.clamp_(clamp_lower, clamp_upper)
            data = X_train[i * batch_size:(i + 1) * batch_size]
            if cuda:
                real_cpu = torch.FloatTensor(data).cuda(gpu_id)
            else:
                real_cpu = torch.FloatTensor(data)
            input.resize_as_(real_cpu).copy_(real_cpu)
            errD_real = netD(input)
            errD_real.backward(one)
            total_errD_real += errD_real.item()

            # train with fake
            noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
            fake = netG(noise)
            pred_coefs = gan_out_2_coefs(fake, c.size, gpu_id, cuda)
            x = lp_function(Q, pred_coefs, G, h, A, b)
            fake2 = mip_sol_to_gan_out(fake, x)
            errD_fake = netD(fake2.detach())
            errD_fake.backward(mone)
            total_errD_fake += errD_fake.item()
            # i += 1
            optimizerD.step()

            ############################
            # (2) Update G network
            ###########################
            netG.zero_grad()
            errG = netD(fake2)
            errG.backward(one)
            total_errG += errG.item()
            optimizerG.step()
            i += 1

        average_errG = total_errG / num_batches
        average_errD_fake = total_errD_fake / num_batches
        average_errD_real = total_errD_real / num_batches

        end_time = time.time()

        print('[%d/%d] Loss_G: %f Loss_D_real: %f Loss_D_fake %f; time: [%d]'
              % (epoch, niter, average_errG, average_errD_real, average_errD_fake, end_time - start_time))

        if epoch % 100 == 99 or epoch == niter - 1:
            netG.eval()
            with torch.no_grad():
                fake = netG(fixed_noise)
                pred_coefs = gan_out_2_coefs(fake, c.size, gpu_id, cuda)
                x = lp_function(Q, pred_coefs, G, h, A, b)
                fake2 = mip_sol_to_gan_out(fake, x)
                im = fake2.data[:, :, :9, :13].cpu().numpy()
                im = np.argmax(im, axis=1)
            with open('{0}/fake_level_epoch_{1}_{2}.json'.format(gan_experiment, epoch, seed), "w") as f:
                f.write(json.dumps(im[0].tolist()))
            torch.save(netG.state_dict(), '{0}/netG_epoch_{1}_{2}.pth'.format(gan_experiment, epoch, seed))
            torch.save(netD.state_dict(), '{0}/netD_epoch_{1}_{2}.pth'.format(gan_experiment, epoch, seed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nz', type=int, default=32, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=50, help='input batch size')
    parser.add_argument('--niter', type=int, default=25000, help='number of epochs to train for')
    parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for Generator, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--gpu_id', type=int, default=0, help='the id of the gpu to use')
    parser.add_argument('--path_netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--path_netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
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
        opt.gpu_id,
        opt.path_netG,
        opt.path_netD,
        opt.clamp_lower,
        opt.clamp_upper,
        opt.n_extra_layers,
        opt.gan_experiment,
        opt.mipaal_experiment,
        opt.adam,
        opt.seed,
        opt.lvl_data)
