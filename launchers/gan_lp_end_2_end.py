from __future__ import print_function

import argparse
import math
import random
import shutil
import time
import os
import json

import matplotlib.pyplot as plt
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torch

import algos.torch.dcgan.dcgan as dcgan
from torch.utils.tensorboard import SummaryWriter
from utils.TrainLevelHelper import get_lvls, get_integer_lvl
from algos.milp.zelda.program import Program
from mipaal.utils import cplex_utils, experiment_utils
from mipaal.mip_solvers import MIPFunction, LPFunction
from algos.milp.zelda.utils import mip_sol_to_gan_out, gan_out_2_coefs

os.chdir(".")
print(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument('--nz', type=int, default=32, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--niter', type=int, default=25000, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')

parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--problem', type=int, default=0, help='Level examples')
parser.add_argument('--json', default=None, help='Json file')
parser.add_argument('--seed', type=int, default=999, help='random seed for reproducibility')
opt = parser.parse_args()
# print(opt)

# now we need to setup mipfunction
experiment_dir = os.path.join(os.path.dirname(__file__), 'milp_gan_end_2_end_experiment')
param_file = os.path.join(experiment_dir, 'params.json')
params = experiment_utils.Params(param_file)
shutil.rmtree(os.path.join(experiment_dir, 'runs'), ignore_errors=True)
writer = SummaryWriter(log_dir=os.path.join(experiment_dir, 'runs'))
models_dir = os.path.join(experiment_dir, 'models')
os.makedirs(models_dir, exist_ok=True)
# Step 1. We translate our cplex model to matrices
milp_program = Program()
cpx = milp_program.get_cplex_prob()  # get the cplex problem from the docplex model
cpx.cleanup(epsilon=0.0001)
c, G, h, A, b, var_type = cplex_utils.cplex_to_matrices(cpx)
# _, inds = sympy.Matrix(A).T.rref()
# A = A[np.array(inds)]
# b = b[np.array(inds)]

G = torch.from_numpy(G).float().cuda()
h = torch.from_numpy(h).float().cuda()
A = torch.from_numpy(A).float().cuda()
b = torch.from_numpy(b).float().cuda()
Q = 1e-5 * torch.eye(A.shape[1]).cuda()
Q = Q.type_as(G).cuda()

if opt.experiment is None:
    opt.experiment = 'samples'
if not os.path.exists(opt.experiment):
    os.system('mkdir {0}'.format(opt.experiment))

print("Manual Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)

cudnn.benchmark = True

opt.cuda = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

map_size = 32

"""
if opt.json is None:
    if opt.problem == 0:
        examplesJson = "NewTrainingLevels.json"
    else:
        examplesJson = "sepEx/examplemario{}.json".format(opt.problem)
else:
    examplesJson = opt.json
X = np.array(json.load(open(examplesJson)))
z_dims = 23 # Number different tile types
"""

index2strJson = json.load(open('index2str.json', 'r'))
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

num_batches = X.shape[0] / opt.batchSize

print("Batch size is " + str(opt.batchSize))

print("SHAPE ", X.shape)
X_onehot = np.eye(z_dims, dtype='uint8')[X]

X_onehot = np.rollaxis(X_onehot, 3, 1)
# print("SHAPE ", X_onehot.shape)

X_train = np.zeros((X.shape[0], z_dims, map_size, map_size))

X_train[:, 1, :, :] = 1.0  # Fill with empty space

# Pad part of level so its a square
X_train[:X.shape[0], :, :X.shape[1], :X.shape[2]] = X_onehot

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

n_extra_layers = int(opt.n_extra_layers)


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
if opt.netG != '':  # load checkpoint if needed
    netG.load_state_dict(torch.load(opt.netG))
# print(netG)

netD = dcgan.DCGAN_D(map_size, nz, z_dims, ndf, ngpu, n_extra_layers)
netD.apply(weights_init)

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
# print(netD)

input = torch.FloatTensor(opt.batchSize, z_dims, map_size, map_size)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
one = torch.FloatTensor([1])
mone = one * -1


def tiles2image(tiles):
    return plt.get_cmap('rainbow')(tiles / float(z_dims))


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:]
    image = np.zeros((height * shape[0], width * shape[1], shape[2]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = img
    return image


if opt.cuda:
    netD.cuda(1)
    netG.cuda(1)
    input = input.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

# setup optimizer
if opt.adam:
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
    print("Using ADAM")
else:
    optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lrD)
    optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG)

gen_iterations = 0
for epoch in range(opt.niter):

    # ! data_iter = iter(dataloader)

    X_train = X_train[torch.randperm(len(X_train))]

    i = 0
    total_time = 0
    num_running = 0
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
            Diters = opt.Diters
        j = 0
        while j < Diters and i < num_batches:  # len(dataloader):
            j += 1

            # clamp parameters to a cube
            for p in netD.parameters():
                p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

            data = X_train[i * opt.batchSize:(i + 1) * opt.batchSize]

            i += 1

            real_cpu = torch.FloatTensor(data)

            if (False):
                # im = data.cpu().numpy()
                print(data.shape)
                real_cpu = combine_images(tiles2image(np.argmax(data, axis=1)))
                print(real_cpu)
                plt.imsave('{0}/real_samples.png'.format(opt.experiment), real_cpu)
                exit()

            netD.zero_grad()
            # batch_size = num_samples #real_cpu.size(0)

            if opt.cuda:
                real_cpu = real_cpu.cuda()

            input.resize_as_(real_cpu).copy_(real_cpu)
            inputv = Variable(input)

            errD_real = netD(inputv)
            errD_real.backward(one)

            # train with fake
            noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
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
        noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        # here we will plug in the mipfunction
        # Step 2. construct coefficients from the output of the netG
        pred_coefs = gan_out_2_coefs(fake, c.size, True)
        # mip_function = MIPFunction(var_type, G, h, A, b, verbose=0,
        #                            input_mps=os.path.join(experiment_dir, 'gomory_prob.mps'),
        #                            gomory_limit=params.gomory_limit,
        #                            test_timing=params.test_timing,
        #                            test_integrality=params.test_integrality,
        #                            test_cuts_generated=params.test_cuts_generated)
        lp_function = LPFunction(var_type, G, h, A, b, input_mps=os.path.join(experiment_dir, 'gomory_prob.mps'))
        start_time = time.time()
        # x = mip_function(Q, pred_coefs, G, h, A, b)
        x = lp_function(Q, pred_coefs, G, h, A, b).cuda()
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
              % (epoch, opt.niter, i, num_batches, gen_iterations,
                 errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))

    print('average time for running the lp_function: {}'.format(total_time / num_running))
    if epoch % 10 == 9 or epoch == opt.niter - 1:  # was 500
        fake = netG(Variable(fixed_noise, volatile=True))
        pred_coefs = gan_out_2_coefs(fake, c.size)
        lp_function = LPFunction(var_type, G, h, A, b,
                                   input_mps=os.path.join(experiment_dir, 'gomory_prob.mps'))
        x = lp_function(Q, pred_coefs, G, h, A, b)
        mip_sol_to_gan_out(fake, x)
        lp_function.release()
        im = fake.data[:, :, :9, :13].cpu().numpy()
        im = np.argmax(im, axis=1)

        f = open('{0}/fake_level_epoch_{1}_{2}.json'.format(opt.experiment, epoch, opt.seed), "w")
        f.write(json.dumps(im[0].tolist()))
        f.close()

        torch.save(netG.state_dict(), '{0}/netG_epoch_{1}_{2}.pth'.format(opt.experiment, epoch, opt.seed))

    # do checkpointing
    # torch.save(netG.state_dict(), '{0}/netG_epoch_{1}_{2}.pth'.format(opt.experiment, epoch, opt.seed))
    # torch.save(netD.state_dict(), '{0}/netD_epoch_{1}_{2}.pth'.format(opt.experiment, epoch, opt.seed))
    # torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
    # torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))