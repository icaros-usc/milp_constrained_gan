"""We want to see if we can design some objective function to use gan's output as a guide for milp solver"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import tqdm

import torch.backends.cudnn as cudnn
import torch.optim as optim

from mip import get_mip_program
#from algos.milp.zelda.program import Program
import algos.torch.dcgan.dcgan as dcgan
from utils.TrainLevelHelper import get_lvls, get_integer_lvl

from mipaal.utils import cplex_utils, experiment_utils

from mipaal.mip_solvers import MIPFunction

from torch.autograd import Variable

import sympy

#oint if needed
#    netG.load_state_dict(torch.load(opt.netG))

#netG_checkpoint = os.path.join(os.path.dirname(__file__), 'samples', 'netG_epoch_23999_999.pth')
#netG.load_state_dict(torch.load(netG_checkpoint))
#netG.to(device)

parser = argparse.ArgumentParser()
parser.add_argument('--nz', type=int, default=32, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
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



def gan_out_2_coefs(gan_output, len_coeff):
    """Use this function to translate the output of generator to the coefficients."""
    num_nodes = 9*13
    out = torch.zeros(len_coeff)
    # flatten it
    out[0:936] = torch.flatten(gan_output[:, :, :9, :13])
    # make it negative because we want to maximize it
    out = -out
    # make it the same size with the coefficients
    # Wc = out[0:num_nodes]
    # Emc = out[num_nodes:2*num_nodes]
    # Kc = out[2*num_nodes:3*num_nodes]
    # Gc = out[3*num_nodes:4*num_nodes]
    # E1c = out[4*num_nodes:5*num_nodes]
    # E2c = out[5*num_nodes:6*num_nodes]
    # E3c = out[6*num_nodes:7*num_nodes]
    # Pc = out[7*num_nodes:8*num_nodes]
  
    return out

if opt.experiment is None:
    opt.experiment = 'samples'
if not os.path.exists(opt.experiment):
    os.system('mkdir {0}'.format(opt.experiment))

print("Manual Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
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

num_batches = X.shape[0] / opt.batchSize
print("Batch size is " + str(opt.batchSize))

#from IPython import embed
#embed()

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
    netD.cuda()
    netG.cuda()
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


output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda', 'fake_mipaal_gan_obj')
os.makedirs(output_path, exist_ok=True)

#program = Program()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gen_iterations = 0
for epoch in range(opt.niter):

    # ! data_iter = iter(dataloader)

    X_train = X_train[torch.randperm(len(X_train))]

    i = 0
    while i < num_batches:  # len(dataloader):
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
            #fixed_noise = torch.FloatTensor(1, 32, 1, 1).normal_(0, 1).to(device)
        noise.resize_(1, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        output = netG(noisev)

        #from IPython import embed
        #embed()
        #with torch.no_grad():
        #  im = output[:, :, :9, :13].cpu()
        #  im = np.argmax(im, axis=1)''
        #from IPython import embed
        #embed()

        #cpx = program._model.get_cplex()
        #program._model.parameters.threads.set(12)
        #Wc, Pc, Kc, Gc, E1c, E2c, E3c, Emc = program.get_objective_params_from_gan_output(im[0])
        #program.set_objective(Wc, Pc, Kc, Gc, E1c, E2c, E3c, Emc)
        cpx = get_mip_program()
        pred_coefs, G, h, A, b, var_type = cplex_utils.cplex_to_matrices(cpx)

        pred_coefs = gan_out_2_coefs(output, pred_coefs.size)


        # preprocess A to remove linearly independent rows (Bryan code)
        #_, inds = sympy.Matrix(A).T.rref(simplified=True)


        #A = A[np.array(inds)]
        #b = b[np.array(inds)]
#A = torch.from_numpy(A)
#            b = torch.from_numpy(b)
        G = torch.from_numpy(G)
        h = torch.from_numpy(h)
        A = torch.from_numpy(A)
        b = torch.from_numpy(b)

        Q = 1e-6 * torch.eye(A.shape[1])
        Q = Q.type_as(G)

        #pred_coefs = torch.from_numpy(pred_coefs)
        mip_function = MIPFunction(var_type, G, h, A, b, verbose=0)

        #from IPython import embed
        #embed()
        
        mip_sol = mip_function(Q, pred_coefs, G, h, A, b)

         #Wc, Pc, Kc, Gc, E1c, E2c, E3c, Emc = program.get_objective_params_from_gan_output(im[0].tolist())
        N = 9 * 13
        x = mip_sol[0, 0:8 * N].view(-1, 9, 13)
        output[0, :, :9, :13] = x
        
        errG = netD(output)
        errG.backward()
        mip_function.release()

        optimizerG.step()
        gen_iterations += 1

        print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                  % (epoch, opt.niter, i, num_batches, gen_iterations,
                     errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))

    if epoch % 100 == 99 or epoch == opt.niter - 1:  # was 500
        fake = netG(Variable(fixed_noise, volatile=True))
        im = fake.data[:, :, :9, :13].cpu().numpy()
        im = np.argmax(im, axis=1)

        f = open('{0}/fake_level_epoch_{1}_{2}.json'.format(opt.experiment, epoch, opt.seed), "w")
        f.write(json.dumps(im[0].tolist()))
        f.close()

        torch.save(netG.state_dict(), '{0}/netG_epoch_{1}_{2}.pth'.format(opt.experiment, epoch, opt.seed))

# exit()

# def vars2grid(si):
#     grid = [[1 for _ in range(13)] for _ in range(9)]
#     for i in range(9):
#         for j in range(13):
#             ind = i * 13 + j
#             if si.get_value(program.W[ind]) == 1:
#                 grid[i][j] = 0
#             elif si.get_value(program.K[ind]) == 1:
#                 grid[i][j] = 2
#             elif si.get_value(program.G[ind]) == 1:
#                 grid[i][j] = 3
#             elif si.get_value(program.E1[ind]) == 1:
#                 grid[i][j] = 4
#             elif si.get_value(program.E2[ind]) == 1:
#                 grid[i][j] = 5
#             elif si.get_value(program.E3[ind]) == 1:
#                 grid[i][j] = 6
#             elif si.get_value(program.P[ind]) == 1:
#                 grid[i][j] = 7
#     return grid


# total_time = 0
# num_iter = 100
# with torch.no_grad():
#     netG.eval()
#     for i in tqdm.tqdm(range(num_iter)):
#         # first we use gan to generate the level
#         fixed_noise = torch.FloatTensor(1, 32, 1, 1).normal_(0, 1).to(device)
#         output = netG(fixed_noise)
#         im = output[:, :, :9, :13].cpu().numpy()
#         im = np.argmax(im, axis=1)

#         # then we use milp to fix the level
#         # We first need to form the solution
#         Wc, Pc, Kc, Gc, E1c, E2c, E3c, Emc = program.get_objective_params_from_gan_output(im[0].tolist())
#         start_time = time.time()
#         program.set_objective(Wc, Pc, Kc, Gc, E1c, E2c, E3c, Emc)
#         end_time = time.time()
#         total_time += end_time - start_time
#         si = program.solve()
#         new_grid = vars2grid(si)

#         with open(os.path.join(output_path, '{}.json'.format(i)), 'w') as f:
#             f.write(json.dumps(new_grid))



# average_time = total_time / num_iter
# print('average time for running the solver: {}'.format(average_time))
