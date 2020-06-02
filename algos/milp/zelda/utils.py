import torch


def gan_out_2_coefs(gan_output, len_coeff, gpu_id=0, if_cuda=False):
    """Use this function to translate the output of generator to the coefficients."""
    out = torch.zeros(gan_output.shape[0], len_coeff)
    if if_cuda:
        out = out.cuda(gpu_id)
    # flatten it
    out[:, 0:936] = gan_output[:, :, :9, :13].reshape(-1, 936)
    # make it negative because we want to maximize it
    out = -out
    # make it the same size with the coefficients

    return out


def mip_sol_to_gan_out(gan_output, mip_sol):
    out = gan_output.clone()
    N = 9 * 13
    x = mip_sol[0, 0:8 * N].view(-1, 9, 13)
    out[0, :, :9, :13] = x
    return out
