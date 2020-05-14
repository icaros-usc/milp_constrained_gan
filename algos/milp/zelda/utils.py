import torch


def gan_out_2_coefs(gan_output, len_coeff):
    """Use this function to translate the output of generator to the coefficients."""
    out = torch.zeros(len_coeff)
    # flatten it
    out[0:936] = torch.flatten(gan_output[:, :, :9, :13])
    # make it negative because we want to maximize it
    out = -out
    # make it the same size with the coefficients

    return out


def mip_sol_to_gan_out(gan_output, mip_sol):
    N = 9 * 13
    x = mip_sol[0, 0:8 * N].view(-1, 9, 13)
    gan_output[0, :, :9, :13] = x
