import torch


def crps(y_samp, y_samp_2, y_true):
    """
    Compute sample-based CRPS based on eq. (NRG) of
    https://link.springer.com/article/10.1007/s11004-017-9709-7
    :param y_samp: of shape [num_samples, n]
    :param y_samp_2: of shape [num_samples, n]
    :param y_true: of shape [n] or [1, n]
    :return:
    """
    crps_part1 = (y_samp - y_true.squeeze()).abs().mean(dim=0)
    crps_part2 = (y_samp - y_samp_2).abs().mean(dim=0) * 0.5
    rslt = crps_part1 - crps_part2
    return rslt


def crps_latent_GP(mean, marginal_var, y_true, y_gen=None, num_samples=5000):
    n = mean.size(0)
    marginal_scale = marginal_var.sqrt()
    f_gen = torch.distributions.Normal(torch.zeros(n), torch.ones(n))
    f_samp = f_gen.sample([num_samples]) * marginal_scale + mean
    f_samp_2 = f_gen.sample([num_samples]) * marginal_scale + mean
    if y_gen is None:
        y_samp = f_samp
        y_samp_2 = f_samp_2
    else:
        y_samp = y_gen(f_samp)
        y_samp_2 = y_gen(f_samp_2)
    return crps(y_samp, y_samp_2, y_true).mean()
