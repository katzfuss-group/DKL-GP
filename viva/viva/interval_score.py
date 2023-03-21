import torch


def interval(y_samp, y_true, alpha=0.05):
    """
    Compute sample-based CRPS based on eq. (NRG) of
    https://link.springer.com/article/10.1007/s11004-017-9709-7
    :param y_samp: of shape [num_samples, n]
    :param y_true: of shape [n] or [1, n]
    :return:
    """
    l = torch.quantile(y_samp, alpha / 2.0, dim=0)
    r = torch.quantile(y_samp, 1 - alpha / 2.0, dim=0)
    rslt = (r - l) + 2.0 / (1.0 - alpha) * ((l - y_true) * (y_true.lt(l)) +
                                            (y_true - r) * (y_true.gt(r)))
    return rslt


def interval_latent_GP(mean, marginal_var, y_true, y_gen=None, alpha=0.05,
                       num_samples=5000):
    n = mean.size(0)
    marginal_scale = marginal_var.sqrt()
    f_gen = torch.distributions.Normal(torch.zeros(n), torch.ones(n))
    f_samp = f_gen.sample([num_samples]) * marginal_scale + mean
    if y_gen is None:
        y_samp = f_samp
    else:
        y_samp = y_gen(f_samp)
    return interval(y_samp, y_true).mean()
