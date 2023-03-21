import torch


def log_score_latent_GP(mean, marginal_var, y_true, likelihood=None,
                        num_samples=5000):
    if likelihood is None:
        raise Exception(
            "The generator function for y given f should not be None")
    n = mean.size(0)
    marginal_scale = marginal_var.sqrt()
    f_gen = torch.distributions.Normal(torch.zeros(n), torch.ones(n))
    f_samp = f_gen.sample([num_samples]) * marginal_scale + mean
    likelihood_fwd = likelihood.forward(f_samp)
    return - likelihood_fwd.log_prob(y_true.expand(num_samples, -1)).exp().\
        mean(dim=0).log().mean()
