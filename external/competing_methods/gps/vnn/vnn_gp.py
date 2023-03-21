import sys

sys.path.append('..')

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational.nearest_neighbor_variational_strategy import \
    NNVariationalStrategy
import torch
from gpytorch.distributions.multivariate_normal import MultivariateNormal as MVN


class VNNGP(ApproximateGP):
    def __init__(self, initial_inducing_response, inducing_points, likelihood, prior_kernel, k=256,
                 training_batch_size=256, jitter_val=None):
        m, d = inducing_points.shape
        self.m = m
        self.k = k
        variational_distribution = gpytorch.variational. \
            MeanFieldVariationalDistribution(m)
        start_dist = torch.distributions.MultivariateNormal(
            initial_inducing_response,
            torch.diag_embed(torch.ones_like(
                initial_inducing_response
            ) * .5))
        variational_distribution.initialize_variational_distribution(start_dist)
        variational_strategy = NNVariationalStrategy(
            self, inducing_points,
            variational_distribution,
            k=k,
            training_batch_size=training_batch_size)
        super(VNNGP, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = prior_kernel
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, prior=False, **kwargs):
        if x is not None:
            if x.dim() == 1:
                x = x.unsqueeze(-1)
        return self.variational_strategy(x=x, prior=False, **kwargs)

    def predict(self, x):
        return self.variational_strategy(x=x, prior=False).mean

    def log_score(self, x, f):
        return - self.variational_strategy(x=x, prior=False).log_prob(f)


def get_VNN(x_train, y_train, m, prior_kernel, likelihood, **kwargs):
    if kwargs.get("train_prior", True) is False:
        for parm in prior_kernel.parameters():
            parm.requires_grad = False
    if kwargs.get("train_lk", True) is False:
        for parm in likelihood.parameters():
            parm.requires_grad = False
    if kwargs.get("init_with_y", True):
        if type(likelihood) is gpytorch.likelihoods.GaussianLikelihood or \
        type(likelihood) is gpytorch.likelihoods.StudentTLikelihood:
            init_post_mean = y_train
        else:
            init_post_mean = y_train * 2.0 - 1.0
    else:
        init_post_mean = torch.zeros(y_train.shape)
    jitter_val = kwargs.get("jitter_val", 1e-2)
    # Note: one should use full training set as inducing points!
    model = VNNGP(
        initial_inducing_response=init_post_mean,
        inducing_points=x_train,
        likelihood=likelihood,
        prior_kernel=prior_kernel,
        k=m,
        training_batch_size=kwargs.get("batch_size", 128),
        jitter_val=jitter_val)
    model, likelihood = fit_vnn_gp(x_train, y_train, model, likelihood,
                                   **kwargs)
    return model, likelihood


def fit_vnn_gp(train_x, train_y, model, likelihood, **kwargs):
    num_batches = model.variational_strategy._total_training_batches
    model.train()
    opt_args = [{'params': model.parameters()}]
    if train_x.size(0) > 50000:
        lr = 0.001
        n_Epoch = 300
    else:
        lr = 0.01
        n_Epoch = 500
    optimizer = torch.optim.Adam(opt_args, lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(n_Epoch*0.75), int(n_Epoch*0.9)], gamma=0.1)
    # Our loss object. We're using the VariationalELBO
    mll = gpytorch.mlls.VariationalELBO(likelihood, model,
                                        num_data=train_y.size(0))
    epochs_iter = range(n_Epoch)
    for epoch in epochs_iter:
        minibatch_iter = range(num_batches)
        for i in minibatch_iter:
            optimizer.zero_grad()
            output = model(x=None)
            # Obtain the indices for mini-batch data
            current_training_indices = model.variational_strategy.current_training_indices
            # Obtain the y_batch using indices. It is important to keep the same order of train_x and train_y
            y_batch = train_y[current_training_indices].squeeze()
            loss = -mll(output, y_batch)
            # minibatch_iter.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model, likelihood
