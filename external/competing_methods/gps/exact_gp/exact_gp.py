import torch
import gpytorch
import numpy as np
import time
import copy


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, X, y, K, likelihood, **kwargs):
        """
        Assumes the last nTest entries in X and y are for testing
        :param X: n X d
        :param y: n
        :param K: gpytorch kernel
        :param likelihood: likelihood of y | f
        """
        with torch.no_grad():
            super(ExactGPModel, self).__init__(X, y, likelihood)
            self.covar_module = K
            self.mean_module = gpytorch.means.ZeroMean()

    def forward(self, X):
        mean_X = self.mean_module(X)
        covar_X = self.covar_module(X)
        return gpytorch.distributions.MultivariateNormal(mean_X, covar_X)


def fit_exact_gp(train_x, train_y, model, likelihood, **kwargs):
    opt_args = [{'params': model.parameters()}]
    if train_x.size(0) > 50000:
        lr = 0.01
        n_Epoch = 30
    else:
        lr = 0.01
        n_Epoch = 50
    optimizer = torch.optim.Adam(opt_args, lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(n_Epoch*0.75), int(n_Epoch*0.9)], gamma=0.1)
    # Our loss object. We're using the VariationalELBO
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    epochs_iter = range(n_Epoch)
    for epoch in epochs_iter:
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        # minibatch_iter.set_postfix(loss=loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
    return model, likelihood


def get_exact_gp(x_train, y_train, m, prior_kernel, likelihood, **kwargs):
    model = ExactGPModel(x_train, y_train, prior_kernel, likelihood)
    model.train()
    likelihood.train()
    if kwargs.get("train_prior", True) is False:
        for parm in prior_kernel.parameters():
            parm.requires_grad = False
    else:
        for parm in prior_kernel.parameters():
            parm.requires_grad = True
    if kwargs.get("train_lk", True) is False:
        for parm in likelihood.parameters():
            parm.requires_grad = False
    else:
        for parm in likelihood.parameters():
            parm.requires_grad = True
    model, likelihood = fit_exact_gp(x_train, y_train, model, likelihood,
                                     **kwargs)
    model.eval()
    likelihood.eval()
    return model, likelihood
