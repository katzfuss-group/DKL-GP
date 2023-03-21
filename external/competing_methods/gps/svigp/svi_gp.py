import gpytorch
import torch
from gpytorch.distributions.multivariate_normal import MultivariateNormal as MVN
from gpytorch.models import ApproximateGP


class SVIGP(ApproximateGP):
    '''SVI-GP model (Hensman 2013)'''

    def __init__(self, inducing_points, prior_kernel):
        variational_distribution = gpytorch.variational. \
            NaturalVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        super(SVIGP, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = prior_kernel
        self.num_outputs = 1

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)

    def predict(self, x):
        return self.variational_strategy(x=x, prior=False).mean

    def log_score(self, x, f):
        return - self.variational_strategy(x=x, prior=False).log_prob(f)


def get_SVI(train_x, train_y, m, prior_kernel, likelihood, **kwargs):
    if kwargs.get("train_prior", True) is False:
        for parm in prior_kernel.parameters():
            parm.requires_grad = False
    if kwargs.get("train_lk", True) is False:
        for parm in likelihood.parameters():
            parm.requires_grad = False
    d = train_x.shape[-1]
    # m = 1024
    x_min = train_x.min()
    x_max = train_x.max()
    inducing_points = (torch.rand((m, d)) * (x_max - x_min) + x_min)
    model = SVIGP(inducing_points=inducing_points, prior_kernel=prior_kernel)
    fit_svi_gp(train_x, train_y, model, likelihood, **kwargs)
    return model, likelihood


def fit_svi_gp(train_x, train_y, model, likelihood, **kwargs):
    if train_x.size(0) > 50000:
        lr = 0.001
        n_Epoch = 300
    else:
        lr = 0.01
        n_Epoch = 500
    variational_ngd_optimizer = gpytorch.optim.NGD(
        model.variational_parameters(),
        num_data=train_y.size(0),
        lr=lr)
    hyperparameter_optimizer = torch.optim.Adam([
        {'params': model.hyperparameters()},
        {'params': likelihood.parameters()},
    ], lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        hyperparameter_optimizer,
        milestones=[int(n_Epoch * 0.75), int(n_Epoch * 0.9)],
        gamma=0.1)
    model.train()
    likelihood.train()
    mll = gpytorch.mlls.VariationalELBO(likelihood, model,
                                        num_data=train_y.size(0))
    dataset = torch.utils.data.TensorDataset(train_x, train_y)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=kwargs.get("batch_size", 128), shuffle=True)

    for epoch in range(n_Epoch):
        for x_batch, y_batch in loader:
            ### Perform NGD step to optimize variational parameters
            variational_ngd_optimizer.zero_grad()
            output = model(x_batch)
            loss = -(mll(output, y_batch))
            loss.backward()
            variational_ngd_optimizer.step()

            hyperparameter_optimizer.zero_grad()
            output = model(x_batch)
            loss = -(mll(output, y_batch))
            loss.backward()
            hyperparameter_optimizer.step()
        scheduler.step()

    return model, likelihood
