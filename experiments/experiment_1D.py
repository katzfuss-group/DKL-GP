import torch
import gpytorch
import viva
import sys
from data_gen import Logit_y_gen, Student_y_gen, Normal_y_gen
from gpytorch.kernels import *
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
from gps.svigp import get_SVI
from gps.vnn import get_VNN
from gps.exact_gp import get_exact_gp
import yaml

sys.path.append('../')
from viva import LogitLikelihood
from viva import VIVAPy as VIVA, my_train_py as my_train
# from viva import VIVACpp as VIVA, my_train_cpp as my_train


def my_plot(x, f, y, fhat, fhat_sd, fn):
    fig_size = (6, 5)
    marker_size = 2
    font_size = 24
    line_width = 2.0
    alpha = 0.3
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    ax_objs = [ax.scatter(x, y, s=marker_size ** 2, alpha=alpha,
                          c='#d62728'),
               ax.plot(x, f, c='#1f77b4', linewidth=line_width)[0],
               ax.errorbar(x, fhat, 2.0 * fhat_sd, alpha=alpha, c='#ff7f0e')]
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.set_ylabel(None, fontsize=font_size)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
    if len(fn.split("_")) > 1:
        ax.set_xlabel(fn.split("_")[1], fontsize=font_size)
    fig.set_tight_layout(True)
    fig.savefig(fn)
    fig_legend = plt.figure(figsize=(4 * 3, 1))
    legend_names = ["y", "f", "f | y"]
    fig_legend.legend(ax_objs, legend_names, 'lower center',
                      fontsize=font_size, ncol=len(legend_names))
    fig_legend.savefig("1D_legend.pdf")


def my_plot_2(xTrain, xTest, fTest, yTrain, fhat, fhat_sd, legends, cols, fn):
    fig_size = (6, 5)
    marker_size = 2
    font_size = 16
    line_width = 1.0
    alpha = 0.3
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    ax_objs = [ax.scatter(xTrain, yTrain, s=marker_size ** 2, alpha=1.0,
                          c='#1f77b4'),
               ax.plot(xTest, fTest, c='#000000', linewidth=line_width)[0]]
    for i in range(len(legends)):
        # ax.fill_between(xTest, fhat[i] - 2.0 * fhat_sd[i],
        #                 fhat[i] + 2.0 * fhat_sd[i],
        #                 color=cols[i], alpha=alpha)
        ax.plot(xTest, fhat[i] - 2.0 * fhat_sd[i], c=cols[i],
                linewidth=line_width, alpha=1.0, linestyle='dashed')
        ax.plot(xTest, fhat[i] + 2.0 * fhat_sd[i], c=cols[i],
                linewidth=line_width, alpha=1.0, linestyle='dashed')
        ax_objs.append(ax.plot(xTest, fhat[i], c=cols[i], linewidth=line_width,
                               alpha=1.0)[0])
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
    if len(fn.split("_")) > 1:
        ax.set_xlabel(fn.split("_")[1], fontsize=font_size)
    fig.set_tight_layout(True)
    ax.set_ylabel(None, fontsize=font_size)
    ax.set_xlabel(None, fontsize=font_size)
    fig.savefig(fn)
    if len(legends) == 4:
        ax.set_ylim(-2, -1)
        ax.set_xlim(0.64, 0.76)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        tmp = fn.split(".")
        tmp[0] += "_zoomed"
        fig.savefig(".".join(tmp))
    legend_names = ["y", "f"]
    legend_names.extend(legends)
    fig_legend = plt.figure(figsize=(3.6 * len(legend_names), 1))
    fig_legend.legend(ax_objs, legend_names, 'lower center',
                      fontsize=26, ncol=len(legend_names))
    for line_obj in fig_legend.legends[0].get_lines():
        line_obj.set_linewidth(10.0)
    fig_legend.legends[0].legendHandles[0].set_sizes([100.0])
    if len(legends) == 4:
        fig_legend.savefig("1D_legend_combined.pdf")


def gen_data_mixed(nTrain, d, lengthscale=None, nugget4f=0.0, nugget4y=0.1,
                   nu=0.5,
                   seed=0, nTest=0, outputscale=1.0, y_gen=None):
    """
    Generate training data randomly and generate testing data on a grid
    """
    n = nTrain + nTest
    # Build locations (X)
    torch.manual_seed(seed)
    X_train = torch.rand(nTrain, d)
    nKnot = int(nTest ** (1 / d)) + 1
    edge = torch.linspace(0, 1, steps=nKnot)
    X_test = torch.flatten(
        torch.stack(torch.meshgrid([edge] * d, indexing='ij')),
        start_dim=1).t()[:nTest]
    X = torch.cat([X_train, X_test], dim=0)
    # Define a covariance kernel
    if lengthscale is None:
        lengthscale = torch.linspace(0.5, 2.5, d)
    logNugget4f = torch.log(torch.tensor(nugget4f))
    logNugget4y = torch.log(torch.tensor(nugget4y))
    K = ScaleKernel(MaternKernel(ard_num_dims=d, nu=nu))
    K.base_kernel.lengthscale = lengthscale
    K.outputscale = outputscale
    kernel4f = viva.Kernel(X, K, torch.exp(logNugget4f).item())
    # Build responses (y)
    covM4f = kernel4f(list(range(n)))
    GPObj4f = MultivariateNormal(torch.zeros(n), covM4f)
    f = GPObj4f.sample()
    if y_gen is None:
        y_gen = lambda x: x + \
                          torch.normal(torch.zeros(len(x)),
                                       nugget4y)
    y = y_gen(f)
    # y = y.sub(y.mean())
    return X, f, y, K, logNugget4f, logNugget4y


with open('setups.yaml', 'r') as config_file:
    tuning_parms = yaml.safe_load(config_file)
n_Epoch = tuning_parms['opt_VIVA'].pop("n_Epoch")[0]
torch.manual_seed(0)
sim_type_list = ["Normal", "Student", "Logit"]
loss_NLL_func = torch.nn.GaussianNLLLoss(reduction="sum", full=True)
marker_size = 2
alpha = 0.3
fig_size = (6, 5)
font_size = 16
line_width = 2.0
for sim_type in sim_type_list:
    if sim_type == "Normal":
        y_gen = Normal_y_gen(0.3)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = 0.3 ** 2
        classify = False
    elif sim_type == "Student":
        y_gen = Student_y_gen(2.0, 0.1)
        likelihood = gpytorch.likelihoods.StudentTLikelihood()
        likelihood.deg_free = 2.0
        likelihood.noise = 0.1 ** 2
        classify = False
    else:
        y_gen = Logit_y_gen()
        likelihood = LogitLikelihood()
        classify = True
    with torch.no_grad():
        n_train = 200
        n_test = 200
        d = 1
        rho = 2.0
        X, f, y, K, logNugget4f, logNugget4y = \
            gen_data_mixed(n_train, d, nugget4f=0.0001, lengthscale=0.1,
                           y_gen=y_gen, nugget4y=0.1, nu=1.5, nTest=n_test)
        X_train = X[:n_train]
        X_test = X[n_train:]
        f_test = f[n_train:]
        y_train = y[:n_train]
        y_test = y[n_train:]
        mu_init = torch.zeros(y.shape)
    model = VIVA(X, y, K, likelihood, rho, n_test=n_test,
                 classify=classify, use_ic0=True)
    my_train(model, n_Epoch=n_Epoch, **(tuning_parms['opt_VIVA']))
    mu_post, var_post = model.predict()
    sd_post = var_post.sqrt()
    mu_post_VIVA = mu_post.detach().clone()
    sd_post_VIVA = sd_post.detach().clone()
    my_plot(X_test[:, 0], f_test.detach(), y_test.detach(),
            mu_post.detach(), sd_post.detach(),
            "1D_VIVA_" + sim_type + ".pdf")
    # SVI
    try:
        m = int(model.prior_sparsity.size(1) / n_train)
        model, _ = get_SVI(
            X_train, y_train, m, K, likelihood,
            train_prior=False, train_lk=False)
        dist_post_f_tmp = model(X_test)
        mu_post_SVI = dist_post_f_tmp.mean.detach().clone()
        sd_post_SVI = dist_post_f_tmp.variance.sqrt().detach().clone()
        my_plot(X_test[:, 0], f_test.detach(), y_test.detach(),
                dist_post_f_tmp.mean.detach(),
                dist_post_f_tmp.variance.sqrt().detach(),
                "1D_SVI_" + sim_type + ".pdf")
    except gpytorch.utils.errors.NotPSDError:
        print("NotPSDError raised for SVIGP", flush=True)

    # VNN
    try:
        model, _ = get_VNN(
            X_train, y_train, m, K, likelihood,
            train_prior=False, train_lk=False)
        dist_post_f_tmp = model(X_test)
        mu_post_VNN = dist_post_f_tmp.mean.detach().clone()
        sd_post_VNN = dist_post_f_tmp.variance.sqrt().detach().clone()
        my_plot(X_test[:, 0], f_test.detach(), y_test.detach(),
                dist_post_f_tmp.mean.detach(),
                dist_post_f_tmp.variance.sqrt().detach(),
                "1D_VNN_" + sim_type + ".pdf")
    except gpytorch.utils.errors.NotPSDError:
        print("NotPSDError raised for VNN", flush=True)

    # exact_GP
    if sim_type == "Normal":
        try:
            model, _ = get_exact_gp(X_train, y_train, m, K, likelihood,
                                    train_prior=False, train_lk=True)
            dist_post_f_tmp = model(X_test)
            mu_post_exact_gp = dist_post_f_tmp.mean.detach().clone()
            sd_post_exact_gp = dist_post_f_tmp.variance.sqrt().detach().clone()
            my_plot(X_test[:, 0], f_test.detach(), y_test.detach(),
                    dist_post_f_tmp.mean.detach(),
                    dist_post_f_tmp.variance.sqrt().detach(),
                    "1D_exact_gp_" + sim_type + ".pdf")
        except gpytorch.utils.errors.NotPSDError:
            print("NotPSDError raised for exact_GP", flush=True)

    # plot joint
    if sim_type == "Normal":
        my_plot_2(X_train[:, 0], X_test[:, 0], f_test.detach(), y_train.detach(),
                  [mu_post_VIVA, mu_post_SVI, mu_post_VNN, mu_post_exact_gp],
                  [sd_post_VIVA, sd_post_SVI, sd_post_VNN, sd_post_exact_gp],
                  ["DKL", "SVI", "VNN", "DenseGP"],
                  ['#2ca02c', '#d62728', '#9467bd', '#cc9963'],
                  '1D_all_' + sim_type + ".pdf")
    else:
        my_plot_2(X_train[:, 0], X_test[:, 0], f_test.detach(), y_train.detach(),
                  [mu_post_VIVA, mu_post_SVI, mu_post_VNN],
                  [sd_post_VIVA, sd_post_SVI, sd_post_VNN],
                  ["DKL", "SVI", "VNN"],
                  ['#2ca02c', '#d62728', '#9467bd'],
                  '1D_all_' + sim_type + ".pdf")
