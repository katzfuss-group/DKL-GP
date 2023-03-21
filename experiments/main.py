import torch
import gpytorch
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import yaml
import copy
import sys
from gps.svigp import get_SVI
from gps.vnn import get_VNN
from gps.exact_gp import get_exact_gp
sys.path.append('../')
# from viva import VIVAPy as VIVA, FICPy as FIC, DiagPy as Diag, \
#     my_train_py as my_train
from viva import VIVACpp as VIVA, FICCpp as FIC, DiagCpp as Diag, \
    my_train_cpp as my_train
from viva.utils import LogitLikelihood


def work(task, X, f, y, n_test, rho, lk_name, method, **kwargs):
    n = len(y)
    d = X.size(1)
    n_train = n - n_test
    kwargs_cp = copy.deepcopy(kwargs)
    scenario = kwargs_cp.pop("scenario")
    scale_f_init = kwargs_cp[scenario].pop(
        'scale_f_init', kwargs_cp.pop('scale_f_init', None))
    kernel_name = kwargs_cp['kernel_name']
    kernel_parms = kwargs_cp['kernel_parms']
    kernel_vars_init = kwargs_cp[scenario]['kernel_vars_init']
    kernel_parms.update({'ard_num_dims': d})
    # data_name = kwargs_cp.pop('data_name', None)
    plot_mu_post = kwargs_cp.pop('plot_mu_post', False)
    seed_torch = kwargs_cp.pop('seed', 0)
    torch.manual_seed(seed_torch)
    K = gpytorch.kernels.ScaleKernel(
        getattr(gpytorch.kernels, kernel_name)(**kernel_parms))
    for var_name in kernel_vars_init.keys():
        setattr(K.base_kernel, var_name, kernel_vars_init[var_name])
    K.outputscale = scale_f_init
    if lk_name == "logit":
        likelihood = LogitLikelihood()
        classify = True
    elif lk_name == "student":
        likelihood = gpytorch.likelihoods.StudentTLikelihood()
        likelihood.deg_free = kwargs_cp[scenario].pop(
            'df_student_init', kwargs_cp.pop('df_student_init', None))
        likelihood.noise = kwargs_cp[scenario].pop(
            'noise_y_init', kwargs_cp.pop('noise_y_init', None))
        classify = False
    elif lk_name == "normal":
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = kwargs_cp[scenario].pop(
            'noise_y_init', kwargs_cp.pop('noise_y_init', None))
        classify = False
    else:
        raise ValueError("Incorrect input lk_name")

    if method in ["FIC", "Diag", "VIVA"]:
        if task == "insample_f_scores":
            # ic0
            with torch.no_grad():
                model = globals()[method](X, y, K, likelihood, rho,
                                          n_test=n_test,
                                          classify=classify, use_ic0=True)
            rmse_ic0 = (
                model.mu_post - f[:n_train][model.order[:model.n_train]]). \
                square().mean().sqrt().detach()
            loss_NLL_func = torch.nn.GaussianNLLLoss()
            mu_post = torch.zeros(y.shape)
            mu_post[model.order] = model.mu_post
            V_COO = torch.sparse_coo_tensor(model.V_indices, model.V_values)
            V_dense = V_COO.to_dense()
            covM_maxmin_order = torch.linalg.inv(V_dense @ V_dense.t())
            var_post = torch.zeros(y.shape)
            var_post[model.order] = covM_maxmin_order.diag()
            NLL_ic0 = loss_NLL_func(mu_post, f[:n_train], var_post).detach()
            if plot_mu_post and method == "VIVA" and lk_name == "normal":
                my_plot_3(f[:n_train], mu_post.detach(),
                          "true posterior mean", "initial posterior mean",
                          f"true_init_nu_{rho}.pdf")
        epoch_nums = kwargs_cp[scenario].pop(
            'n_Epoch', kwargs_cp.pop('n_Epoch', None))
        for i in range(len(epoch_nums)):
            with torch.no_grad():
                model = globals()[method](X, y, K, likelihood, rho,
                                          n_test=n_test,
                                          classify=classify, use_ic0=True)
                m = int(model.prior_sparsity_train.size(1) / n_train)
                print(f"n = {n}, d = {d}, rho = {rho}: m = {m}", flush=True)
            my_train(model, n_Epoch=epoch_nums[i], **kwargs_cp)
        if task == "insample_f_scores":
            rmse_trained = (model.mu_post -
                            f[:n_train][model.order[:model.n_train]]). \
                square().mean().sqrt().detach()
            loss_NLL_func = torch.nn.GaussianNLLLoss()
            mu_post = torch.zeros(y.shape)
            mu_post[model.order] = model.mu_post
            V_COO = torch.sparse_coo_tensor(model.V_indices, model.V_values)
            V_dense = V_COO.to_dense()
            covM_maxmin_order = torch.linalg.inv(V_dense @ V_dense.t())
            var_post = torch.zeros(y.shape)
            var_post[model.order] = covM_maxmin_order.diag()
            NLL_trained = loss_NLL_func(mu_post, f[:n_train], var_post).detach()
            if plot_mu_post and method == "VIVA" and lk_name == "normal":
                my_plot_3(f[:n_train], mu_post.detach(),
                          "true posterior mean", "est'd posterior mean",
                          f"true_est_nu_{rho}.pdf")
            return [rmse_ic0, rmse_trained, NLL_ic0, NLL_trained]
        elif task == "outsample_f_scores":
            loss_MSE_func = torch.nn.MSELoss()
            loss_NLL_func = torch.nn.GaussianNLLLoss()
            f_mean_pred, f_var_pred = model.predict()
            return [loss_MSE_func(f_mean_pred, f[n_train:]).sqrt().detach(),
                    loss_NLL_func(f_mean_pred, f[n_train:],
                                  f_var_pred).detach()]
        elif task == "outsample_y_scores":
            loss_MSE_func = torch.nn.MSELoss()
            loss_NLL_func = torch.nn.GaussianNLLLoss()
            f_mean_pred, f_var_pred = model.predict()
            if type(likelihood) == LogitLikelihood:
                y_mean_pred = 1.0 / (1.0 + (- f_mean_pred.detach()).exp())
                y_var_pred = torch.zeros(y_mean_pred.shape) * torch.nan
            else:
                y_mean_pred = f_mean_pred.detach()
                y_var_pred = (f_var_pred + likelihood.noise).detach()
            # pd.DataFrame(y_mean_pred).to_csv(
            #     method + '_' + data_name + ".csv")
            return [loss_MSE_func(y_mean_pred, y[n_train:]).sqrt().detach(),
                    loss_NLL_func(y_mean_pred, y[n_train:],
                                  y_var_pred).detach()]
        else:
            raise ValueError(f"task = {task} is undefined")
    else:
        try:
            with torch.no_grad():
                model = VIVA(X, y, K, likelihood, rho, n_test=n_test,
                             classify=classify, use_ic0=False)
            m = int(model.prior_sparsity_train.size(1) / n_train)
            print(f"n = {n}, d = {d}, rho = {rho}: m = {m}", flush=True)
            model, _ = globals()["get_" + method](
                X[:n_train], y[:n_train], m, K, likelihood,
                train_prior=kwargs_cp[scenario].pop(
                    'train_prior', kwargs_cp.pop('train_prior', None)),
                train_lk=kwargs_cp[scenario].pop(
                    'train_lk', kwargs_cp.pop('train_lk', None)))
            loss_MSE_func = torch.nn.MSELoss()
            loss_NLL_func = torch.nn.GaussianNLLLoss()
            if task == "insample_f_scores":
                dist_post_f = model(X[:n_train])
                return [loss_MSE_func(dist_post_f.mean,
                                      f[:n_train]).sqrt().detach(),
                        loss_NLL_func(dist_post_f.mean, f[:n_train],
                                      dist_post_f.variance).detach()]
            elif task == "outsample_f_scores":
                dist_post_f = model(X[n_train:])
                return [loss_MSE_func(dist_post_f.mean,
                                      f[n_train:]).sqrt().detach(),
                        loss_NLL_func(dist_post_f.mean, f[n_train:],
                                      dist_post_f.variance).detach()]
            elif task == "outsample_y_scores":
                dist_post_f = model(X[n_train:])
                f_mean_pred = dist_post_f.mean
                f_var_pred = dist_post_f.variance
                if type(likelihood) == LogitLikelihood:
                    y_mean_pred = 1.0 / (1.0 + (- f_mean_pred).exp()).detach()
                    y_var_pred = torch.zeros(y_mean_pred.shape) * torch.nan
                else:
                    y_mean_pred = f_mean_pred.detach()
                    y_var_pred = (f_var_pred + likelihood.noise).detach()
                # pd.DataFrame(y_mean_pred).to_csv(
                #     method + '_' + data_name + ".csv")
                return [loss_MSE_func(y_mean_pred, y[n_train:]).sqrt().detach(),
                        loss_NLL_func(y_mean_pred, y[n_train:],
                                      y_var_pred).detach()]
            else:
                raise ValueError(f"task = {task} is undefined")
        except gpytorch.utils.errors.NotPSDError:
            print(f"Not positive definite found for {method}", flush=True)
            return [torch.nan, torch.nan]
        except RuntimeError as err:
            print(err, flush=True)
            return [torch.nan, torch.nan]


def find_unique_X(X, threshold):
    n = len(X)
    mask = torch.ones(n, dtype=torch.bool)
    with torch.no_grad():
        for i in range(n - 1, 0, -1):
            if torch.cdist(X[i:i + 1], X[:i]).min() < threshold:
                mask[i] = False
    return mask


def read_data(dataset_root, data_name, task, **kwargs):
    n_train_max = kwargs.get("n_train_max", None)
    n_test_max = kwargs.get("n_test_max", None)
    scale_X = kwargs.get("scale_X", False)
    scale_y = kwargs.get("scale_y", False)
    X_train = torch.from_numpy(
        pd.read_csv(
            os.path.join(dataset_root, data_name, "train",
                         'x.csv')).to_numpy()). \
        type(torch.float)
    if task != "outsample_y_scores":
        f_train = torch.from_numpy(
            pd.read_csv(
                os.path.join(dataset_root, data_name, "train",
                             'f.csv')).to_numpy()). \
            type(torch.float).squeeze()
    y_train = torch.from_numpy(
        pd.read_csv(
            os.path.join(dataset_root, data_name, "train",
                         'y.csv')).to_numpy()). \
        type(torch.float).squeeze()
    if n_train_max is not None:
        X_train = X_train[:min(n_train_max, len(X_train))]
        y_train = y_train[:min(n_train_max, len(y_train))]
        if "f_train" in locals():
            f_train = f_train[:min(n_train_max, len(f_train))]
    if task == "insample_f_scores":
        X = X_train
        y = y_train
        f = f_train
        n_test = 0
    else:
        X_test = torch.from_numpy(
            pd.read_csv(
                os.path.join(dataset_root, data_name, "test",
                             'x.csv')).to_numpy()). \
            type(torch.float)
        if task != "outsample_y_scores":
            f_test = torch.from_numpy(
                pd.read_csv(
                    os.path.join(dataset_root, data_name, "test",
                                 'f.csv')).to_numpy()). \
                type(torch.float).squeeze()
        y_test = torch.from_numpy(
            pd.read_csv(
                os.path.join(dataset_root, data_name, "test",
                             'y.csv')).to_numpy()). \
            type(torch.float).squeeze()
        if n_test_max is not None:
            X_test = X_test[:min(n_test_max, len(X_test))]
            y_test = y_test[:min(n_test_max, len(y_test))]
            if "f_test" in locals():
                f_test = f_test[:min(n_test_max, len(f_test))]
        X = torch.cat([X_train, X_test], dim=0)
        y = torch.cat([y_train, y_test])
        if task != "outsample_y_scores":
            f = torch.cat([f_train, f_test])
        else:
            f = None
        n_test = len(y_test)
    n_train = len(y) - n_test
    if scale_X:
        X = (X - X.min(0, keepdim=True)[0]) / \
            (X.max(0, keepdim=True)[0] - X.min(0, keepdim=True)[0])
        X_std = X.std(dim=0)
        if data_name == "covtype":
            X = X[:, X_std.ge(0.01)]
        else:
            X = X[:, X_std.ge(0.0001)]
    if scale_y:
        y = (y - y.mean()) / y.std()
    if data_name == "covtype":
        mask_all = find_unique_X(X, 0.01)
    else:
        mask_all = find_unique_X(X, 0.001)
    X = X[mask_all]
    y = y[mask_all]
    if task != "outsample_y_scores":
        f = f[mask_all]
    n_train = mask_all[:n_train].sum()
    n_test = mask_all.sum() - n_train

    return X, y, f, n_test


def my_plot(task, rslt_fn, method_list, rho_list, nrep):
    results = torch.load(rslt_fn)
    score_FIC_init = torch.zeros(len(rho_list))  # score at ic0 initialization
    score_FIC_train = torch.zeros(
        len(rho_list))  # score from opt ELBO, equivalent to backward KL
    score_Diag_init = torch.zeros(len(rho_list))
    score_Diag_train = torch.zeros(len(rho_list))
    score_SVI_train = torch.zeros(len(rho_list))
    score_VNN_train = torch.zeros(len(rho_list))
    score_VIVA_init = torch.zeros(len(rho_list))
    score_VIVA_train = torch.zeros(len(rho_list))
    legend_names = ["DKL-G", "DKL-D", "DKL", "SVI", "VNN"]
    n_score = 2
    score_names = {0: "RMSE", 1: "NLL"}
    for j in range(n_score):
        idx = 0
        for i in range(len(rho_list)):
            for method in method_list:
                if (method in ["FIC", "Diag", "VIVA"]) and \
                    task == "insample_f_scores":
                    array_tmp = np.concatenate(results[idx:idx + nrep])
                    locals()["score_" + method + "_init"][i] = array_tmp[
                        range(j * 2, nrep * 2 * n_score,
                              2 * n_score)].mean().item()
                    locals()["score_" + method + "_train"][i] = array_tmp[
                        range(j * 2 + 1, nrep * 2 * n_score,
                              2 * n_score)].mean().item()
                else:
                    array_tmp = np.concatenate(results[idx:idx + nrep])
                    locals()["score_" + method + "_train"][
                        i] = array_tmp[j:nrep * n_score:n_score].mean().item()
                idx += nrep
        fig_size = (6, 5)
        marker_size = 10
        font_size = 24
        line_width = 2.0
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        ax_objs = []
        if task == "insample_f_scores":
            ax.plot(rho_list, score_VIVA_init, c='#2ca02c',
                    linestyle="dotted",
                    linewidth=line_width)
        ax_objs.append(ax.scatter(rho_list, score_FIC_train, s=marker_size ** 2,
                                  c='#1f77b4',
                                  marker="+"))
        ax_objs.append(
            ax.scatter(rho_list, score_Diag_train, s=marker_size ** 2,
                       c='#ff7f0e',
                       marker="o"))
        ax_objs.append(
            ax.scatter(rho_list, score_VIVA_train, s=marker_size ** 2,
                       c='#2ca02c',
                       marker="*"))
        ax_objs.append(ax.scatter(rho_list, score_SVI_train, s=marker_size ** 2,
                                  c='#d62728',
                                  marker="^"))
        ax_objs.append(ax.scatter(rho_list, score_VNN_train, s=marker_size ** 2,
                                  c='#9467bd',
                                  marker="v"))
        fig_legend = plt.figure(figsize=(2.4 * len(ax_objs), 1))
        fig_legend.legend(ax_objs, legend_names, 'lower center',
                          fontsize=font_size, ncol=len(ax_objs))
        ax.set_xlabel("rho", fontsize=font_size)
        ax.set_ylabel(score_names[j] + "-" + rslt_fn.split("_")[-3],
                   fontsize=font_size)
        ax.plot(rho_list, score_FIC_train, linestyle="dashed",
                 c='#1f77b4')
        ax.plot(rho_list, score_Diag_train, linestyle="dashed",
                 c='#ff7f0e')
        ax.plot(rho_list, score_VIVA_train, linestyle="dashed",
                 c='#2ca02c')
        ax.plot(rho_list, score_SVI_train, linestyle="dashed",
                 c='#d62728')
        ax.plot(rho_list, score_VNN_train, linestyle="dashed",
                 c='#9467bd')
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        if score_names[j] == "NLL":
            ax.set_ylim(top=5.0)
        fig.set_tight_layout(True)
        fig.savefig(f"{score_names[j]}_" + rslt_fn[:-3] + ".pdf")
        fig_legend.set_tight_layout(True)
        fig_legend.savefig("legend.pdf")


def my_plot_2(task, rslt_fns, method_list, rho_list, nrep):
    score_FIC_init = torch.zeros(len(rho_list))  # score at ic0 initialization
    score_FIC_train = torch.zeros(
        len(rho_list))  # score from opt ELBO, equivalent to backward KL
    score_Diag_init = torch.zeros(len(rho_list))
    score_Diag_train = torch.zeros(len(rho_list))
    score_SVI_train = torch.zeros(len(rho_list))
    score_VNN_train = torch.zeros(len(rho_list))
    score_VIVA_init = torch.zeros(len(rho_list))
    score_VIVA_train = torch.zeros(len(rho_list))
    legend_names = ["DKL-G", "DKL-D", "DKL", "SVI", "VNN"]
    n_score = 2
    score_names = {0: "RMSE", 1: "NLL"}
    fig_size = (18, 10)
    marker_size = 10
    font_size = 24
    line_width = 2.0
    fig, axs = plt.subplots(2, 3, sharex=True, figsize=fig_size)
    idx_ax = 0
    for j in range(n_score):
        for rslt_fn in rslt_fns:
            results = torch.load(rslt_fn)
            ax = axs[int(idx_ax / 3), int(idx_ax % 3)]
            idx = 0
            for i in range(len(rho_list)):
                for method in method_list:
                    if (method in ["FIC", "Diag", "VIVA"]) and \
                        task == "insample_f_scores":
                        array_tmp = np.concatenate(results[idx:idx + nrep])
                        locals()["score_" + method + "_init"][i] = array_tmp[
                            range(j * 2, nrep * 2 * n_score,
                                  2 * n_score)].mean().item()
                        locals()["score_" + method + "_train"][i] = array_tmp[
                            range(j * 2 + 1, nrep * 2 * n_score,
                                  2 * n_score)].mean().item()
                    else:
                        array_tmp = np.concatenate(results[idx:idx + nrep])
                        locals()["score_" + method + "_train"][
                            i] = array_tmp[j:nrep * n_score:n_score].mean().item()
                    idx += nrep
            if task == "insample_f_scores":
                ax.plot(rho_list, score_VIVA_init, c='#2ca02c',
                        linestyle="dotted",
                        linewidth=line_width)
            ax_objs = []
            ax_objs.append(ax.scatter(rho_list, score_FIC_train, s=marker_size ** 2,
                                      c='#1f77b4',
                                      marker="+"))
            ax_objs.append(
                ax.scatter(rho_list, score_Diag_train, s=marker_size ** 2,
                           c='#ff7f0e',
                           marker="o"))
            ax_objs.append(
                ax.scatter(rho_list, score_VIVA_train, s=marker_size ** 2,
                           c='#2ca02c',
                           marker="*"))
            ax_objs.append(ax.scatter(rho_list, score_SVI_train, s=marker_size ** 2,
                                      c='#d62728',
                                      marker="^"))
            ax_objs.append(ax.scatter(rho_list, score_VNN_train, s=marker_size ** 2,
                                      c='#9467bd',
                                      marker="v"))
            if idx_ax > 2:
                ax.set_xlabel("rho", fontsize=font_size)
            else:
                ax.set_xlabel(None)
            if idx_ax == 0:
                ax.set_ylabel(score_names[j], fontsize=font_size)
            elif idx_ax == 3:
                ax.set_ylabel(score_names[j], fontsize=font_size)
            else:
                ax.set_ylabel(None)
            if idx_ax == 0:
                ax.set_title("Gaussian", fontsize=font_size)
            if idx_ax == 1:
                ax.set_title("Student-$t$", fontsize=font_size)
            if idx_ax == 2:
                ax.set_title("Bernoulli-logit", fontsize=font_size)
            ax.plot(rho_list, score_FIC_train, linestyle="dashed",
                     c='#1f77b4')
            ax.plot(rho_list, score_Diag_train, linestyle="dashed",
                     c='#ff7f0e')
            ax.plot(rho_list, score_VIVA_train, linestyle="dashed",
                     c='#2ca02c')
            ax.plot(rho_list, score_SVI_train, linestyle="dashed",
                     c='#d62728')
            ax.plot(rho_list, score_VNN_train, linestyle="dashed",
                     c='#9467bd')
            ax.tick_params(axis='both', which='major', labelsize=font_size)
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
            if score_names[j] == "NLL":
                ax.set_ylim(top=5.0)
            if idx_ax == 0:
                fig_legend = plt.figure(figsize=(2.4 * len(ax_objs), 1))
                fig_legend.legend(ax_objs, legend_names, 'lower center',
                                  fontsize=font_size, ncol=len(ax_objs))
                fig.set_tight_layout(True)
                fig_legend.set_tight_layout(True)
                fig_legend.savefig("legend.pdf")
            idx_ax += 1
    fig_fn = rslt_fn[:-3].split("_")
    fig_fn.pop(3)
    fig.savefig("-".join(fig_fn) + ".pdf")


def my_plot_3(x, y, xLabel, yLabel, fn):
    fig_size = (6, 5)
    marker_size = 10
    font_size = 24
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    ax.scatter(x, y,  s=marker_size ** 2)
    ax.set_xlabel(xLabel, fontsize=font_size)
    ax.set_ylabel(yLabel, fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.axline([x.min().item(), x.min().item()],
              [x.max().item(), x.max().item()],
              linestyle="dashed", c='red')
    fig.set_tight_layout(True)
    fig.savefig(fn)


if __name__ == "__main__":
    with open('setups.yaml', 'r') as config_file:
        tuning_parms = yaml.safe_load(config_file)
    task = tuning_parms['run']['task']
    scenario = tuning_parms['run']['scenario']
    n_cross_valid = tuning_parms['run'].pop('n_cross_valid', 1)
    read_data_kwargs = {}
    if scenario == "real_data":
        data_name = tuning_parms[scenario]['data_name']
        lk_name = tuning_parms[scenario]['data_lk'][data_name]
        rho_list = tuning_parms[scenario]['data_rho_list'].get(
            data_name, tuning_parms['run']['rho_list'])
        data_max_n = tuning_parms[scenario]['data_max_n'].get(data_name, None)
        if data_max_n is not None:
            read_data_kwargs["n_train_max"] = data_max_n["n_train_max"]
            read_data_kwargs["n_test_max"] = data_max_n["n_test_max"]
        read_data_kwargs["scale_X"] = True
        if lk_name != "logit":
            read_data_kwargs["scale_y"] = True
    else:
        lk_name = tuning_parms['run']['lk_name']
        rho_list = tuning_parms['run']['rho_list']
    method_list = tuning_parms['run']['method_list']
    dataset_root = tuning_parms['run']['dataset_root']
    if task == "insample_f_scores":
        plot_mu_post = True
    else:
        plot_mu_post = False
    if scenario.count("simulation") > 0:
        n = tuning_parms[scenario].get('n', None)
        d = tuning_parms[scenario].get('d', None)
        kernel_name = tuning_parms[scenario]['kernel_name']
        kernel_parms = tuning_parms[scenario]['kernel_parms']
        seed_list = tuning_parms['run']['seed_list']
        data_names = [f"{kernel_name}_{lk_name}_n{n}_d{d}_seed{seed}"
                      for seed in seed_list]
        nrep = len(seed_list)
        rslt_fn = f"{task}_{lk_name}_{n}_{d}.pt"
        rslt_fns = [f"{task}_{lk_name}_{n}_{d}.pt"
                    for lk_name in ["normal", "student", "logit"]]
    else:
        data_names = [tuning_parms[scenario]["data_name"]]
        nrep = 1
        rslt_fn = task + "_" + tuning_parms[scenario]["data_name"] + ".pt"
        kernel_name = 'MaternKernel'
        kernel_parms = {'nu': 1.5}
    tuning_parms['opt_VIVA'].update({
        'kernel_name': kernel_name,
        'kernel_parms': kernel_parms
    })

    results = []
    for rho in rho_list:
        for method in method_list:
            for data_name in data_names:
                X, y, f, n_test = read_data(dataset_root, data_name, task,
                                            **read_data_kwargs)
                result_cv = []
                for k in range(n_cross_valid):
                    n = X.shape[0]
                    ind = (torch.arange(n) + k * int(n / n_cross_valid)) % n
                    X = X[ind]
                    y = y[ind]
                    if f is not None:
                        f = f[ind]
                    time0 = time.perf_counter()
                    result = work(
                        task, X, f, y, n_test, rho, lk_name, method,
                        scenario=scenario, data_name=data_name,
                        plot_mu_post=plot_mu_post,
                        **(tuning_parms['opt_VIVA']))
                    time1 = time.perf_counter()
                    print(f"{method} used {time1 - time0} "
                          f"seconds at rho = {rho}", flush=True)
                    result_cv.append(result)
                if len(result_cv) > 1:
                    print(result_cv, flush=True)
                results.append(torch.tensor(result_cv).mean(dim=0).tolist())
    torch.save(results, rslt_fn)
    print(results)

    my_plot(task, rslt_fn, method_list, rho_list, nrep)
    if task in ["outsample_f_scores", "insample_f_scores"] and \
        all([os.path.isfile(x) for x in rslt_fns]):
        my_plot_2(task, rslt_fns, method_list, rho_list, nrep)
