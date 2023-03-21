import torch
import gpytorch
import numpy as np
import time
import copy
from itertools import compress
from .utils import Logit_dydf, Normal_dydf, find_csc, find_inv_chol, \
    update_mean_inv_chol, LogitLikelihood
from .MM_order import order_X_f_y
from .kernel import Kernel
from .Laplace_init import find_f_map_Laplace
from .predict import predict_f
from .Ancestor import Ancestor
from .utils import MC_likelihood


def compute_ELBO(y, mean, mean_post,
                    U_sub_dict, V_dict, V_values,
                    ancestor: Ancestor or None, likelihood=None, noise=None,
                    U=None, V=None, sparse=True, mini_indices=None,
                    timing=False):
    """
    Computes and returns the VIVA ELBO.
    Inputs:
    y - observations
    mean - prior mean of f
    mean_post - posterior mean of f
    U_sub_dict - sub columns of prior inverse upper Cholesky corresponding to
        mini_indices as a dictionary
    V_dict - match 1D hash to the index in V_COO.values()
    V_values - V_COO.values()
    ancestor - the ancestor set
    likelihood - the log-likelihood of the model
    noise - assume that y = f + N(0, noise) if noise is not None.
            Otherwise, Monte Carlo is used for the expectation
    U, V - dense representation of the sparse matrices, used when
        sparse=False for testing purpose
    sparse - a bool flag for switching between using dense and sparse storage
    mini_indices - mini-batch indices in range(n)
    """
    time_trisolve = 0.0
    time_sub_matrix = 0.0
    n = len(mean)
    if mini_indices is None:
        mini_indices = torch.arange(n)
    batch_size = len(mini_indices)
    if not sparse:
        chol_post = torch.triangular_solve(torch.eye(n), V,
                                           upper=True).solution.t()
        marginalVarPost = chol_post[mini_indices].square().sum(dim=1)
        logDet = U[mini_indices, mini_indices].log().sum() - \
                 V[mini_indices, mini_indices].log().sum()
        innerMean = - torch.mm(U[:, mini_indices].t(),
                               (mean - mean_post).unsqueeze(1)).square(). \
            sum() / 2.0
        innerCov = - torch.triangular_solve(U[:, mini_indices], V, upper=True). \
            solution.square().sum() / 2.0
    else:
        # compute marginal var, inner prod with cov mat, and inner prod with
        # mean difference
        ances_ind = [ancestor.get_ancestor_idx(j).numpy()
                     for j in mini_indices]
        ances_ind_len = [len(x) for x in ances_ind]
        ances_ind_len_max = max(ances_ind_len)

        time0 = time.perf_counter()

        V_sub = torch.eye(ances_ind_len_max).unsqueeze(0). \
            repeat(batch_size, 1, 1)
        V_sub_fullind = np.indices(
            [batch_size, ances_ind_len_max, ances_ind_len_max])
        V_sub_mask = np.logical_and(
            V_sub_fullind[1] >
            (ances_ind_len_max - np.array(ances_ind_len) - 1)[:, None, None],
            V_sub_fullind[2] >
            (ances_ind_len_max - np.array(ances_ind_len) - 1)[:, None, None],
        )
        V_sub_1Dhash = np.concatenate(
            [(x * n).repeat(len(x)) + np.tile(x, len(x))
             for x in ances_ind]).tolist()
        V_sub_1Dhash_mask = [i in V_dict for i in V_sub_1Dhash]
        V_sub_1Dhash = list(compress(V_sub_1Dhash, V_sub_1Dhash_mask))
        V_sub_val = V_values[[V_dict[i] for i in V_sub_1Dhash]]
        V_sub_flatten = torch.flatten(V_sub)
        V_sub_flatten_ind = torch.arange(len(V_sub_flatten)).view(V_sub.shape)
        V_sub_flatten_ind = V_sub_flatten_ind[torch.from_numpy(V_sub_mask)][
            V_sub_1Dhash_mask]
        V_sub_flatten[V_sub_flatten_ind] = V_sub_val

        U_sub = torch.zeros(batch_size, ances_ind_len_max)
        U_sub_mask = np.arange(ances_ind_len_max - 1, -1, -1) < \
                     np.array(ances_ind_len)[:, None]
        U_sub_1Dhash = np.concatenate(
            [(x * n) + x[-1] for x in ances_ind]).tolist()
        U_sub_1Dhash_mask = [i in U_sub_dict for i in U_sub_1Dhash]
        U_sub_1Dhash = list(compress(U_sub_1Dhash, U_sub_1Dhash_mask))
        U_sub_val = torch.stack([U_sub_dict[i] for i in U_sub_1Dhash])
        U_sub_flatten = torch.flatten(U_sub)
        U_sub_flatten_ind = torch.arange(len(U_sub_flatten)).view(U_sub.shape)
        U_sub_flatten_ind = U_sub_flatten_ind[torch.from_numpy(U_sub_mask)][
            U_sub_1Dhash_mask]
        U_sub_flatten[U_sub_flatten_ind] = U_sub_val

        mean_diff_sub = torch.zeros(batch_size, ances_ind_len_max)
        mean_diff_sub[torch.from_numpy(U_sub_mask)] = torch.cat(
            [mean[i] - mean_post[i] for i in ances_ind])

        time1 = time.perf_counter()
        time_sub_matrix += time1 - time0

        ej = torch.zeros(batch_size, ances_ind_len_max)
        ej[:, -1] = 1

        time0 = time.perf_counter()
        marginalVarPost = \
            torch.linalg.solve_triangular(V_sub, ej.unsqueeze(2), upper=True). \
                square().sum(dim=[1, 2])
        innerCov = \
            - torch.linalg.solve_triangular(V_sub, U_sub.unsqueeze(2),
                                            upper=True). \
                square().sum() / 2.0
        time1 = time.perf_counter()
        time_trisolve += time1 - time0
        # compute inner prod with mean
        #   the line below computes the same thing but Adam currently does not support
        #   sparse gradient:
        #   innerMean = - torch.sparse.mm(U.t(), (mean - mean_post).
        #       unsqueeze(1)).square().sum() / 2.0
        time0 = time.perf_counter()
        innerMean = - torch.bmm(U_sub.unsqueeze(1), mean_diff_sub.unsqueeze(2)). \
            square().sum() / 2.0
        time1 = time.perf_counter()
        time_innerMean = time1 - time0
        # compute log determinants
        time0 = time.perf_counter()
        mini_indices_list = mini_indices.tolist()
        logDet = torch.stack(
            [U_sub_dict[i * n + i] for i in mini_indices_list]). \
                     log().sum() - \
                 V_values[[V_dict[i * n + i] for i in mini_indices_list]].abs(). \
                     log().sum()
        time1 = time.perf_counter()
        time_logDet = time1 - time0
    # compute the 1-d expectation in ELBO
    time0 = time.perf_counter()
    if noise is None:
        expected_likelihood = MC_likelihood(y[mini_indices],
                                            mean_post[mini_indices],
                                            marginalVarPost, likelihood)
    else:
        expected_likelihood = - batch_size / 2.0 * torch.log(
            2.0 * torch.pi * noise) - \
                              ((y[mini_indices] - mean_post[
                                  mini_indices]).square().sum() +
                               marginalVarPost.sum()) / 2.0 / noise
    time1 = time.perf_counter()
    time_expected_likelihood = time1 - time0
    if timing:
        return logDet + innerMean + innerCov + expected_likelihood, \
               time_trisolve, time_sub_matrix, time_innerMean, time_logDet, \
               time_expected_likelihood
    else:
        return logDet + innerMean + innerCov + expected_likelihood
    

class Base(torch.nn.Module):
    def __init__(self, X, y, K, likelihood, rho, **kwargs):
        """
        Assumes the last nTest entries in X and y are for testing
        :param X: n X d
        :param y: n
        :param K: gpytorch kernel
        :param likelihood: likelihood of y | f
        :param rho: no smaller than 1.0
        Optional args:
        :param n_test: number of testing locs
        :param use_ic0: whether to use Laplacian init
        :param mu0: whether to initialize mean_post with mu0
        :param nugget4f: noise (variance) of f
        :param classify: whether the likelihood has a noise parameter
        :param max_num_ancestor: max number of reduced ancestor per observation
        """
        with torch.no_grad():
            super(Base, self).__init__()
            n_test = kwargs.get("n_test", 0)
            classify = kwargs.get("classify", False)
            use_ic0 = kwargs.get("use_ic0", True)
            nugget4f = kwargs.get("nugget4f", 0.0001)

            self.n_train = y.size(0) - n_test
            self.n_test = n_test
            self.K = K
            self.nugget4f = nugget4f
            self.likelihood = likelihood
            self.X = X
            self.y = y
            self.classify = classify
            self.use_ic0 = use_ic0

            f_init = torch.zeros(self.y.shape)
            time0 = time.perf_counter()
            X_order, _, y_order, priorSparsity, ancestorApprox, order = \
                order_X_f_y(self.X, f_init, self.y,
                            self.K.base_kernel.lengthscale,
                            rho, nTest=self.n_test, max_num_ancestor=None)
            time1 = time.perf_counter()
            time_maxmin = time1 - time0
            print(
                f"Maxmin ordering for n = {self.X.size(0)}, d = {self.X.size(1)}, "
                f"rho = {rho} used {time_maxmin} seconds")
            self.rho = rho
            self.order = order
            self.prior_sparsity, self.post_sparsity, self.prior_sparsity_train, \
            self.post_sparsity_train, self.ancestor, = \
                self.find_sparsity(priorSparsity, ancestorApprox)
            self.x_train = X_order[:self.n_train]
            self.y_train = y_order[:self.n_train]
            self.x_test = X_order[self.n_train:]
            self.y_test = y_order[self.n_train:]
            self.kernel4f = Kernel(self.x_train, self.K, self.nugget4f)

            self.rowIdxU, self.ccolIdxU = find_csc(self.n_train,
                                                   self.prior_sparsity_train)
            COOIdxUCSROrder, valUCSROrder, _ = find_inv_chol(
                self.n_train, self.kernel4f, self.rowIdxU, self.ccolIdxU)
            U = torch.sparse_coo_tensor(
                COOIdxUCSROrder, valUCSROrder,
                [self.n_train, self.n_train]).coalesce()
            if self.classify:
                likelihood = LogitLikelihood()
                dydf = Logit_dydf()
            else:
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                likelihood.noise = self.likelihood.noise.detach().clone()
                dydf = Normal_dydf(
                    self.likelihood.noise.sqrt())
            f_train_init = f_init[:self.n_train]
            if self.use_ic0:
                time0 = time.perf_counter()
                mu_post, V = find_f_map_Laplace(
                    self.y_train, f_train_init, U, self.post_sparsity_train,
                    dydf,
                    likelihood, niter=20, conv=0.01 * (self.n_train ** 0.5))
                self.mu_post = torch.nn.Parameter(mu_post)
                time1 = time.perf_counter()
                time_ic0 = time1 - time0
                print(f"Laplacian initialization used {time_ic0} seconds")
            else:
                mu_post = f_train_init.detach().clone()
                self.mu_post = torch.nn.Parameter(mu_post)
                self.rowIdxV, self.ccolIdxV = find_csc(self.n_train,
                                                       self.post_sparsity_train)
                COOIdxVCSROrder, valVCSROrder, _ = find_inv_chol(
                    self.n_train, self.kernel4f, self.rowIdxV, self.ccolIdxV)
                V = torch.sparse_coo_tensor(
                    COOIdxVCSROrder, valVCSROrder,
                    [self.n_train, self.n_train]).coalesce()
            mu0 = kwargs.get("mu0", None)
            if mu0 is not None:
                self.mu_post = mu0[self.order[:self.n_train]]
            self.V_indices = V.indices()
            self.V_indices_1dhash = np.ravel_multi_index(
                self.V_indices.tolist(), (self.n_train, self.n_train)).tolist()
            self._V_values = torch.nn.Parameter(V.values().detach().clone())
            self.V_dict = {self.V_indices_1dhash[i]: i
                           for i in range(len(self.V_indices_1dhash))}
            self.V_values_trans_mask = torch.zeros(self._V_values.size(0),
                                                   dtype=torch.bool)
            V_CSR = V.to_sparse_csr()
            self.V_values_trans_mask[V_CSR.crow_indices()[:-1]] = True
            self._V_values[self.V_values_trans_mask] = (
                self._V_values[self.V_values_trans_mask] - 0.0001).log()

    def find_sparsity(self, sparsity, ancestor):
        raise NotImplementedError

    @property
    def V_values(self):
        V_values_tmp = self._V_values.clone()
        V_values_tmp[self.V_values_trans_mask] = \
            self._V_values[self.V_values_trans_mask].exp() + 0.0001
        return V_values_tmp

    def train(self, train_prior=True, train_lk=True):
        self.mu_post.requires_grad = True
        self._V_values.requires_grad = True
        if train_prior:
            for parm in self.K.parameters():
                parm.requires_grad = True
        else:
            for parm in self.K.parameters():
                parm.requires_grad = False
        if train_lk:
            for parm in self.likelihood.parameters():
                parm.requires_grad = True
        else:
            for parm in self.likelihood.parameters():
                parm.requires_grad = False

    def predict(self):
        with torch.no_grad():
            x_all = torch.cat((self.x_train, self.x_test), dim=0)
            n = self.n_train + self.n_test
            kernel4f = Kernel(x_all, self.K, self.nugget4f)
            rowIdxU, ccolIdxU = find_csc(n, self.post_sparsity)
            COOIdxUCSROrder, valUCSROrder, _ = find_inv_chol(
                n, kernel4f, rowIdxU, ccolIdxU)
            U_full = torch.sparse_coo_tensor(
                COOIdxUCSROrder, valUCSROrder,
                [n, n]).coalesce()
            V_full = U_full.clone()
            V_full._values()[V_full.indices()[1] < self.n_train] = \
                self.V_values
            mean_unknown, var_unknown = predict_f(
                self.x_train, self.x_test, self.K, torch.tensor(0.0),
                torch.zeros(self.y_train.shape), self.mu_post,
                V_full, self.ancestor, joint=True
            )
            order_test = torch.tensor(self.order[self.n_train:]) - self.n_train
            mean_unknown_ori_order = mean_unknown.clone()
            mean_unknown_ori_order[order_test] = mean_unknown
            var_unknown_ori_order = var_unknown.clone()
            var_unknown_ori_order[order_test] = var_unknown
            return mean_unknown_ori_order, var_unknown_ori_order

    def log_score(self, f_true):
        with torch.no_grad():
            V_COO = torch.sparse_coo_tensor(self.V_indices, self.V_values)
            V_dense = V_COO.to_dense()
            f_true_order = f_true[self.order[:self.n_train]]
            logscr = - self.n_train * 0.5 * torch.tensor(torch.pi * 2).log() + \
                     V_dense[torch.arange(self.n_train),
                             torch.arange(self.n_train)].log().sum() - \
                     0.5 * torch.matmul(V_dense.t(),
                                        f_true_order - self.mu_post). \
                         square().sum()
            return - logscr

    def forward(self, inds):
        inds = inds.sort().values

        time0 = time.perf_counter()
        _, _, U_values_sub_CSC, _ = \
            update_mean_inv_chol(
                self.kernel4f, self.rowIdxU, self.ccolIdxU, inds
            )
        U_sub_row_ind = np.concatenate(
            [self.rowIdxU[self.ccolIdxU[j]:self.ccolIdxU[j + 1]].
             tolist() for j in inds])
        U_sub_col_ind = np.concatenate(
            [[j] * (self.ccolIdxU[j + 1] - self.ccolIdxU[j]).item()
             for j in inds])
        U_sub_indices_1dhash = np.ravel_multi_index(
            [U_sub_row_ind, U_sub_col_ind], (self.n_train, self.n_train))
        U_sub_dict = {ind_1dhash: value for ind_1dhash, value in
                      zip(U_sub_indices_1dhash, U_values_sub_CSC)}
        time1 = time.perf_counter()
        time_update_U = time1 - time0

        time0 = time.perf_counter()
        ELBO, time_sp_trisolve, time_sub_matrix, time_innerMean, time_logDet, \
        time_expected_likelihood = compute_ELBO(
            self.y_train, torch.zeros(self.y_train.shape), self.mu_post,
            U_sub_dict, self.V_dict, self.V_values,
            self.ancestor, likelihood=self.likelihood,
            mini_indices=inds, timing=True)
        time1 = time.perf_counter()
        time_ELBO_fwd = time1 - time0
        return ELBO, time_update_U, time_ELBO_fwd, time_sp_trisolve, \
               time_sub_matrix, time_innerMean, time_logDet, \
               time_expected_likelihood


def my_train(model: Base, **kwargs):
    kwargs_cp = copy.deepcopy(kwargs)
    optimizer = kwargs_cp.pop("optimizer", 'Adam')
    scheduler = kwargs_cp.pop("scheduler", 'ExponentialLR')
    train_prior = kwargs_cp.pop("train_prior", True)
    train_lk = kwargs_cp.pop("train_lk", True)
    optimizer_args = kwargs_cp.pop("optimizer_args", {"lr": 0.1})
    scheduler_args = kwargs_cp.pop("scheduler_args", {"gamma": 0.9})
    batsz = kwargs_cp.pop("batsz", 128)
    n_Epoch = kwargs_cp.pop("n_Epoch", 10)
    verbose = kwargs_cp.pop("verbose", True)
    timing = kwargs_cp.pop("timing", True)
    n_train = model.n_train
    time_update_U = 0.0
    time_ELBO_fwd = 0.0
    time_ELBO_bwd = 0.0
    time_sp_trisolve = 0.0
    time_sub_matrix = 0.0
    time_innerMean = 0.0
    time_logDet = 0.0
    time_expected_likelihood = 0.0
    model.train(train_prior, train_lk)
    optimizer = getattr(torch.optim, optimizer)(
        model.parameters(), **optimizer_args)
    scheduler = getattr(torch.optim.lr_scheduler, scheduler)(
        optimizer, **scheduler_args)
    ELBOLst = np.zeros(n_Epoch)
    K_parmLst = [{name: cons.transform(parm).squeeze() for
                  (name, parm), cons in
                  zip(model.K.named_parameters(), model.K.constraints())}]
    # optimization with mini-batch
    loader = torch.utils.data.DataLoader(torch.arange(n_train),
                                         batch_size=batsz, shuffle=True)
    for i in range(n_Epoch):
        ELBOLstEpoch = []
        for inds in loader:
            optimizer.zero_grad()
            ELBO, time_update_U_iter, time_ELBO_fwd_iter, \
            time_sp_trisolve_iter, time_sub_matrix_iter, \
            time_innerMean_iter, time_logDet_iter, \
            time_expected_likelihood_iter = \
                model.forward(inds)
            ELBOLstEpoch.append(ELBO.detach().clone())
            negELBO = - ELBO
            time_update_U += time_update_U_iter
            time_ELBO_fwd += time_ELBO_fwd_iter
            time_sp_trisolve += time_sp_trisolve_iter
            time_sub_matrix += time_sub_matrix_iter
            time_innerMean += time_innerMean_iter
            time_logDet += time_logDet_iter
            time_expected_likelihood += time_expected_likelihood_iter
            time0 = time.perf_counter()
            negELBO.backward()
            time1 = time.perf_counter()
            time_ELBO_bwd += time1 - time0
            optimizer.step()
        ELBOLst[i] = torch.stack(ELBOLstEpoch).sum()
        K_parmLst.extend([{name: cons.transform(parm).squeeze() for
                           (name, parm), cons in
                           zip(model.K.named_parameters(),
                               model.K.constraints())}])
        scheduler.step()
        if timing:
            print(f"time_update_U = {time_update_U}, "
                  f"time_ELBO_fwd = {time_ELBO_fwd}, "
                  f"time_ELBO_bwd = {time_ELBO_bwd}, "
                  f"time_sp_trisolve = {time_sp_trisolve}, "
                  f"time_sub_matrix = {time_sub_matrix}",
                  f"time_innerMean = {time_innerMean}",
                  f"time_logDet = {time_logDet}",
                  f"time_expected_likelihood = {time_expected_likelihood}",
                  flush=True)
        if verbose:
            print(f"Epoch {i} ELBO is {ELBOLst[i]}", flush=True)
    return ELBOLst, K_parmLst


class VIVA(Base):
    def __init__(self, X, y, K, likelihood, rho, **kwargs):
        super(VIVA, self).__init__(X, y, K, likelihood, rho, **kwargs)

    def find_sparsity(self, sparsity, ancestor):
        prior_sparsity = sparsity
        post_sparsity = sparsity
        prior_sparsity_train = prior_sparsity[:,
                               prior_sparsity[1] < self.n_train]
        post_sparsity_train = post_sparsity[:, post_sparsity[1] < self.n_train]
        return prior_sparsity, post_sparsity, prior_sparsity_train, \
               post_sparsity_train, ancestor


class FIC(Base):
    def __init__(self, X, y, K, likelihood, rho, **kwargs):
        super(FIC, self).__init__(X, y, K, likelihood, rho, **kwargs)

    def find_sparsity(self, sparsity, ancestor):
        n = self.n_train + self.n_test
        m = int(sparsity.size(1) / n)
        SP = torch.stack([torch.arange(m).repeat(n),
                          torch.arange(n).repeat(m, 1).t().reshape((-1,))],
                         0)
        SP = SP[:, SP[0] < SP[1]]
        SP = torch.cat([SP,
                        torch.stack([torch.arange(n), torch.arange(n)], 0)
                        ], dim=1)
        prior_sparsity = SP
        post_sparsity = SP
        prior_sparsity_train = prior_sparsity[:,
                               prior_sparsity[1] < self.n_train]
        post_sparsity_train = post_sparsity[:, post_sparsity[1] < self.n_train]
        ancestor = Ancestor(SP, n, lower=False)
        return prior_sparsity, post_sparsity, prior_sparsity_train, \
               post_sparsity_train, ancestor


class Diag(Base):
    def __init__(self, X, y, K, likelihood, rho, **kwargs):
        super(Diag, self).__init__(X, y, K, likelihood, rho, **kwargs)

    def find_sparsity(self, sparsity, ancestor):
        n = self.n_train + self.n_test
        prior_sparsity = sparsity
        mask_tmp = \
            (prior_sparsity[0].eq(prior_sparsity[1])).logical_or(
                prior_sparsity[1].ge(self.n_train))
        post_sparsity = prior_sparsity[:, mask_tmp]
        prior_sparsity_train = prior_sparsity[:,
                               prior_sparsity[1] < self.n_train]
        post_sparsity_train = post_sparsity[:, post_sparsity[1] < self.n_train]
        ancestor = Ancestor(post_sparsity, n, lower=False)
        return prior_sparsity, post_sparsity, prior_sparsity_train, \
               post_sparsity_train, ancestor
