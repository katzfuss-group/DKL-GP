import torch
import gpytorch
import numpy as np
import time
import copy
import indexing  # local module
from .utils import Logit_dydf, Normal_dydf, find_csc, find_inv_chol, \
    LogitLikelihood
from .MM_order import order_X_f_y
from .kernel import Kernel
from .Laplace_init import find_f_map_Laplace
from .predict import predict_f
from .Ancestor import Ancestor
from .utils import MC_likelihood


class MySpMatrix(torch.nn.Module):
    def __init__(self, n, idx, val, ancCidx, ancIdx, **kwargs):
        """
        Self-defined class for storing a sparse matrix that supports variable
            transformation, extracting sub-matrices and rows based on the
            `Ancestor' set. The non-zero values are stored in the CSR order.
        :param n: dimension of the matrix, i.e., n X n
        :param idx: indices for non-zero entries 2 X nnz, where nnz is the
            number of non-zero entries
        :param val: values of the non-zero entries, whose order matches that of
            idx
        :param ancCidx: the cumulative-number-of-ancestor set
        :param ancIdx: the ancestor indices
        """
        super(MySpMatrix, self).__init__()
        self.nnz = val.size(0)
        self.n = n
        # map the initial order to CSR order
        init_odr_to_CSR = torch.sort(idx[0] * self.n + idx[1]).indices
        # treat values in the sparse matrix as torch.nn.parameters or not
        self.train = kwargs.get("train", False)
        if self.train:
            self._values = torch.nn.Parameter(torch.zeros(self.nnz))
        else:
            self._values = torch.zeros(self.nnz)
        # value transformation
        self.trans_mask = kwargs.get(
            "trans_mask",
            torch.zeros(self.nnz, dtype=torch.bool))[init_odr_to_CSR]
        # only "exp" supported now
        self.trans_type = kwargs.get("trans_type", None)
        self.trans_scale = kwargs.get("trans_scale", torch.tensor(1.0))
        self.trans_shift = kwargs.get("trans_shift", torch.tensor(0.0001))
        # sparsity and ancestor sets
        tmp_mat = torch.sparse_coo_tensor(idx, val).to_sparse_csr()
        self.sp_cidx_csr = tmp_mat.crow_indices()
        self.sp_idx_csr = tmp_mat.col_indices()
        self.anc_cidx = ancCidx
        self.anc_idx = ancIdx
        self.n_anc = self.anc_cidx[1:] - self.anc_cidx[:-1]
        # the sparsity and ancestor class defined in C++
        self.sp_anc = indexing.Ancestor(
            self.n, self.sp_cidx_csr.tolist(), self.sp_idx_csr.tolist(),
            self.anc_cidx.tolist(), self.anc_idx.tolist()
        )
        # initialize values with input `val`
        with torch.no_grad():
            self.update_rows(torch.arange(self.n), val[init_odr_to_CSR])

    @property
    def values(self):
        values_tmp = self._values.clone()
        if self.trans_type is None:
            values_tmp[self.trans_mask] = \
                self.trans_scale * self._values[self.trans_mask] + \
                self.trans_shift
        elif self.trans_type == "exp":
            values_tmp[self.trans_mask] = \
                self.trans_scale * self._values[self.trans_mask].exp() + \
                self.trans_shift
        else:
            raise NotImplementedError
        return values_tmp

    def update_rows(self, row_idx, val_CSR):
        """
        Setting values for rows of a sparse matrix S
        :param row_idx: the indices for the rows to be updated
        :param val_CSR: the values whose length match the number of non-zero
            entries in S[row_idx], stored in CSR order
        """
        idx_mem = torch.cat([
            torch.arange(self.sp_cidx_csr[i], self.sp_cidx_csr[i + 1])
            for i in row_idx
        ])  # indices in memory for non-zero coeff in S[row_idx]
        # mask for reverse transformation on val_CSR
        sub_trans_mask = self.trans_mask[idx_mem]
        val_CSR_rev_trans = val_CSR.clone()
        if self.trans_type is None:
            val_CSR_rev_trans[sub_trans_mask] = \
                (val_CSR[sub_trans_mask] - self.trans_shift) / self.trans_scale
        elif self.trans_type == "exp":
            val_CSR_rev_trans[sub_trans_mask] = torch.log(
                (val_CSR[sub_trans_mask] - self.trans_shift) / self.trans_scale)
        else:
            raise NotImplementedError
        self._values[idx_mem] = val_CSR_rev_trans

    def get_sub_rows(self, bat_idxs):
        """
        Get `bat_size` X `anc_idx_len_max` sub-rows. Selected rows correspond to
            `bat_idxs`. For the i-th row, if selected, the selected entries
            correspond to the ancestor set of i-th variable (location). Sub-rows
            smaller than `max_anc_size` are stored in the right segments.
        :return: a tensor of size `bat_size` X `max_anc_size`
        """
        bat_size = bat_idxs.shape[0]
        anc_idx_len = self.n_anc[bat_idxs]
        anc_idx_len_max = max(anc_idx_len)
        rslt = torch.zeros(bat_size, anc_idx_len_max)
        rslt_flat_view = rslt.flatten()
        # indices in the result and indices in the memory
        idxs = self.sp_anc.query_row(bat_idxs.tolist())
        rslt_flat_view[idxs[0]] = self.values[idxs[1]]
        return rslt

    def get_sub_mat(self, bat_idxs, unit_diag=True):
        """
        Get `bat_size` X `anc_idx_len_max` X `anc_idx_len_max` sub-matrices of
            the sparse matrix S. Selected sub-matrices correspond to
            S[Ancestor[i],Ancestor[i]]  for i in `bat_idxs`. If a sub-matrix  is
            smaller than `anc_idx_len_max` X `anc_idx_len_max`. It is stored in
            the bottom right corner.

        :param bat_idxs: indices for querying the Ancestor sets
        :param unit_diag: whether to set diag entries of each sub-matrix to one
            or zero
        :return: a tensor of size `bat_size` X `max_anc_size`
        """
        bat_size = bat_idxs.shape[0]
        ances_ind_len = self.n_anc[bat_idxs]
        ances_ind_len_max = max(ances_ind_len)
        if unit_diag:
            rslt = torch.eye(ances_ind_len_max).repeat(bat_size, 1, 1)
        else:
            rslt = torch.zeros(bat_size, ances_ind_len_max, ances_ind_len_max)
        rslt_flat_view = rslt.flatten()
        inds = self.sp_anc.query(bat_idxs.tolist())
        rslt_flat_view[inds[0]] = self.values[inds[1]]
        return rslt

    def to_dense(self):
        return torch.sparse_csr_tensor(
            self.sp_cidx_csr, self.sp_idx_csr, self.values,
            size=[self.n, self.n]
        ).to_dense()

    def detach(self):
        self._values = self._values.detach()


def compute_ELBO(y, mean, mean_post, UTrans: MySpMatrix, V: MySpMatrix,
                    likelihood=None, noise=None, mini_indices=None,
                    timing=False):
    """
    Computes and returns the VIVA ELBO.
    Inputs:
    y - observations
    mean - prior mean of f
    mean_post - posterior mean of f
    UTrans - transpose of sparse inverse Chol of the prior
    V - sparse inverse Chol of the posterior
    likelihood - the log-likelihood of the model
    noise - assume that y = f + N(0, noise) if noise is not None.
            Otherwise, Monte Carlo is used for the expectation
    mini_indices - mini-batch indices in range(n)
    """
    time_trisolve = 0.0
    time_sub_matrix = 0.0
    n = len(mean)
    if mini_indices is None:
        mini_indices = torch.arange(n)
    batch_size = len(mini_indices)
    # extract sub-matrices and sub-rows
    time0 = time.perf_counter()
    V_sub = V.get_sub_mat(mini_indices, unit_diag=True)
    U_sub = UTrans.get_sub_rows(mini_indices)
    time1 = time.perf_counter()
    time_sub_matrix += time1 - time0
    # mean difference, each row of `bat_mean_diff_sub` is of length
    #   max_anc_size. mean[Ancestor[i]] - mean_post[Ancestor[i]] is stored,
    #   for i in `mini_indices`
    U_n_anc_sub_max = max(UTrans.n_anc[mini_indices])
    bat_mean_diff_sub = torch.zeros(batch_size, U_n_anc_sub_max)
    bat_mean_diff_sub_mask = UTrans.n_anc[mini_indices].unsqueeze(-1) > \
                             torch.arange(U_n_anc_sub_max - 1, -1,
                                          -1).unsqueeze(0)
    bat_mean_diff_sub[bat_mean_diff_sub_mask] = torch.cat(
        [mean[UTrans.anc_idx[UTrans.anc_cidx[i]:UTrans.anc_cidx[i + 1]]] -
         mean_post[UTrans.anc_idx[UTrans.anc_cidx[i]:UTrans.anc_cidx[i + 1]]]
         for i in mini_indices])
    # marginal variance (V^{-1}ej) and deviation from the prior (V^{-1}U_{j})
    time0 = time.perf_counter()
    ej = torch.zeros(batch_size, max(V.n_anc[mini_indices]))
    ej[:, -1] = 1
    marginalVarPost = torch.linalg.solve_triangular(
        V_sub, ej.unsqueeze(2), upper=True).square().sum(dim=[1, 2])
    innerCov = - torch.linalg.solve_triangular(
        V_sub, U_sub.unsqueeze(2), upper=True).square().sum() / 2.0
    time1 = time.perf_counter()
    time_trisolve += time1 - time0
    # compute prior likelihood, inner prod between U and mean
    time0 = time.perf_counter()
    innerMean = - torch.bmm(
        U_sub.unsqueeze(1), bat_mean_diff_sub.unsqueeze(2)).square().sum() / 2.0
    time1 = time.perf_counter()
    time_innerMean = time1 - time0
    # compute log determinants of U and V,
    #   `UTrans` is lower-tri `V` is upper-tri
    time0 = time.perf_counter()
    logDet = UTrans.values[UTrans.sp_cidx_csr[mini_indices + 1] - 1]. \
                 log().sum() - V.values[V.sp_cidx_csr[mini_indices]].log().sum()
    time1 = time.perf_counter()
    time_logDet = time1 - time0
    # compute the 1-d expectation in ELBO
    time0 = time.perf_counter()
    if noise is None:
        expected_likelihood = MC_likelihood(
            y[mini_indices], mean_post[mini_indices], marginalVarPost,
            likelihood
        )
    else:
        expected_likelihood = - batch_size / 2.0 * \
                              torch.log(2.0 * torch.pi * noise) - \
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
            self.classify = classify
            self.use_ic0 = use_ic0

            f_init = torch.zeros(y.shape)
            time0 = time.perf_counter()
            # maxmin ordering, also find the sparsity and the reduced ancestor 
            #     sets
            X_order, _, y_order, priorSparsity, ancestorApprox, order = \
                order_X_f_y(X, f_init, y,
                            self.K.base_kernel.lengthscale,
                            rho, nTest=self.n_test, max_num_ancestor=None)
            time1 = time.perf_counter()
            time_maxmin = time1 - time0
            print(f"Maxmin ordering for n = {X.size(0)}, "
                  f"d = {X.size(1)}, rho = {rho} used "
                  f"{time_maxmin} seconds")
            self.rho = rho
            self.order = order
            self.prior_sparsity, self.post_sparsity, self.prior_sparsity_train, \
            post_sparsity_train, self.ancestor = \
                self.find_sparsity(priorSparsity, ancestorApprox)
            self.x_train = X_order[:self.n_train]
            self.y_train = y_order[:self.n_train]
            self.x_test = X_order[self.n_train:]
            self.y_test = y_order[self.n_train:]
            self.kernel4f = Kernel(self.x_train, self.K, self.nugget4f)
            # Build U transpose as the field U_trans
            # A separate U is needed for calling the Laplace initialization
            #     currently
            rowIdxU, ccolIdxU = find_csc(self.n_train,
                                         self.prior_sparsity_train)
            COOIdxUCSC, valUCSC, _ = find_inv_chol(
                self.n_train, self.kernel4f, rowIdxU, ccolIdxU)
            self.U_trans = MySpMatrix(self.n_train, COOIdxUCSC[[1, 0]], valUCSC,
                                      self.ancestor.cidx, self.ancestor.idx)
            U = torch.sparse_coo_tensor(
                COOIdxUCSC, valUCSC,
                [self.n_train, self.n_train]).coalesce()
            # Initialization of mu_post and V
            # Use Laplace initialization if `use_ic0`
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
                    self.y_train, f_train_init, U, post_sparsity_train,
                    dydf, likelihood, niter=20,
                    conv=0.01 * (self.n_train ** 0.5))
                self.mu_post = torch.nn.Parameter(mu_post)
                time1 = time.perf_counter()
                time_ic0 = time1 - time0
                print(f"Laplacian initialization used {time_ic0} seconds")
            else:
                mu_post = f_train_init.detach().clone()
                self.mu_post = torch.nn.Parameter(mu_post)
                self.rowIdxV, self.ccolIdxV = find_csc(self.n_train,
                                                       post_sparsity_train)
                COOIdxVCSC, valVCSC, _ = find_inv_chol(
                    self.n_train, self.kernel4f, self.rowIdxV, self.ccolIdxV)
                V = torch.sparse_coo_tensor(
                    COOIdxVCSC, valVCSC,
                    [self.n_train, self.n_train]).coalesce()
            mu0 = kwargs.get("mu0", None)
            if mu0 is not None:
                self.mu_post = mu0[self.order[:self.n_train]]
            self.V = MySpMatrix(
                self.n_train, V.indices(), V.values().detach().clone(),
                self.ancestor.cidx, self.ancestor.idx,
                trans_mask=(V.indices()[0] == V.indices()[1]),
                trans_type="exp", train=True
            )

    def find_sparsity(self, sparsity, ancestor):
        raise NotImplementedError

    def train(self, train_prior=True, train_lk=True):
        self.mu_post.requires_grad = True
        for parm in self.V.parameters():
            parm.requires_grad = True
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
            COOIdxUCSC, valUCSC, _ = find_inv_chol(
                n, kernel4f, rowIdxU, ccolIdxU)
            U_full = torch.sparse_coo_tensor(
                COOIdxUCSC, valUCSC,
                [n, n]).coalesce()
            V_full = U_full.clone()
            V_full._values()[V_full.indices()[1] < self.n_train] = \
                self.V.values
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
            V_dense = self.V.to_dense()
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
        _, U_values_sub_CSC, _ = find_inv_chol(
            self.n_train, self.kernel4f, self.U_trans.sp_idx_csr,
            self.U_trans.sp_cidx_csr, inds
        )
        self.U_trans.detach()
        self.U_trans.update_rows(inds, U_values_sub_CSC)
        time1 = time.perf_counter()
        time_update_U = time1 - time0

        time0 = time.perf_counter()
        ELBO, time_sp_trisolve, time_sub_matrix, time_innerMean, time_logDet, \
        time_expected_likelihood = compute_ELBO(
            self.y_train, torch.zeros_like(self.y_train), self.mu_post,
            self.U_trans, self.V, likelihood=self.likelihood, mini_indices=inds,
            timing=True)
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
