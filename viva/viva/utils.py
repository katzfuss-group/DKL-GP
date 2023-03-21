import torch
import numpy as np
from gpytorch.likelihoods import _OneDimensionalLikelihood
from gpytorch.distributions import base_distributions
import maxmin_cpp
from .kernel import Kernel
from .ichol0_wrap import ichol0
from .sp_fwd_solve import spsolve_triangular


def MC_likelihood(y, mean, marginal_var, likelihood, num_samples=5000):
    """
    Returns a Monte Carlo estimate of the likelihood in the VIVA ELBO.
    Inputs :
    y - length n torch tensor of observed data.
    mean - mean of f.
    marginal_var - marginal var of f.
    likelihood - univariate log density log p(y|f) that can be used with
        tensor input
    num_samples - (optional) the number of Monte Carlo samples.
    Outputs :
    expected_log_likelihood (expct. wrt q) - torch tensor esimating
        likelihood.
    """
    n = len(mean)
    marginal_sd = marginal_var.sqrt()
    f = torch.normal(mean=torch.zeros(num_samples, n),
                     std=torch.ones(num_samples, n)) * \
        marginal_sd.expand(num_samples, -1) + mean.expand(num_samples, -1)
    likelihood_fwd = likelihood.forward(f)
    prob = likelihood_fwd.log_prob(y.expand(num_samples, -1)).sum() / \
           num_samples
    return prob


def KL_GP(mu1, mu2, cov1, cov2):
    return 0.5 * (torch.logdet(cov2) - torch.logdet(cov1) - mu1.size(0) +
                  torch.trace(torch.mm(cov2.inverse(), cov1)) +
                  torch.matmul(cov2.inverse(), mu2 - mu1).dot(mu2 - mu1))


def update_mean_inv_chol(kernel4f: Kernel, UrowIdx, UccolIdx,
                         col_range=None):
    """
    Update the prior mean and inverse Cholesky factor based on the input kernel
    K and the col_range
    Inputs :
    kernel4f - prior kernel for f
    UrowIdx, UccolIdx - CSC representation for the prior inverse upper Chol
    col_range - decides the columns of U and the coeffs in mean to be updated
    Outputs :
    meanIdxUpdate, meanValsUpdate - the indicated update is
        mean[meanIdxUpdate] = meanValsUpdate
    UvalsSubCSC, UccolIdxSub - the indicated update is
        UvalCSC[UccolIdx[col_range[k]]:UccolIdx[col_range[k] + 1]] = \
    #         UvalsSubCSC[UccolIdxSub[k]:UccolIdxSub[k + 1]]
    """
    if col_range is None:
        return
    n = len(UccolIdx) - 1
    UIdxCOOSubCSC, UvalsSubCSC, UccolIdxSub = \
        find_inv_chol(n, kernel4f, UrowIdx, UccolIdx, col_range)
    # A more general function for mean can be used
    meanIdxUpdate = torch.unique(UIdxCOOSubCSC[0])
    meanValsUpdate = torch.zeros(len(meanIdxUpdate))
    return meanIdxUpdate, meanValsUpdate, UvalsSubCSC, UccolIdxSub


def find_inv_chol(n, kernel, rowIdx, ccolIdx, col_range=None):
    """
        Build a subset of all columns of the upper sparse inverse Cholesky
        Inputs :
        n - size of the sparse matrix
        kernel - covariance kernel accepting indices
        rowIdx, ccolIdx - sparsity pattern for an upper triangular matrix, csc format
        col_range - the indices of columns to build
        Outputs :
        the upper sparse inverse Cholesky, whose columns are decided by col_range
        COOIndsCSCOrder - COO indices in CSC order
        valsCSCOrder - values in CSC order
        ccolIdxSub - cumulative column indices of the submatrix indicated by
            col_range
    """
    if col_range is None:
        col_range = torch.arange(n)
    else:
        col_range = col_range.sort().values
    col_range = col_range.tolist()
    rowIdxSub = [rowIdx[ccolIdx[j]:ccolIdx[j + 1]].tolist() for j in col_range]
    colIdxSub = [[j] * (ccolIdx[j + 1] - ccolIdx[j]).item() for j in col_range]
    ccolIdxSub = [0].extend([len(x) for x in rowIdxSub])
    rowIdx_len = [len(x) for x in rowIdxSub]
    rowIdx_len_max = max(rowIdx_len)
    covmat_padded = kernel(rowIdxSub)
    covmat = torch.eye(rowIdx_len_max).unsqueeze(0). \
        repeat(len(col_range), 1, 1)
    covmat_sub_fullind = np.indices(
        [len(col_range), rowIdx_len_max, rowIdx_len_max])
    covmat_sub_mask = np.logical_and(
        covmat_sub_fullind[1] > \
        (rowIdx_len_max - np.array(rowIdx_len) - 1)[:, None, None],
        covmat_sub_fullind[2] > \
        (rowIdx_len_max - np.array(rowIdx_len) - 1)[:, None, None],
    )
    covmat[torch.from_numpy(covmat_sub_mask)] = covmat_padded[
        torch.from_numpy(covmat_sub_mask)]
    covmat_inv = covmat.inverse()
    valsCSCOrder_padded = covmat_inv[:, -1, :] / \
                          torch.sqrt(covmat_inv[:, -1, -1:])
    valsCSCOrder_mask = np.arange(rowIdx_len_max - 1, -1, -1) < \
                        np.array(rowIdx_len)[:, None]
    valsCSCOrder = valsCSCOrder_padded[torch.from_numpy(valsCSCOrder_mask)]
    rowIdxSub = np.concatenate(rowIdxSub).tolist()
    colIdxSub = np.concatenate(colIdxSub).tolist()
    COOIndsCSCOrder = torch.tensor([rowIdxSub, colIdxSub])

    return COOIndsCSCOrder, valsCSCOrder, ccolIdxSub


def find_inv_chol_post(U: torch.Tensor, diag_var: torch.Tensor,
                       post_sparsity: torch.Tensor):
    """
    Returns the posterior inv chol of `f`, i.e., the V matrix
    Inputs :
    U - the prior inv chol
    diag_var - the marginal variances for y | f
    post_sparsity - sparsity for the upper chol of the posterior
        inverse covariance matrix
    Outputs :
    the V matrix
    """
    n = U.shape[0]
    SigmaPostInvPart1Vals = torch.stack([
        torch.sum(torch.mul(U[x[0]], U[x[1]]).coalesce().values())
        for x in post_sparsity.t()])
    SigmaPostInvPart1 = torch.sparse_coo_tensor(post_sparsity,
                                                SigmaPostInvPart1Vals,
                                                [n, n])
    SigmaPostInvPart2 = torch.sparse_coo_tensor(
        torch.tensor([range(n), range(n)]),
        1.0 / diag_var, [n, n])
    SigmaPostInv = SigmaPostInvPart1 + SigmaPostInvPart2
    V = ichol0(SigmaPostInv, upper=True)
    return V


def find_mean_post(y: torch.Tensor, V: torch.Tensor, diag_var: torch.Tensor):
    """
    Returns the posterior mean of `f`, i.e., the nu vector
    Inputs :
    y - the (pseudo-)observations
    V - the posterior inv chol
    diag_var - the marginal variances for y | f
    Outputs :
    the nu vector
    """
    n = V.shape[0]
    valuesV = V.values()
    crowIdxV, colIdxV = find_csr(n, V.indices())
    rowIdxV, ccolIdxV = find_csc(n, V.indices())
    odrCSR2CSC = csr_to_csc_order(n, V.indices())
    valuesVCSC = valuesV[odrCSR2CSC]
    muPost = spsolve_triangular(crowIdxV, colIdxV,
                                valuesV, y / diag_var, lower=False)
    muPost = spsolve_triangular(ccolIdxV, rowIdxV,
                                valuesVCSC, muPost, lower=True)
    return muPost


def find_csc(n, indices: torch.Tensor):
    """
    Find the compressed sparse column representation given a COO index set
    Inputs:
    n - dimension of the sparse matrix
    indices - the COO index set
    Outputs:
    The CSC index representation
    """
    tmp = torch.sparse_coo_tensor(indices,
                                  torch.zeros(indices.shape[1]),
                                  [n, n]).t().to_sparse_csr()
    ccolIdx = tmp.crow_indices()
    rowIdx = tmp.col_indices()
    return rowIdx, ccolIdx


def find_csr(n, indices: torch.Tensor):
    """
    Find the compressed sparse row representation given a COO index set
    Inputs:
    n - dimension of the sparse matrix
    indices - the COO index set
    Outputs:
    The CSR index representation
    """
    tmp = torch.sparse_coo_tensor(indices,
                                  torch.zeros(indices.shape[1]),
                                  [n, n]).to_sparse_csr()
    crowIdx = tmp.crow_indices()
    colIdx = tmp.col_indices()
    return crowIdx, colIdx


def csr_to_csc_order(n, indices: torch.Tensor):
    """
    Find the order that converts values stored based on CSR to those based on CSC
    Inputs:
    n - dimension of the sparse matrix
    indices - the COO index set
    Outputs:
    An order vector odrCSR2CSC s.t. values_CSC = values_CSR[odrCSR2CSC]
    """
    hash = indices[0] + indices[1] * n
    return hash.sort().indices


def find_csr_sub_mat(crowIdx, colIdx, values, rowIdxSub, colIdxSub):
    """
    Find the submatrix of the input CSR matrix
    Inputs:
    crowIdx, colIdx, values - input CSR matrix
    rowIdxSub, colIdxSub - submatrix indices
    Outputs:
    The CSR representation of the submatrix
    """
    # output variables
    crowIdxOut = torch.tensor([0])
    colIdxOut = torch.tensor([], dtype=colIdx.dtype)
    valuesOut = torch.tensor([], dtype=values.dtype)
    # in case they are not in ascending order
    if (not all(
        rowIdxSub[i] <= rowIdxSub[i + 1] for i in range(len(rowIdxSub) - 1))) or \
        (not all(colIdxSub[i] <= colIdxSub[i + 1] for i in
                 range(len(colIdxSub) - 1))):
        raise Exception(
            "Only ascending index order supported in taking the submatrix of a CSR matrix ")
    # Mapping function converting indices relative to the original matrix to
    #   those relative to the submatrix
    colIdxMapOld2New = {colIdxSub[j].item(): j for j in range(len(colIdxSub))}
    # consider each selected row
    for i in rowIdxSub:
        colIdxRowI = colIdx[crowIdx[i]:crowIdx[i + 1]]
        maskColIdxRowI = torch.isin(colIdxRowI, colIdxSub)
        colIdxRowISub = colIdxRowI[maskColIdxRowI]
        colIdxRowINew = torch.tensor(
            [colIdxMapOld2New[j.item()] for j in colIdxRowISub])
        valuesRowISub = values[crowIdx[i]:crowIdx[i + 1]][maskColIdxRowI]
        colIdxOut = torch.cat([colIdxOut, colIdxRowINew])
        valuesOut = torch.cat([valuesOut, valuesRowISub])
        crowIdxOut = torch.cat(
            [crowIdxOut, crowIdxOut[-1].add(len(valuesRowISub)).unsqueeze(0)])
    return crowIdxOut, colIdxOut, valuesOut


class LogitLikelihood(_OneDimensionalLikelihood):
    def __init__(self):
        super().__init__()

    def forward(self, function_samples, **kwargs):
        function_samples_clamp = torch.clamp(function_samples,
                                             min=-10.0, max=10.0)
        output_probs = 1.0 / (1.0 + (-function_samples_clamp).exp())
        return base_distributions.Bernoulli(probs=output_probs)


class Logit_dydf:
    def __init__(self):
        pass

    def __call__(self, y, f):
        grad_1st = y - 1.0 + (-f).exp() / (1.0 + (-f).exp())
        diag_var = (1.0 + (-f).exp()) * (1.0 + f.exp())
        return grad_1st, diag_var


class Normal_dydf:
    def __init__(self, scale=torch.tensor(1.0)):
        self.scale = scale

    def __call__(self, y, f):
        grad_1st = (y - f) / (self.scale ** 2)
        diag_var = torch.ones(len(f)) * (self.scale ** 2)
        return grad_1st, diag_var


class Student_dydf:
    def __init__(self, df=torch.tensor(2.0), scale=torch.tensor(1.0)):
        self.df = df
        self.scale = scale

    def __call__(self, y, f):
        z = (y - f) / self.scale
        grad_1st = (self.df + 1) * z / (self.df + z ** 2) / self.scale
        grad_2nd = (self.df + 1) * (- self.df + z ** 2) / \
                   ((self.df + z ** 2) ** 2) / (self.scale ** 2)
        return grad_1st, grad_2nd
