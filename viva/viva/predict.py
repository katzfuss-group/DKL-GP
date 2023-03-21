import torch
from .sp_fwd_solve import spsolve_triangular
from .utils import find_csr_sub_mat


def predict_f(X_known, X_unknown, K, nugget4f, mean, mean_post,
              V_full_COO, ancestor, joint=False):
    """
    Predict f_unknown | y_known in terms of posterior marginal mean and
        variance. f_unknown is assumed to have zero prior mean
    :param X_known: design matrix of known responses
    :param X_unknown: design matrix of unknown responses
    :param K, nugget4f: prior kernel
    :param mean: prior mean of known responses
    :param mean_post: posterior mean of known responses
    :param V_full_COO: posterior inverse Cholesky factor, incl. testing locs
    :param ancestor: the ancestor set of all (known and unknown) responses
    :param joint: whether joint prediction is performed
    :return: mean_unknown, var_unknown
    """
    nKnown = X_known.size(0)
    nUnknown = X_unknown.size(0)
    var_unknown = torch.zeros(nUnknown)
    V_full_CSR = V_full_COO.to_sparse_csr()
    V_full_crow_indices = V_full_CSR.crow_indices()
    V_full_col_indices = V_full_CSR.col_indices()
    V_full_values_CSR = V_full_CSR.values()
    # posterior mean
    if joint:
        meanDiff = mean_post - mean
        V_pred_known = torch.index_select(torch.index_select(
            V_full_COO, 0, torch.arange(nKnown)
        ), 1, torch.arange(nKnown, nKnown + nUnknown))
        V_t_sub_CSR = \
            torch.index_select(torch.index_select(
                V_full_COO, 0, torch.arange(nKnown, nKnown + nUnknown)
            ), 1, torch.arange(nKnown, nKnown + nUnknown)).t().to_sparse_csr()
        V_t_sub_crow_indices = V_t_sub_CSR.crow_indices()
        V_t_sub_col_indices = V_t_sub_CSR.col_indices()
        V_t_sub_values_CSR = V_t_sub_CSR.values()
        mean_unknown_stp1 = - torch.matmul(V_pred_known.t(), meanDiff)
        mean_unknown = spsolve_triangular(
            V_t_sub_crow_indices, V_t_sub_col_indices, V_t_sub_values_CSR,
            mean_unknown_stp1, lower=True)
    else:
        mean_unknown = torch.zeros(nUnknown)
    for k in range(nUnknown):
        j = nKnown + k
        ind = ancestor.get_ancestor_idx(j)
        if not joint:
            ind = ind[torch.logical_or(ind.lt(nKnown), ind.eq(j))]
        if len(ind) == 1:
            mean_unknown[k] = 0.0
            var_unknown[k] = K(X_unknown[k:k + 1], X_unknown[k:k + 1]). \
                                 evaluate().squeeze() + nugget4f
            continue
        # posterior mean
        if not joint:
            meanDiff = mean_post[ind[:-1]] - mean[ind[:-1]]
            V_kcol = torch.index_select(torch.index_select(
                V_full_COO, 1, torch.tensor([j])
            ), 0, ind[:-1]).to_dense().squeeze()
            mean_unknown[k] = - torch.dot(meanDiff, V_kcol) / \
                              V_full_COO[j, j]
        # posterior var
        ej = torch.zeros(len(ind))
        ej[len(ind) - 1] = 1
        V_sub_crow_indices, V_sub_col_indices, V_sub_values_CSR = \
            find_csr_sub_mat(
                V_full_crow_indices, V_full_col_indices, V_full_values_CSR,
                ind, ind)
        var_unknown[k] = spsolve_triangular(
            V_sub_crow_indices, V_sub_col_indices, V_sub_values_CSR, ej,
            lower=False).square().sum()

    return mean_unknown, var_unknown
