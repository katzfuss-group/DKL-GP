import torch
import numpy
import ichol0 as lib


def ichol0(spMatCOO: torch.Tensor, upper=False):
    if not spMatCOO.is_coalesced():
        spMatCOO = spMatCOO.coalesce()
    nrow = spMatCOO.shape[0]
    indices = spMatCOO.indices()
    vals = spMatCOO.values()
    if upper:
        indices = - indices + nrow - 1
    inds_inds_low = indices[0] >= indices[1]
    inds_low = indices[:, inds_inds_low]
    vals_low = vals[inds_inds_low]
    spMatCOONew = torch.sparse_coo_tensor(inds_low, vals_low,
                                          [nrow, nrow]).coalesce()
    inds_low = spMatCOONew.indices()
    # vals_low = spMatCOONew.coalesce().values()
    spMatCSR = spMatCOONew.to_sparse_csr()
    crow_inds = spMatCSR.crow_indices().tolist()
    col_inds = spMatCSR.col_indices().tolist()
    vals = spMatCSR.values().detach().clone().tolist()
    rslt = lib.ichol0(crow_inds, col_inds, vals, nrow)
    if upper:
        inds_low = - inds_low + nrow - 1
    return torch.sparse_coo_tensor(inds_low,
                                   torch.tensor(rslt),
                                   [nrow, nrow]).coalesce()
