import torch
import maxmin_cpp
from .Ancestor import Ancestor


def order_X_f_y(X, f, y, lengthscale, rho, initInd=0, nTest=0,
                max_num_ancestor=None):
    n = len(y)
    X_scale = X / lengthscale.squeeze().unsqueeze(0).expand(n, -1)
    orderObj = maxmin_cpp.MaxMincpp(X_scale, rho, initInd, nTest)
    X_order = X[orderObj[0], :]
    f_order = f[orderObj[0]]
    y_order = y[orderObj[0]]
    ancestorApprox = torch.tensor([orderObj[3], orderObj[2]])
    if max_num_ancestor is not None:
        mask = torch.zeros(ancestorApprox.size(1), dtype=torch.int)
        counter = {i: 0 for i in range(n)}
        for i in range(ancestorApprox.size(1)):
            if ancestorApprox[0][i] == ancestorApprox[1][i] or \
                ancestorApprox[1][i] >= n - nTest:
                continue
            mask[i] = counter[ancestorApprox[1][i].item()]
            counter[ancestorApprox[1][i].item()] += 1
        mask = mask.le(max_num_ancestor)
        sparsity = ancestorApprox[:,
                   mask.logical_and(torch.tensor(orderObj[4]))]
        ancestorApprox = ancestorApprox[:, mask]
    else:
        sparsity = ancestorApprox[:, orderObj[4]]
        ancestorApprox = ancestorApprox[:, ancestorApprox[1].ge(0)]
        sparsity = sparsity[:, sparsity[1].ge(0)]
    ancestorApprox = Ancestor(ancestorApprox, n, lower=False)
    maxmin_order = orderObj[0]
    return X_order, f_order, y_order, sparsity, ancestorApprox, maxmin_order
