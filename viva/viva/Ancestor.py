import torch
from .utils import find_csr

class Ancestor:
    def __init__(self, indices, n, lower=True):
        """
        input:
        indices - sparsity parttern of either lower-triangular or upper-triangular
            in the COO format
        n - number of observations
        """
        if type(indices) is torch.Tensor:
            pass
        else:
            indices = torch.tensor(indices)
        if lower & torch.any(torch.lt(indices[0], indices[1])):
            raise Exception("Input sparsity is not lower triangular")
        if (not lower) & torch.any(torch.gt(indices[0], indices[1])):
            raise Exception("Input sparsity is not upper triangular")
        if not lower:
            indices = torch.flip(indices, [0])
        self.n = n
        self.cidx, self.idx = find_csr(self.n, indices)

    def get_ancestor_idx(self, i):
        if i >= self.n:
            raise Exception(f"Input i = {i} exceeds the number of observations")
        return self.idx[self.cidx[i]:self.cidx[i + 1]]

    def find_DAG(self):
        """
        Find the DAG based on the current Ancestor set
        Output :
        Another object of the Ancestor class
        """
        cidxNew = torch.tensor([0, 1])
        idxDepNew = torch.tensor([0])
        idxNew = torch.tensor([0])
        for i in range(1, self.n):
            idx1stAncestI = self.get_ancestor_idx(i)
            idxDepI = torch.tensor([], dtype=idxDepNew.dtype)
            for j in idx1stAncestI[:-1]:
                idxDepI = torch.cat(
                    [idxDepI, idxDepNew[cidxNew[j]:cidxNew[j + 1]]])
            idxDepI = torch.unique_consecutive(idxDepI.sort().values)
            idxDepNew = torch.cat([idxDepNew, idxDepI, torch.tensor([i])])
            cidxNew = torch.cat([cidxNew,
                                 (cidxNew[-1] + len(idxDepI) + 1).unsqueeze(0)])
            idxNew = torch.cat([idxNew, torch.tensor([i] * (len(idxDepI) + 1))])
        return Ancestor(torch.cat([idxNew.unsqueeze(0), idxDepNew.unsqueeze(0)],
                                  dim=0),
                        self.n)
