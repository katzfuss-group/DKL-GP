import torch
import gpytorch


class Kernel:
    def __init__(self, loc: torch.Tensor,
                 kernel_gpytorch: gpytorch.kernels.Kernel, nugget: float):
        self.loc = loc
        self.kernel = kernel_gpytorch
        self.nugget = nugget

    def __call__(self, ind1: list, ind2: list = None, **kwargs):
        try:
            ind1_len = [len(x) for x in ind1]
            ind1_len_max = max(ind1_len)
            ind1_padded = [[0] * (ind1_len_max - len(x)) + x for x in ind1]
            if ind2 is not None:
                ind2_len = [len(x) for x in ind2]
                ind2_len_max = max(ind2_len)
                ind2_padded = [[0] * (ind2_len_max - len(x)) + x for x in ind2]
        except:
            ind1_padded = ind1
            ind2_padded = ind2
        if ind2 is None:
            loc = torch.stack([self.loc[i] for i in ind1_padded])
            covMat = self.kernel(loc, **kwargs).evaluate()
        else:
            loc1 = torch.stack([self.loc[i] for i in ind1_padded])
            loc2 = torch.stack([self.loc[i] for i in ind2_padded])
            covMat = self.kernel(loc1, loc2, **kwargs).evaluate()
        # nugget effect
        if ind2 is None:
            if covMat.ndim == 3:
                covMat[:, torch.arange(ind1_len_max),
                torch.arange(ind1_len_max)] += self.nugget
            else:
                covMat[torch.arange(len(ind1)),
                       torch.arange(len(ind1))] += self.nugget
        return covMat
