import torch
from .utils import find_inv_chol_post, find_mean_post


def find_f_map_Laplace(y, f_init, U, post_sparsity, dydf, likelihood,
                       conv=1e-4, niter=20, test=False):
    """
    Find the maximum a posteriori est of f using laplacian approx
    Inputs :
    y - the length n response vector
    f_init - the initial value of f for optimization
    U - the prior inv chol
    post_sparsity - sparsity for the upper chol of the posterior
        inverse covariance matrix
    dydf - the function that produces the 1st and 2nd order derivatives of
        log p(y | f)
    likelihood - the log-likelihood of the model
    conv - convergence threshold for f MAP
    niter - maximum number of iter for optimizing f
    Outputs :
    the MAP estimation of f
    """
    f = f_init.detach().clone()
    nan_flag = False
    for i in range(niter):
        grad_1st, diag_var = dydf(y, f)
        pseudo_y = f + diag_var * grad_1st
        if test:
            U_dense = U.to_dense()
            covM_post_inv = U_dense @ U_dense.t() + torch.diag(1.0 / diag_var)
            covM_post = covM_post_inv.inverse()
            f_new = covM_post @ (pseudo_y / diag_var)
            V = U
        else:
            V = find_inv_chol_post(U, diag_var, post_sparsity)
            f_new = find_mean_post(pseudo_y, V, diag_var)
        if i == 0:
            likelihood_fwd = likelihood.forward(f_new)
            p_f_given_y = -0.5 * torch.matmul(U.t(), f_new).square().sum() + \
                          likelihood_fwd.log_prob(y).sum()
        else:
            likelihood_fwd = likelihood.forward(f_new)
            p_f_given_y_new = -0.5 * torch.matmul(U.t(), f_new).square().sum() + \
                              likelihood_fwd.log_prob(y).sum()
            while p_f_given_y_new < p_f_given_y:
                f_new = (f_new - f) * 0.5 + f
                if (f_new - f).square().mean().sqrt() < conv:
                    return f_new, V
                likelihood_fwd = likelihood.forward(f_new)
                p_f_given_y_new = -0.5 * torch.matmul(U.t(),
                                                      f_new).square().sum() + \
                                  likelihood_fwd.log_prob(y).sum()
            p_f_given_y = p_f_given_y_new
        if torch.any(torch.isnan(f_new)):
            nan_flag = True
            break
        if (f - f_new).square().mean().sqrt() < conv:
            return f_new, V
        else:
            f = f_new
    if nan_flag:
        print("Nan produced when calling find_f_map_Laplace", flush=True)
        V = U.detach().clone()
        f = f_init.detach().clone()
    return f, V

