import torch
import numpy as np
import pandas as pd
import os
import maxmin_cpp
import viva
import time
import matplotlib.pyplot as plt

root = '../../datasets'
d = 5
lengthscale = torch.linspace(0.2, 1.0, steps=d)
n_list = [250, 500, 1000, 2000, 5000, 10000, 20000, 40000]
rho_list = torch.linspace(1.0, 3.0, 11)
num_sparsity = torch.zeros([len(n_list), len(rho_list)], dtype=torch.int)
num_red_ancestor = torch.zeros([len(n_list), len(rho_list)], dtype=torch.int)
num_ancestor = torch.zeros([len(n_list), len(rho_list)], dtype=torch.int)
max_num_red_ancestor = \
    torch.zeros([len(n_list), len(rho_list)], dtype=torch.int)
idx_rho = int((1 + len(rho_list)) / 2) - 1
idx_n = 5
# for i in range(len(n_list)):
#     n = n_list[i]
#     data_name = f"normal_logit_n{n}_d5_seed0"
#     X = torch.from_numpy(
#         pd.read_csv(os.path.join(root, data_name, "train", 'x.csv')).to_numpy())
#     n_train = n - int(n / 5)
#     X_scale = X / lengthscale.squeeze().unsqueeze(0).expand(n_train, -1)
#     for j in range(len(rho_list)):
#         time0 = time.perf_counter()
#         rho = rho_list[j]
#         if j != idx_rho and i != idx_n:
#             continue
#         initInd = 0
#         orderObj = maxmin_cpp.MaxMincpp(X_scale, rho, initInd, 0)
#         ancestorApprox = torch.tensor([orderObj[3], orderObj[2]])
#         sparsity = ancestorApprox[:, orderObj[4]]
#         num_sparsity[i, j] = sparsity.size(1)
#         num_red_ancestor[i, j] = ancestorApprox.size(1)
#         sparsity = viva.Ancestor(sparsity, n_train, lower=False)
#         ancestorFull = sparsity.find_DAG()
#         num_ancestor[i, j] = ancestorFull.idx.size(0)
#         ancestorApprox = viva.Ancestor(ancestorApprox, n_train, lower=False)
#         max_num_red_ancestor[i, j] = max(
#             [len(ancestorApprox.get_ancestor_idx(i)) for i in range(n_train)])
#         time1 = time.perf_counter()
#         print(f"n = {n}, rho = {rho} used {time1 - time0} secs", flush=True)
#
# torch.save({"idx_n": idx_n, "idx_rho": idx_rho, "n_list": n_list,
#             "rho_list": rho_list, "num_sparsity": num_sparsity,
#             "num_red_ancestor": num_red_ancestor,
#             "num_ancestor": num_ancestor,
#             "max_num_red_ancestor": max_num_red_ancestor},
#            "n_nonzero_vs_rho_n.pt")

rslt = torch.load("n_nonzero_vs_rho_n.pt")
idx_n = rslt['idx_n']
idx_rho = rslt['idx_rho']
n_list = rslt['n_list']
rho_list = rslt['rho_list']
num_sparsity = rslt['num_sparsity']
num_red_ancestor = rslt['num_red_ancestor']
num_ancestor = rslt['num_ancestor']
max_num_red_ancestor = rslt['max_num_red_ancestor']
fig_size = (6, 5)
font_size = 16
# n = 2000
plt.figure(figsize=fig_size)
y_tmp = torch.cat([num_sparsity[idx_n:(idx_n + 1), :],
                   num_red_ancestor[idx_n:(idx_n + 1), :],
                   num_ancestor[idx_n:(idx_n + 1), :]],
                  dim=0).t() / (n_list[idx_n] * 0.8)
y_ticks = (np.array([1, 10, 100, 600]))
plt.plot(rho_list, y_tmp)
plt.xlabel("rho", fontsize=font_size)
plt.ylabel(None)
plt.tick_params(axis='both', which='major', labelsize=font_size)
plt.legend(['sparsity', 'reduced ancestor', 'full ancestor'],
           fontsize=font_size)
plt.tight_layout()
plt.savefig(f"m_vs_rho_n{n_list[idx_n]}.pdf")

# rho = 2.0
plt.figure(figsize=fig_size)
plt.plot(np.array(n_list) * 0.8,
         num_sparsity[:, idx_rho] / (torch.tensor(n_list) * 0.8), c="green")
plt.plot(np.array(n_list) * 0.8,
         num_red_ancestor[:, idx_rho] / (torch.tensor(n_list) * 0.8), c="blue")
plt.plot(np.array(n_list) * 0.8,
         num_ancestor[:, idx_rho] / (torch.tensor(n_list) * 0.8), c="red")
plt.xticks(ticks=[0, 8000, 16000, 24000, 32000])
plt.xlabel("n", fontsize=font_size)
plt.ylabel(None)
# plt.yscale("log")
plt.tick_params(axis='both', which='major', labelsize=font_size)
plt.legend(['sparsity', 'reduced ancestor', 'full ancestor'],
           fontsize=font_size)
plt.tight_layout()
plt.savefig(f"m_vs_n_rho{rho_list[idx_rho]}.pdf")

# rho = 2.0, max_num_red_ancestor
plt.figure(figsize=fig_size)
plt.plot(np.array(n_list) * 0.8, max_num_red_ancestor[:, idx_rho])
plt.xlabel("n", fontsize=font_size)
plt.ylabel("max number of ancestors", fontsize=font_size)
plt.tick_params(axis='both', which='major', labelsize=font_size)
plt.tight_layout()
plt.savefig(f"max_num_ances_vs_n_rho{rho_list[idx_rho]}.pdf")
