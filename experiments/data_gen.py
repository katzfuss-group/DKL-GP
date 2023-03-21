import torch
import gpytorch
from gpytorch.kernels import *
from torch.distributions.multivariate_normal import MultivariateNormal
import pandas as pd
from pathlib import Path
import viva
import yaml

# Currently only used in the testing functions in 'tests' folder
def gen_data(nTrain, d, lengthscale=None, nugget4f=0.0, nugget4y=0.1, nu=0.5,
             seed=0, nTest=0, outputscale=1.0, y_gen=None, grid=False):
    n = nTrain + nTest
    # Build locations (X)
    torch.manual_seed(seed)
    if grid:
        nKnot = int(n ** (1 / d)) + 1
        edge = torch.linspace(0, 1, steps=nKnot)
        X = torch.flatten(
            torch.stack(torch.meshgrid([edge] * d, indexing='ij')),
            start_dim=1).t()[:n]
    else:
        X = torch.rand(n, d)
    # Define a covariance kernel
    if lengthscale is None:
        lengthscale = torch.linspace(0.5, 2.5, d)
    logNugget4f = torch.log(torch.tensor(nugget4f))
    logNugget4y = torch.log(torch.tensor(nugget4y))
    K = ScaleKernel(MaternKernel(ard_num_dims=d, nu=nu))
    K.base_kernel.lengthscale = lengthscale
    K.outputscale = outputscale
    kernel4f = viva.Kernel(X, K, torch.exp(logNugget4f).item())
    # Build responses (y)
    covM4f = kernel4f(list(range(n)))
    GPObj4f = MultivariateNormal(torch.zeros(n), covM4f)
    f = GPObj4f.sample()
    if y_gen is None:
        y_gen = lambda x: x + \
                          torch.normal(torch.zeros(len(x)),
                                       nugget4y)
    y = y_gen(f)
    # y = y.sub(y.mean())
    return X, f, y, K, logNugget4f, logNugget4y


class Normal_y_gen:
    def __init__(self, scale=torch.tensor(1.0)):
        self.scale = scale

    def __call__(self, f):
        noise_gen = torch.distributions.normal.Normal(loc=torch.tensor(0.0),
                                                      scale=self.scale)
        return f + noise_gen.sample(f.shape)


class Logit_y_gen:
    def __init__(self):
        pass

    def __call__(self, f):
        prob = 1.0 / (1.0 + torch.exp(-f))
        return torch.bernoulli(prob)


class Student_y_gen:
    def __init__(self, df=torch.tensor(2.0), scale=torch.tensor(1.0)):
        self.df = df
        self.scale = scale

    def __call__(self, f):
        noise_gen = \
            torch.distributions.studentT.StudentT(self.df, scale=self.scale)
        return f + noise_gen.sample(f.shape)


if __name__ == "__main__":
    with open('setups.yaml', 'r') as config_file:
        tuning_parms = yaml.safe_load(config_file)
    scenario = tuning_parms['gen']['scenario']
    n = tuning_parms[scenario]['n']
    nTrain = tuning_parms[scenario]['n_train']
    nTest = tuning_parms[scenario]['n_test']
    d = tuning_parms[scenario]['d']
    lengthscale = tuning_parms[scenario]['lengthscale']
    nugget4f = tuning_parms[scenario]['noise_f']
    scale4y = tuning_parms[scenario]['noise_y'] ** 0.5
    seedLst = tuning_parms[scenario]['seed_list']
    grid = tuning_parms[scenario]['grid']
    dfStudentT = tuning_parms[scenario]['df_student']
    outputscale = tuning_parms[scenario]['scale_f']
    kernel_name = tuning_parms[scenario]['kernel_name']
    kernel_parms = tuning_parms[scenario]['kernel_parms']
    kernel_vars = tuning_parms[scenario]['kernel_vars']
    kernel_parms.update({'ard_num_dims': d})
    kernel_vars.update({'lengthscale': lengthscale})

    with torch.no_grad():
        root = "../datasets/"
        normal_y_gen = Normal_y_gen(torch.tensor(scale4y))
        student_y_gen = Student_y_gen(torch.tensor(dfStudentT),
                                      torch.tensor(scale4y))
        logit_y_gen = Logit_y_gen()
        # generate datasets
        for seed in seedLst:
            torch.manual_seed(seed)
            if grid:
                nKnot = int(n ** (1 / d)) + 1
                edge = torch.linspace(0, 1, steps=nKnot)
                X = torch.flatten(
                    torch.stack(torch.meshgrid([edge] * d, indexing='ij')),
                    start_dim=1).t()[:n]
            else:
                X = torch.rand(n, d)
            # shuffle rows of X
            X = X[torch.randperm(n)]
            # covariance kernel of f
            K = ScaleKernel(getattr(gpytorch.kernels, kernel_name)(
                **kernel_parms))
            for kernel_var_name in kernel_vars.keys():
                setattr(K.base_kernel, kernel_var_name,
                        kernel_vars[kernel_var_name])
            K.outputscale = outputscale
            kernel4f = viva.Kernel(X, K, nugget4f)
            # generate f
            covM4f = kernel4f(list(range(n)))
            GPObj4f = MultivariateNormal(torch.zeros(n), covM4f)
            f = GPObj4f.sample()
            y_normal = normal_y_gen(f)
            y_student = student_y_gen(f)
            y_logit = logit_y_gen(f)
            fn_normal = root + f"{kernel_name}_normal_n{n}_d{d}_seed{seed}"
            fn_student = root + f"{kernel_name}_student_n{n}_d{d}_seed{seed}"
            fn_logit = root + f"{kernel_name}_logit_n{n}_d{d}_seed{seed}"
            # make directories
            Path(fn_normal + "/train").mkdir(parents=True, exist_ok=True)
            Path(fn_normal + "/test").mkdir(parents=True, exist_ok=True)
            Path(fn_student + "/train").mkdir(parents=True, exist_ok=True)
            Path(fn_student + "/test").mkdir(parents=True, exist_ok=True)
            Path(fn_logit + "/train").mkdir(parents=True, exist_ok=True)
            Path(fn_logit + "/test").mkdir(parents=True, exist_ok=True)
            # create dataframes
            X_train_df = pd.DataFrame(X[:nTrain].numpy())
            X_test_df = pd.DataFrame(X[nTrain:].numpy())
            f_train_df = pd.DataFrame(f[:nTrain].numpy())
            f_test_df = pd.DataFrame(f[nTrain:].numpy())
            y_normal_train_df = pd.DataFrame(y_normal[:nTrain].numpy())
            y_normal_test_df = pd.DataFrame(y_normal[nTrain:].numpy())
            y_student_train_df = pd.DataFrame(y_student[:nTrain].numpy())
            y_student_test_df = pd.DataFrame(y_student[nTrain:].numpy())
            y_logit_train_df = pd.DataFrame(y_logit[:nTrain].numpy())
            y_logit_test_df = pd.DataFrame(y_logit[nTrain:].numpy())
            # write to directors
            X_train_df.to_csv(fn_normal + "/train/x.csv", index=False,
                              index_label=False)
            X_test_df.to_csv(fn_normal + "/test/x.csv", index=False,
                             index_label=False)
            f_train_df.to_csv(fn_normal + "/train/f.csv", index=False,
                              index_label=False)
            f_test_df.to_csv(fn_normal + "/test/f.csv", index=False,
                             index_label=False)
            y_normal_train_df.to_csv(fn_normal + "/train/y.csv",
                                     index=False, index_label=False)
            y_normal_test_df.to_csv(fn_normal + "/test/y.csv", index=False,
                                    index_label=False)
            X_train_df.to_csv(fn_student + "/train/x.csv", index=False,
                              index_label=False)
            X_test_df.to_csv(fn_student + "/test/x.csv", index=False,
                             index_label=False)
            f_train_df.to_csv(fn_student + "/train/f.csv", index=False,
                              index_label=False)
            f_test_df.to_csv(fn_student + "/test/f.csv", index=False,
                             index_label=False)
            y_student_train_df.to_csv(fn_student + "/train/y.csv",
                                      index=False, index_label=False)
            y_student_test_df.to_csv(fn_student + "/test/y.csv",
                                     index=False, index_label=False)
            X_train_df.to_csv(fn_logit + "/train/x.csv", index=False,
                              index_label=False)
            X_test_df.to_csv(fn_logit + "/test/x.csv", index=False,
                             index_label=False)
            f_train_df.to_csv(fn_logit + "/train/f.csv", index=False,
                              index_label=False)
            f_test_df.to_csv(fn_logit + "/test/f.csv", index=False,
                             index_label=False)
            y_logit_train_df.to_csv(fn_logit + "/train/y.csv",
                                    index=False, index_label=False)
            y_logit_test_df.to_csv(fn_logit + "/test/y.csv", index=False,
                                   index_label=False)
