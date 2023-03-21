from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

ext_modules_omp = [
    Pybind11Extension(
        "indexing",
        ["indexing/indexing.cpp",],
        extra_compile_args=["-O3", "-fopenmp", "-std=c++11"],
        extra_link_args=["-fopenmp"],
    ),
]

ext_modules_seq = [
    Pybind11Extension(
        "indexing",
        ["indexing/indexing.cpp",],
        extra_compile_args=["-O3", "-std=c++11"],
    ),
]

try:
    setup(name="indexing",
          version="0.0.1",
          ext_modules=ext_modules_omp)
except:
    print("Installation of the OMP parallel feature was not successfully. "
          "Sequentially implementation is used.")
    setup(name="indexing",
          version="0.0.1",
          ext_modules=ext_modules_seq)


