from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "ichol0",
        ["ichol0/ichol0.cpp",],
        extra_compile_args=["-std=c++11"]
    ),
]

setup(name="ichol0",
      version="0.0.1",
      ext_modules=ext_modules)


