from setuptools import setup, find_packages
from setuptools import Extension
from torch.utils import cpp_extension

setup(name='maxmin_cpp',
      ext_modules=[cpp_extension.CppExtension('maxmin_cpp', ['maxMin.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
