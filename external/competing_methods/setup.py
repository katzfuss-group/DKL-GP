from setuptools import setup, find_packages
from setuptools import Extension

setup(
    name="gps",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["numpy>=1.17.3", "torch>=1.3.0", "gpytorch>=0.3.6", "matplotlib>=3.1.1"],
)
