""" klujax installer """

import os
import site
import sys
from glob import glob

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

CWD = os.path.dirname(os.path.abspath(__file__))

match sys.platform:
    case "darwin":
        extra_compile_args = ["-std=c++11"]
    case _:
        extra_compile_args = []

match sys.platform:
    case "linux":
        extra_link_args = ["-static-libgcc", "-static-libstdc++"]
    case _:
        extra_link_args = []


include_dirs = [
    os.path.join(CWD, "xla"),
    os.path.join(os.path.dirname(pybind11.__file__), "include"),
    os.path.join(CWD, "suitesparse/SuiteSparse_config"),
    os.path.join(CWD, "suitesparse/AMD/Include"),
    os.path.join(CWD, "suitesparse/COLAMD/Include"),
    os.path.join(CWD, "suitesparse/BTF/Include"),
    os.path.join(CWD, "suitesparse/KLU/Include"),
]

sources = [
    "suitesparse/SuiteSparse_config/SuiteSparse_config.c",
    *glob("suitesparse/AMD/Source/*.c"),
    *glob("suitesparse/COLAMD/Source/*.c"),
    *glob("suitesparse/BTF/Source/*.c"),
    *glob("suitesparse/KLU/Source/*.c"),
]


klujax_cpp = Extension(
    name="klujax_cpp",
    sources=["klujax.cpp", *sources],
    include_dirs=include_dirs,
    library_dirs=site.getsitepackages(),
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++",
)


setup(
    name="klujax",
    version="0.2.10",
    author="Floris Laporte",
    author_email="floris.laporte@gmail.com",
    description="a KLU solver for JAX",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/flaport/klujax",
    py_modules=["klujax"],
    ext_modules=[klujax_cpp],
    cmdclass={"build_ext": build_ext},  # type: ignore
    install_requires=["jax>=0.4.35", "jaxlib>=0.4.35", "pybind11"],
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    package_data={
        "*": [
            "LICENSE",
            "README.md",
            "MANIFEST.in",
        ],
    },
)
