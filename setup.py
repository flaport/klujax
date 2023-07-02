""" klujax installer """

import os
import sys
import site
from glob import glob
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

python = f"python{sys.version_info.major}.{sys.version_info.minor}"
site_packages = os.path.abspath(os.path.expanduser(site.getsitepackages()[0]))
env = os.path.dirname(os.path.dirname(os.path.dirname(site_packages)))
libroot = os.path.dirname(os.path.dirname(os.__file__))
if os.name == "nt":  # Windows
    pybind11_include = os.path.join(libroot, "Library", "include")
else:  # Linux / Mac OS
    pybind11_include = os.path.join(os.path.dirname(libroot), "include")

klujax_cpp = Extension(
    name="klujax_cpp",
    sources=[
        "suitesparse/SuiteSparse_config/SuiteSparse_config.c",
        *glob("suitesparse/AMD/Source/*.c"),
        *glob("suitesparse/COLAMD/Source/*.c"),
        *glob("suitesparse/BTF/Source/*.c"),
        *glob("suitesparse/KLU/Source/*.c"),
        "klujax.cpp",
    ],
    include_dirs=[
        libroot,
        pybind11_include,
        "suitesparse/SuiteSparse_config",
        "suitesparse/AMD/Include",
        "suitesparse/COLAMD/Include",
        "suitesparse/BTF/Include",
        "suitesparse/KLU/Include",
    ],
    library_dirs=[
        site_packages,
    ],
    extra_compile_args=["-std=c++11"] if sys.platform=="darwin" else [],
    extra_link_args= [] if sys.platform=="darwin" else ["-static-libgcc", "-static-libstdc++"],
    language="c++",
)

setup(
    name="klujax",
    version="0.1.4",
    author="Floris Laporte",
    author_email="floris.laporte@gmail.com",
    description="a KLU solver for JAX",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/flaport/klujax",
    py_modules=["klujax"],
    ext_modules=[klujax_cpp],
    cmdclass={"build_ext": build_ext},  # type: ignore
    install_requires=["jax", "jaxlib"],
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
        ],
    },
)
