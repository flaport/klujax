""" klujax installer """

import os
import sys
import site
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

python = f"python{sys.version_info.major}.{sys.version_info.minor}"
site_packages = os.path.abspath(os.path.expanduser(site.getsitepackages()[0]))
env = os.path.dirname(os.path.dirname(os.path.dirname(site_packages)))

libroot = os.path.dirname(os.path.dirname(os.__file__))
if os.name == "nt":  # Windows
    suitesparse_lib = os.path.join(libroot, "Library", "lib")
    suitesparse_include = os.path.join(libroot, "Library", "include", "suitesparse")
    pybind11_include = os.path.join(libroot, "Library", "include")
else:  # Linux / Mac OS
    suitesparse_lib = os.path.join(os.path.dirname(libroot), "lib")
    suitesparse_include = os.path.join(os.path.dirname(libroot), "include")
    pybind11_include = os.path.join(os.path.dirname(libroot), "include")

klujax_cpp = Extension(
    name="klujax_cpp",
    sources=["klujax.cpp"],
    include_dirs=[
        f"{env}/include",
        f"/usr/include",
        f"{env}/include/suitesparse",
        f"/usr/include/suitesparse",
        f"{env}/include/{python}",
        f"{site_packages}/pybind11/include",
        suitesparse_include,
        pybind11_include,
    ],
    library_dirs=[
        f"{env}/lib",
        f"/usr/lib",
        f"{env}/lib64",
        f"/usr/lib64",
        f"{env}/lib/{python}",
        f"{env}/lib64/{python}",
        f"{site_packages}",
        suitesparse_lib,
    ],
    extra_compile_args=[],
    extra_link_args=["-static-libgcc", "-static-libstdc++"],
    libraries=[
        "klu",
        "btf",
        "amd",
        "colamd",
        "suitesparseconfig",
    ],
    language="c++",
)

setup(
    name="klujax",
    version="0.1.2",
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
