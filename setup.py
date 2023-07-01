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
        "./suitesparse/AMD/Include",
        "./suitesparse/BTF/Include",
        "./suitesparse/CAMD/Include",
        "./suitesparse/CCOLAMD/Include",
        "./suitesparse/CHOLMOD/Demo",
        "./suitesparse/CHOLMOD/GPU",
        "./suitesparse/CHOLMOD/Include",
        "./suitesparse/CHOLMOD/MATLAB",
        "./suitesparse/CHOLMOD/Partition",
        "./suitesparse/CHOLMOD/SuiteSparse_metis/GKlib",
        "./suitesparse/CHOLMOD/SuiteSparse_metis/GKlib/original",
        "./suitesparse/CHOLMOD/SuiteSparse_metis/include",
        "./suitesparse/CHOLMOD/SuiteSparse_metis/include/original",
        "./suitesparse/CHOLMOD/SuiteSparse_metis/libmetis",
        "./suitesparse/CHOLMOD/SuiteSparse_metis/programs",
        "./suitesparse/CHOLMOD/Tcov",
        "./suitesparse/COLAMD/Include",
        "./suitesparse/CSparse/Demo",
        "./suitesparse/CSparse/Include",
        "./suitesparse/CSparse/MATLAB/CSparse",
        "./suitesparse/CSparse/Tcov",
        "./suitesparse/CXSparse/Demo",
        "./suitesparse/CXSparse/Include",
        "./suitesparse/CXSparse/MATLAB/CSparse",
        "./suitesparse/CXSparse/Tcov",
        "./suitesparse/Example/Include",
        "./suitesparse/KLU/Include",
        "./suitesparse/KLU/User",
        "./suitesparse/LDL/Include",
        "./suitesparse/MATLAB_Tools/SFMULT",
        "./suitesparse/MATLAB_Tools/sparseinv",
        "./suitesparse/MATLAB_Tools/spok",
        "./suitesparse/MATLAB_Tools/SSMULT",
        "./suitesparse/MATLAB_Tools/waitmex",
        "./suitesparse/Mongoose/External/mmio/Include",
        "./suitesparse/RBio/Include",
        "./suitesparse/SPEX/Include",
        "./suitesparse/SPEX/SPEX_Left_LU/Demo",
        "./suitesparse/SPEX/SPEX_Left_LU/MATLAB/Source",
        "./suitesparse/SPEX/SPEX_Left_LU/Source",
        "./suitesparse/SPEX/SPEX_Left_LU/Tcov",
        "./suitesparse/SPEX/SPEX_Util/Source",
        "./suitesparse/SPQR/Include",
        "./suitesparse/SuiteSparse_config",
        "./suitesparse/UMFPACK/Demo",
        "./suitesparse/UMFPACK/Include",
        "./suitesparse/UMFPACK/Source",
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
    extra_compile_args=["-std=c++11"] if sys.platform=="darwin" else [],
    extra_link_args= [] if sys.platform=="darwin" else ["-static-libgcc", "-static-libstdc++"],
    extra_objects=[
        "./suitesparse/SuiteSparse_config/build/CMakeFiles/FortranCInterface/libsymbols.a",
        "./suitesparse/SuiteSparse_config/build/CMakeFiles/FortranCInterface/libmyfort.a",
        "./suitesparse/SuiteSparse_config/build/libsuitesparseconfig.a",
        "./suitesparse/AMD/build/libamd.a",
        "./suitesparse/COLAMD/build/libcolamd.a",
        "./suitesparse/BTF/build/libbtf.a",
        "./suitesparse/KLU/build/libklu.a",
        #"./suitesparse/KLU/build/libklu_cholmod.a",
    ],
    libraries=[
#       "klu",
#       "btf",
#       "amd",
#       "colamd",
#       "suitesparseconfig",
    ],
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
