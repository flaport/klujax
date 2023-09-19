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

os_name = os.name
if sys.platform == "darwin":
    os_name = "darwin"

pybind11_include = {
    "nt": "pybind11/include",
    "darwin": "pybind11/include",
    "posix": "pybind11/include",
}
extra_compile_args = {
    "nt": [],
    "darwin": ["-std=c++11"],  #
    "posix": [],
}
extra_link_args = {
    "nt": [],
    "darwin": [],  # "-static-libgcc", "-static-libstdc++"
    "posix": ["-static-libgcc", "-static-libstdc++"],
}

sources = [
    "suitesparse/SuiteSparse_config/SuiteSparse_config.c",
    *glob("suitesparse/AMD/Source/*.c"),
    *glob("suitesparse/COLAMD/Source/*.c"),
    *glob("suitesparse/BTF/Source/*.c"),
    *glob("suitesparse/KLU/Source/*.c"),
]


include_dirs = [
    libroot,
    pybind11_include[os_name],
    "suitesparse/SuiteSparse_config",
    "suitesparse/AMD/Include",
    "suitesparse/COLAMD/Include",
    "suitesparse/BTF/Include",
    "suitesparse/KLU/Include",
]

library_dirs = [site_packages]

suitesparse_headers = [
    *glob("suitesparse/SuiteSparse_config/*.h"),
    *glob("suitesparse/AMD/Include/*.h"),
    *glob("suitesparse/COLAMD/Include/*.h"),
    *glob("suitesparse/BTF/Include/*.h"),
    *glob("suitesparse/KLU/Include/*.h"),
]

if os_name == "darwin":
    klujax_cpp = Extension(
        name="klujax_cpp",
        sources=["klujax.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_compile_args=extra_compile_args[os_name],
        extra_link_args=extra_link_args[os_name],
        language="c++",
    )

    sparse_c = Extension(
        name="sparse_c",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_compile_args=[],
        extra_link_args=extra_link_args[os_name],
        language="c",
    )
    exts = [sparse_c, klujax_cpp]
else:
    klujax_cpp = Extension(
        name="klujax_cpp",
        sources=sources + ["klujax.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_compile_args=extra_compile_args[os_name],
        extra_link_args=extra_link_args[os_name],
        language="c++",
    )
    exts = [klujax_cpp]


setup(
    name="klujax",
    version="0.2.3",
    author="Floris Laporte",
    author_email="floris.laporte@gmail.com",
    description="a KLU solver for JAX",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/flaport/klujax",
    py_modules=["klujax"],
    ext_modules=exts,
    cmdclass={"build_ext": build_ext},  # type: ignore
    install_requires=["jax", "jaxlib", "pybind11"],
    python_requires=">=3.8",
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
