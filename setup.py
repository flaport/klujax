import os
import sys
import site
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

python = f"python{sys.version_info.major}.{sys.version_info.minor}"
site_packages = os.path.abspath(os.path.expanduser(site.getsitepackages()[0]))
env = os.path.dirname(os.path.dirname(os.path.dirname(site_packages)))


klujax_cpp = Extension(
    name="klujax_cpp",
    sources=["klujax.cpp"],
    include_dirs=[
        os.path.join(env, "include"),
        os.path.join(env, "include", python),
        os.path.join(site_packages, "pybind11", "include"),
    ],
    library_dirs=[
        os.path.join(env, "lib"),
        os.path.join(env, "lib", python),
        site_packages,
    ],
    extra_compile_args=[],
    extra_link_args=["-static-libstdc++"],
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
    version="0.0.2",
    author="Floris Laporte",
    author_email="floris.laporte@gmail.com",
    description="a KLU solver for JAX",
    long_description=open("readme.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/flaport/klujax",
    py_modules=["klujax"],
    ext_modules=[klujax_cpp],
    cmdclass={"build_ext": build_ext},
    install_requires=["jax==0.2.14", "jaxlib==0.1.67"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
