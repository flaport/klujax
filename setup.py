""" KLUJAX Setup. """

import os
import site
import sys
from glob import glob

from setuptools import Extension, setup
from setuptools._distutils._log import log
from setuptools._distutils._modified import newer_group
from setuptools.command.build_ext import build_ext

CWD = os.path.dirname(os.path.abspath(__file__))

include_dirs = [
    os.path.join(CWD, "xla"),
    os.path.join(CWD, "pybind11", "include"),
    os.path.join(CWD, "suitesparse", "SuiteSparse_config"),
    os.path.join(CWD, "suitesparse", "AMD", "Include"),
    os.path.join(CWD, "suitesparse", "COLAMD", "Include"),
    os.path.join(CWD, "suitesparse", "BTF", "Include"),
    os.path.join(CWD, "suitesparse", "KLU", "Include"),
]

suitesparse_sources = [
    os.path.join(CWD, "suitesparse", "SuiteSparse_config", "SuiteSparse_config.c"),
    *glob(os.path.join(CWD, "suitesparse", "AMD", "Source", "*.c")),
    *glob(os.path.join(CWD, "suitesparse", "COLAMD", "Source", "*.c")),
    *glob(os.path.join(CWD, "suitesparse", "BTF", "Source", "*.c")),
    *glob(os.path.join(CWD, "suitesparse", "KLU", "Source", "*.c")),
]


if sys.platform == "linux":  # gcc
    extension = Extension(
        name="klujax_cpp",
        sources=["klujax.cpp", *suitesparse_sources],
        include_dirs=include_dirs,
        library_dirs=site.getsitepackages(),
        extra_compile_args=["-std=c++17"],
        extra_link_args=["-static-libgcc", "-static-libstdc++"],
        language="c++",
    )
elif sys.platform == "win32":  # cl
    extension = Extension(
        name="klujax_cpp",
        sources=["klujax.cpp", *suitesparse_sources],
        include_dirs=include_dirs,
        library_dirs=site.getsitepackages(),
        extra_compile_args=["/std:c++17"],
        extra_link_args=[],
        language="c++",
    )
elif sys.platform == "darwin":  # MacOS: clang
    extension = Extension(
        name="klujax_cpp",
        sources=["klujax.cpp", *suitesparse_sources],
        include_dirs=include_dirs,
        library_dirs=site.getsitepackages(),
        extra_compile_args=["-std=c++17"],
        extra_link_args=[],
        language="c++",
    )
else:
    raise RuntimError(f"Platform {sys.platform} not supported.")


# Custom BuildExt to enable combined build of C and C++ files on MacOs (clang)
# However, this class also removes some warnings when used on linux (gcc) and
# Windows (cl) so we use it everywhere.
class BuildExt(build_ext):
    def build_extension(self, ext):
        sources = ext.sources
        c_sources = sorted([s for s in sources if s.endswith("c")])
        cpp_sources = sorted([s for s in sources if s not in c_sources])
        ext_path = self.get_ext_fullpath(ext.name)
        macros = ext.define_macros[:]
        for undef in ext.undef_macros:
            macros.append((undef,))
        c_objects = self.compiler.compile(
            c_sources,
            output_dir=self.build_temp,
            macros=macros,
            include_dirs=ext.include_dirs,
            debug=self.debug,
            extra_postargs=[
                f
                for f in ext.extra_compile_args
                if not f in ["-std=c++17", "/std:c++17"]  # THIS IS OUR HACK
            ],
            depends=ext.depends,
        )
        cpp_objects = self.compiler.compile(
            cpp_sources,
            output_dir=self.build_temp,
            macros=macros,
            include_dirs=ext.include_dirs,
            debug=self.debug,
            extra_postargs=ext.extra_compile_args,
            depends=ext.depends,
        )
        objects = c_objects + cpp_objects
        extra_args = ext.extra_link_args or []
        self.compiler.link_shared_object(
            objects,
            ext_path,
            libraries=self.get_libraries(ext),
            library_dirs=ext.library_dirs,
            runtime_library_dirs=ext.runtime_library_dirs,
            extra_postargs=extra_args,
            export_symbols=self.get_export_symbols(ext),
            debug=self.debug,
            build_temp=self.build_temp,
            target_lang=ext.language,
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
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExt},
    install_requires=["jax>=0.4.35", "jaxlib>=0.4.35"],
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
