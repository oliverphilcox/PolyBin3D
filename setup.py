from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize
import numpy

# Compile pyx files with C
modules = ['utils']
for module in modules:
    ext_modules = [
        Extension(
            name=f'PolyBin3D.cython.{module}',
            sources=[f'PolyBin3D/cython/{module}.pyx'],
            libraries=['m'],
            extra_compile_args=["-fopenmp","-O3", "-ffast-math", "-march=haswell"],
            extra_link_args=["-fopenmp"],
            define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
            include_dirs=[numpy.get_include()],
            )
    ]
    
setup(
    ext_modules=cythonize(ext_modules), # Compile Cython modules and include them in the package
)