from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

# Compile pyx files with C
modules = ['utils']
for module in modules:
    ext_modules = [
        Extension(
            module,
            [module+'.pyx'],
            libraries=['mvec','m'],
            extra_compile_args=["-fopenmp","-O3", "-ffast-math", "-march=broadwell"],
            extra_link_args=["-fopenmp"],
            define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
            )
    ]
    setup(
        name=module,
        ext_modules=cythonize(ext_modules),
        include_dirs=[numpy.get_include()]
    )