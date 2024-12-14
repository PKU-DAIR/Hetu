from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

extensions = [
    Extension(
        "trainer.utils.dp_bucket",
        ["trainer/utils/dp_bucket.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp']
    ),
    Extension(
        "trainer.utils.combine_scheme_to_strategy_candidates",
        ["trainer/utils/scheme_to_strategy.pyx"],
        language="c++",
        extra_compile_args=["-std=c++17"],
    )
]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
)
