#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules =
        [Extension("gomory_solver",
            sources=["c_gomory_solver.c", "gomory_solver.pyx"],
            include_dirs=[numpy.get_include(), 
                "/Applications/CPLEX_Studio128/cplex/include"],
            libraries=["cplex"],
            library_dirs=["/Applications/CPLEX_Studio128/cplex/lib/x86-64_osx/static_pic/"],
            gdb_debug=True,
            )]
)
