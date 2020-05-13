"""
gomory_solver.pyx

install via python setup.py build_ext --inplace

simple cython test of solving a problem with gomory cuts

the C function: c_solve_gomory solves a problem specified by 
c, var type, var ub, var lb, G, h, A, b with the following formulation

min     c^T x
s.t.    Gx <= h
        Ax  = b
        lb <= x <= ub
        x in type (integer, binary, continuous)

CPX_CONTINUOUS  C   continuous
CPX_BINARY      B   binary
CPX_INTEGER     I   general integer
CPX_SEMICONT    S   semi-continuous
CPX_SEMIINT     N   semi-integer

"""

import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as cnp

# declare the interface to the C code
cdef extern int c_gomory_solver(
        cnp.PyArrayObject* obj_coef_c,
        cnp.PyArrayObject* var_type,
        cnp.PyArrayObject* G,
        cnp.PyArrayObject* h,
        cnp.PyArrayObject* A,
        cnp.PyArrayObject* b,
        cnp.PyArrayObject* new_G,
        cnp.PyArrayObject* new_h,
        cnp.PyArrayObject* new_A,
        cnp.PyArrayObject* new_b)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef gomory_solver(
                cnp.ndarray[double, ndim=1, mode="c"] obj_coef_c,
                cnp.ndarray[char, ndim=1, mode="c"]   var_type,
                cnp.ndarray[double, ndim=2, mode="c"] G,
                cnp.ndarray[double, ndim=1, mode="c"] h,
                cnp.ndarray[double, ndim=2, mode="c"] A,
                cnp.ndarray[double, ndim=1, mode="c"] b):
    """
    gomory_solver

    Takes in information specifying a MIP and solves using gomory cutting plane methods.
    This function returns the generated cuts in the passed in numpy matrices.

    Solves MIPs in the form:

    min         obj_coef_c*x
    subject to  Gx <= h
                Ax = b
                var_lb <= x <= var_ub
                x is var_type

    :param obj_coef_c: double array of objective coefficients for all decision variables
    :param var_type: char array of variable types to be fed into cplex
    :param var_lb: double array of variable lower bounds for all decision variables
    :param var_ub: double array of variable upper bounds for all decision variables
    :param G: inequality constraint coefficient matrix G
    :param h: inequality constraint rhs
    :param A: equality constraint coefficient matrix A
    :param b: equality rhs
    :return: 0 if completes successfully, cplex error code otherwise
    """

    cdef int retval

    # initialize to pass by reference
    # these will be reshaped and re-referenced but we must initialize to something
    cdef cnp.ndarray[double, ndim=2, mode="c"] new_G = np.zeros_like(G)
    cdef cnp.ndarray[double, ndim=1, mode="c"] new_h = np.zeros_like(h)
    cdef cnp.ndarray[double, ndim=2, mode="c"] new_A = np.zeros_like(A)
    cdef cnp.ndarray[double, ndim=1, mode="c"] new_b = np.zeros_like(b)

    retval = c_gomory_solver(
        <cnp.PyArrayObject*> <void *> obj_coef_c,
        <cnp.PyArrayObject*> <void *> var_type,
        <cnp.PyArrayObject*> <void *> G, <cnp.PyArrayObject*> <void *> h,
        <cnp.PyArrayObject*> <void *> A, <cnp.PyArrayObject*> <void *> b,
        <cnp.PyArrayObject*> <void *> new_G, <cnp.PyArrayObject*> <void *> new_h,
        <cnp.PyArrayObject*> <void *> new_A, <cnp.PyArrayObject*> <void *> new_b
        )

    return new_G, new_h, new_A, new_b





