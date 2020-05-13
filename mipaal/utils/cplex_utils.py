import cplex
import numpy as np


# These are the main functions eating up memory, try to find a way to better track data

# @profile
def cplex_to_matrices(cpx: cplex.Cplex) -> (np.ndarray,
                                            np.ndarray, np.ndarray,
                                            np.ndarray, np.ndarray,
                                            np.ndarray):
    """
    converts cplex model to standardized minimization problem
    ready to feed into gomory/QP solvers
    note: does not change cplex object to standard form!
    :param cpx: Cplex object containing model information
    :return: c, G, h, A, b, var_type specifying optimization problem
    """

    assert cpx.get_problem_type() == cpx.problem_type.MILP, \
        "currently only converts MILP, was {}".format(cpx.get_problem_type())

    # get array of objective coefficient
    c = np.array(cpx.objective.get_linear())
    if cpx.objective.get_sense() == cpx.objective.sense.maximize:
        c = -c

    # get variable types, and modify binary variables to integral
    # note that if the variable is binary, the bounds will be set
    var_type = np.array(cpx.variables.get_types())
    for var_ind, v_type in enumerate(var_type):
        if v_type == cpx.variables.type.binary:
            assert cpx.variables.get_lower_bounds(var_ind) == 0, \
                "binary variable must have lower bound of 0"
            assert cpx.variables.get_upper_bounds(var_ind) == 1, \
                "binary variable must have upper bound of 1"
            var_type[var_ind] = "I"

    # get constraints
    num_vars = cpx.variables.get_num()
    num_cons = cpx.linear_constraints.get_num()

    # get num equality, initialize A, b
    # Ax = b
    con_senses = cpx.linear_constraints.get_senses()
    num_equality = sum([sen == 'E' for sen in con_senses])
    A = np.zeros((num_equality, num_vars))
    b = np.zeros(num_equality)

    # get num inequality, initialize G, h
    # Gx <= h
    num_inequality = num_cons - num_equality
    G = np.zeros((num_inequality, num_vars))
    h = np.zeros(num_inequality)

    # to keep track of the current (in)equality constraint index
    current_eq_ind = 0
    current_ineq_ind = 0

    # iterate over sparsepairs
    for row_ind, row in enumerate(cpx.linear_constraints.get_rows()):
        con_sense = cpx.linear_constraints.get_senses(row_ind)
        rhs = cpx.linear_constraints.get_rhs(row_ind)
        # if equality constraint, store into A and b
        if con_sense == "E":
            for var_ind, coef in zip(row.ind, row.val):
                A[current_eq_ind, var_ind] = coef
            b[current_eq_ind] = rhs
            current_eq_ind += 1
        # if <= constraint, store into G and h
        if con_sense == "L":
            for var_ind, coef in zip(row.ind, row.val):
                G[current_ineq_ind, var_ind] = coef
            h[current_ineq_ind] = rhs
            current_ineq_ind += 1
        # if >= constraint, store negation into G and h
        if con_sense == "G":
            for var_ind, coef in zip(row.ind, row.val):
                G[current_ineq_ind, var_ind] = -coef
            h[current_ineq_ind] = -rhs
            current_ineq_ind += 1

    # compute lower bound constraint matrix and rhs
    # needs to be negated since we need to format it as Gx <= h
    lb_constraints = [(var_ind, lb) for var_ind, lb in enumerate(cpx.variables.get_lower_bounds())
                      if lb > -cplex.infinity]
    lb_mat = np.zeros((len(lb_constraints), num_vars))
    lb_rhs = np.zeros(len(lb_constraints))
    for con_ind, (var_ind, lb) in enumerate(lb_constraints):
        lb_mat[con_ind, var_ind] = -1
        lb_rhs[con_ind] = -lb

    # compute upper bound constraint matrix and rhs
    ub_constraints = [(var_ind, ub) for var_ind, ub in enumerate(cpx.variables.get_upper_bounds())
                      if ub < cplex.infinity]
    ub_mat = np.zeros((len(ub_constraints), num_vars))
    ub_rhs = np.zeros(len(ub_constraints))
    for con_ind, (var_ind, ub) in enumerate(ub_constraints):
        ub_mat[con_ind, var_ind] = 1
        ub_rhs[con_ind] = ub

    G = np.concatenate([G, lb_mat, ub_mat])
    h = np.concatenate([h, lb_rhs, ub_rhs])

    return c, G, h, A, b, var_type.astype("bytes")


def cplex_to_sparse_matrices(cpx: cplex.Cplex):
    """
    converts cplex model to sparse standardized minimization problem
    ready to feed into gomory/QP solvers
    note: does not change cplex object to standard form!
    :param cpx: Cplex object containing model information
    :return: c, G, h, A, b, var_type sparse specifying optimization problem
    """

    assert cpx.get_problem_type() == cpx.problem_type.MILP, \
        "currently only converts MILP, was {}".format(cpx.get_problem_type())

    # get array of objective coefficient
    c = np.array(cpx.objective.get_linear())
    if cpx.objective.get_sense() == cpx.objective.sense.maximize:
        c = -c

    # get variable types, and modify binary variables to integral
    # note that if the variable is binary, the bounds will be set
    var_type = np.array(cpx.variables.get_types())
    for var_ind, v_type in enumerate(var_type):
        if v_type == cpx.variables.type.binary:
            assert cpx.variables.get_lower_bounds(var_ind) == 0, \
                "binary variable must have lower bound of 0"
            assert cpx.variables.get_upper_bounds(var_ind) == 1, \
                "binary variable must have upper bound of 1"
            var_type[var_ind] = "I"

    # get constraints
    num_vars = cpx.variables.get_num()
    num_cons = cpx.linear_constraints.get_num()

    # get num equality, initialize A, b
    # Ax = b
    con_senses = cpx.linear_constraints.get_senses()
    num_equality = sum([sen == 'E' for sen in con_senses])
    A = np.zeros((num_equality, num_vars))
    b = np.zeros(num_equality)

    # get num inequality, initialize G, h
    # Gx <= h
    num_inequality = num_cons - num_equality
    G = np.zeros((num_inequality, num_vars))
    h = np.zeros(num_inequality)

    # to keep track of the current (in)equality constraint index
    current_eq_ind = 0
    current_ineq_ind = 0

    # iterate over sparsepairs
    for row_ind, row in enumerate(cpx.linear_constraints.get_rows()):
        con_sense = cpx.linear_constraints.get_senses(row_ind)
        rhs = cpx.linear_constraints.get_rhs(row_ind)
        # if equality constraint, store into A and b
        if con_sense == "E":
            for var_ind, coef in zip(row.ind, row.val):
                A[current_eq_ind, var_ind] = coef
            b[current_eq_ind] = rhs
            current_eq_ind += 1
        # if <= constraint, store into G and h
        if con_sense == "L":
            for var_ind, coef in zip(row.ind, row.val):
                G[current_ineq_ind, var_ind] = coef
            h[current_ineq_ind] = rhs
            current_ineq_ind += 1
        # if >= constraint, store negation into G and h
        if con_sense == "G":
            for var_ind, coef in zip(row.ind, row.val):
                G[current_ineq_ind, var_ind] = -coef
            h[current_ineq_ind] = -rhs
            current_ineq_ind += 1

    # compute lower bound constraint matrix and rhs
    # needs to be negated since we need to format it as Gx <= h
    lb_constraints = [(var_ind, lb) for var_ind, lb in enumerate(cpx.variables.get_lower_bounds())
                      if lb > -cplex.infinity]
    lb_mat = np.zeros((len(lb_constraints), num_vars))
    lb_rhs = np.zeros(len(lb_constraints))
    for con_ind, (var_ind, lb) in enumerate(lb_constraints):
        lb_mat[con_ind, var_ind] = -1
        lb_rhs[con_ind] = -lb

    # compute upper bound constraint matrix and rhs
    ub_constraints = [(var_ind, ub) for var_ind, ub in enumerate(cpx.variables.get_upper_bounds())
                      if ub < cplex.infinity]
    ub_mat = np.zeros((len(ub_constraints), num_vars))
    ub_rhs = np.zeros(len(ub_constraints))
    for con_ind, (var_ind, ub) in enumerate(ub_constraints):
        ub_mat[con_ind, var_ind] = 1
        ub_rhs[con_ind] = ub

    G = np.concatenate([G, lb_mat, ub_mat])
    h = np.concatenate([h, lb_rhs, ub_rhs])

    return c, G, h, A, b, var_type.astype("bytes")


# @profile
def matrices_to_cplex(c: np.ndarray,
                      G: np.ndarray, h: np.ndarray,
                      A: np.ndarray, b: np.ndarray,
                      var_type: np.ndarray,
                      verbose: int = 0,
                      epsilon: float = 1e-4) -> cplex.Cplex:
    """
    converts matrix format to cplex object of the form
    min         cTx
    subject to  Gx <= h
                Ax = b
                x is var_type

    :param c: objective coefficients
    :param G: inequality constraint coefficients
    :param h: inequality constraint rhs
    :param A: equality constraint coefficients
    :param b: equality constraint rhs
    :param var_type: variable type, integer or continuous
    :param verbose: verbosity, if 0 then hide all output
    :return: a cplex object specifying the
    """
    cpx = cplex.Cplex()

    # add variables with objective coefficients
    # and variable types

    cpx.variables.add(obj=list(map(float, c)), types=np.array(var_type).astype("str"))
    cpx.objective.set_sense(cpx.objective.sense.minimize)

    # subject to
    #     G * x <= h
    if G is not None:
        for i in range(G.shape[0]):
            cpx.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=range(G.shape[1]), val=[float(x) for x in G[i]])],
                senses=['L'],
                rhs=[h[i]])

    # subject to
    #     A * x == b
    if A is not None:
        for i in range(A.shape[0]):
            cpx.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=range(A.shape[1]), val=[float(x) for x in A[i]])],
                senses=['E'],
                rhs=[b[i]])

    if verbose == 0:
        cpx.set_results_stream(None)
        cpx.set_log_stream(None)

    cpx.cleanup(epsilon=epsilon)
    cpx.set_results_stream(None)
    cpx.set_log_stream(None)
    cpx.set_warning_stream(None)

    return cpx


def set_cplex_objective(cpx: cplex.Cplex, c, Q=None,
                        epsilon: float = 1e-4) -> cplex.Cplex:
    """
    set_cplex_objective(cpx, c, Q) sets the objective of cplex object cpx
    to have linear objective coefficients c
    and quadratic objective coefficients Q
    essentially sets problem to have objective of the form

    min cTx + xT Q x

    with original constraints etc
    :param cpx: cplex object to modify
    :param c: linear objective coefficients
    :param Q: quadratic objective coefficients
    :return: cplex object with set objective coefficients
    """
    n = cpx.variables.get_num()

    assert len(c) == n, "c must have {} items but len(c) is {}".format(
        n, len(c)
    )

    # set linear coefficients
    #       cTx
    for i in range(n):
        cpx.objective.set_linear(i, float(c[i]))

    if Q is not None and len(Q.nonzero()[0]) > 0:
        assert len(Q) == n, "Q must have {} items but len(Q) is {}".format(
            n, len(Q)
        )
        # quadratic coefficients
        #     x.T * Q * x
        quadratic_coefs = []
        for i in range(Q.shape[0]):
            indices = Q[i].nonzero()[0].tolist()
            if len(indices) == 0:
                sparse_pair = cplex.SparsePair(ind=[], val=[])
            else:
                vals = Q[i, indices].tolist()
                sparse_pair = cplex.SparsePair(ind=indices,
                                               val=vals)
            quadratic_coefs.append(sparse_pair)

        cpx.objective.set_quadratic(quadratic_coefs)
    else:
        cpx.set_problem_type(cpx.problem_type.MILP)

    cpx.objective.set_sense(cpx.objective.sense.minimize)
    cpx.cleanup(epsilon=epsilon)
    return cpx
