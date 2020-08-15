import torch
from torch.autograd import Function
from mipaal.qpthlocal.qp import QPFunction
from mipaal.qpthlocal.qp import QPSolvers
from mipaal.utils import cplex_utils
import subprocess
import os
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np

script_path = os.path.abspath(os.path.dirname(__file__))


class MIPFunction(Function):
    def __init__(self, var_type, G, h, A, b, eps=1e-12, verbose=0, notImprovedLim=3, maxIter=20, model_params=None,
                 custom_solver=None, input_mps="gomory_prob.mps", gomory_limit=-1,
                 test_timing=False, test_integrality=False,
                 test_cuts_generated=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gomory_limit = gomory_limit
        self.qp_function = QPFunction(solver=QPSolvers.GUROBI, verbose=True)
        self.var_type = var_type
        self.verbose = verbose
        self.input_mps = input_mps
        self.G = G
        self.h = h
        self.A = A
        self.b = b
        self.test_timing = test_timing
        self.test_integrality = test_integrality
        self.test_cuts_generated = test_cuts_generated
        if self.test_timing:  # TODO get timing
            # timings are in seconds
            self.time_mipsolve = 0
            self.time_backprop = 0
        if self.test_integrality:  # TODO get integrality
            # % of integral variables not equal to optimal value
            self.percent_same = 0
            self.percent_integer = 0
            self.avg_difference = 0
        if self.test_cuts_generated:  # TODO get cuts generated
            self.num_cuts_generated = 0
            self.num_initial_cuts = A.shape[0] + G.shape[0]

    def forward(self, Q, p, G, h, A, b):
        cpx = cplex_utils.matrices_to_cplex(c=p.detach().numpy().astype('float64'),
                                            G=G.numpy().astype('float64'), h=h.numpy().astype('float64'),
                                            A=A.numpy().astype('float64'), b=b.numpy().astype('float64'),
                                            var_type=self.var_type)
        # write to mps file
        cpx.write(self.input_mps)

        start = time.time()
        # solve with script
        if self.gomory_limit == -1:
            subprocess.check_output(
                [os.path.join(".", script_path, "gomory_solver_cplex", "cpx_gomory"), self.input_mps, self.input_mps])
        else:
            subprocess.check_output(
                [os.path.join(".", script_path, "gomory_solver_cplex", "gomory_limited"), self.input_mps,
                 self.input_mps, str(self.gomory_limit)])

        # read with cplex
        cpx.read(self.input_mps)

        # get matrices from cplex
        c_new, G_new, h_new, A_new, b_new, var_type_new = cplex_utils.cplex_to_matrices(cpx)

        end = time.time()
        if self.test_timing:
            self.time_mipsolve += end - start
            print(self.time_mipsolve)

        # from IPython import embed; import sys; embed(); sys.exit(1)

        # how many are integral fraction of integer variables that are non-integral
        if self.test_integrality:
            vartypes = np.array(self.var_type)
            integral_vars = (vartypes == b'B') | (vartypes == b'I')

            cpx.solve()

            LP_solution_values = np.array(cpx.solution.get_values())

            cpx = cplex_utils.matrices_to_cplex(c=p.detach().numpy(),
                                                G=G.numpy(), h=h.numpy(),
                                                A=A.numpy(), b=b.numpy(),
                                                var_type=self.var_type)
            cpx.solve()

            MIP_solution_values = np.array(cpx.solution.get_values())

            self.percent_same = np.mean(
                np.isclose(LP_solution_values[integral_vars], MIP_solution_values[integral_vars]))
            self.percent_integer = np.mean(
                np.isclose(LP_solution_values[integral_vars], np.round(LP_solution_values[integral_vars])))
            self.avg_difference = np.mean(
                np.abs(LP_solution_values[integral_vars] - MIP_solution_values[integral_vars]))
            # print(self.percent_same, self.percent_integer, self.avg_difference)

        if self.test_cuts_generated:
            num_cons = G_new.shape[0] + A_new.shape[0]
            self.num_cuts_generated = num_cons - self.num_initial_cuts
            # print("num_cuts", self.num_cuts_generated, self.num_initial_cuts)
            # print("is integral:", is_integral)
            # abs_differences = np.abs(solution_values - true_integral_soln)
            # abs_differences

            # print("Q:", Q.shape)
            # print("c:", c_new.shape)
            # print("G:", G_new.shape)
            # print("h:", h_new.shape)
            # print("A:", A_new.shape)
            # print("b:", b_new.shape)
            # if self.verbose:
            # self.G = torch.tensor(G_new)
            # self.h = torch.tensor(h_new)
            # self.A = torch.tensor(A_new)
            # self.b = torch.tensor(b_new)

        # from IPython import embed; import sys; embed(); sys.exit(1)

        # return self.qp_function(Q, p, self.G, self.h, self.A, self.b)
        return self.qp_function(Q, p, torch.tensor(G_new),
                                torch.tensor(h_new),
                                torch.tensor(A_new),
                                torch.tensor(b_new))

    def backward(self, dl_dzhat):
        start = time.time()
        back = self.qp_function.backward(dl_dzhat)
        end = time.time()
        if self.test_timing:
            self.time_backprop += end - start
            # print("backprop time", self.time_backprop)
        return back

    def release(self):
        del self.qp_function

        del self.var_type
        del self.input_mps

        del self.G
        del self.h
        del self.A
        del self.b


class LPFunction(Function):
    def __init__(self, solver, verbose=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qp_function = QPFunction(solver=solver, verbose=verbose)

    # @profile
    def forward(self, Q, p, G, h, A, b):
        sol = self.qp_function(Q, p, G, h, A, b)
        return sol

    # @profile
    def backward(self, dl_dzhat):
        back = self.qp_function.backward(dl_dzhat)
        # del self.qp_function
        return back
