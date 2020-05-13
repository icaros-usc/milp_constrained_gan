import cplex
from torch.utils import data
import numpy as np
import torch
from functools import reduce
import operator
from torch import nn
from sklearn import metrics
from mipaal.utils import cplex_utils
from matplotlib import colors
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt


def item2name(item):
    return f"x{item}"


def knapsack_constraint_name():
    return "knapsack"


def make_knapsack_problem(num_items: int, budget: int = 5, verbose: int = 0):
    """
    Creates the constraints and variables for a knapsack instance
    Does not contain objective coefficients
    :param num_items: number of items in knapsack
    :param budget: budget for number of items to include
    :param verbose: verbosity for cplex model
    :return:
    """

    cpx = cplex.Cplex()

    # add decision variables per item
    cpx.variables.add(types=["I" for _ in range(num_items)],
                      names=[item2name(i) for i in range(num_items)],
                      lb=[0.0 for _ in range(num_items)],
                      ub=[1.0 for _ in range(num_items)])

    # add budget constraint
    cpx.linear_constraints.add(names=[knapsack_constraint_name()],
                               senses=["E"],
                               rhs=[budget])

    # make variable coefficients in budget constraint 1 (unit weighted)
    cpx.linear_constraints.set_linear_components(knapsack_constraint_name(),
                                                 [[item2name(i) for i in range(num_items)],
                                                  [1.0 for _ in range(num_items)]])

    if verbose == 0:
        cpx.set_results_stream(None)
        cpx.set_log_stream(None)

    return cpx


class Net(nn.Module):
    def __init__(self, X, Y, hidden_layer_sizes, dropout):
        super(Net, self).__init__()
        # Initialize linear layer with least squares solution
        flat_X = X.reshape(-1, X.shape[-1]).cpu().numpy()
        flat_y = Y.reshape(-1, Y.shape[-1]).cpu().numpy()
        X_ = np.hstack([flat_X, np.ones((flat_X.shape[0], 1))])
        Theta = np.linalg.solve(X_.T.dot(X_), X_.T.dot(flat_y))

        self.lin = nn.Linear(flat_X.shape[1], flat_y.shape[1])
        W, b = self.lin.parameters()
        W.data = torch.Tensor(Theta[:-1, :].T)
        b.data = torch.Tensor(Theta[-1, :])

        W.requires_grad = False
        b.requires_grad = False

        # from IPython import embed; import sys; embed(); sys.exit(1)

        # Set up non-linear network of
        # Linear -> BatchNorm -> LeakyReLU -> Dropout layers
        layer_sizes = [flat_X.shape[1]] + hidden_layer_sizes
        layers = reduce(operator.add,
                        [[nn.Linear(a, b),
                          # nn.InstanceNorm1d(b),
                          nn.LeakyReLU(),
                          nn.Dropout(p=dropout)]
                         for a, b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], flat_y.shape[1])]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # from IPython import embed; import sys; embed(); sys.exit(1)
        # softmax = torch.nn.Softmax(dim=1)
        # return softmax(self.lin(x) + self.net(x))
        return self.lin(x)+self.net(x)


class KPDataset(data.Dataset):
    def __init__(self, X, y):
        self.X = X.float()
        self.y = y.float()

    def __len__(self):
        # return 1
        return self.X.shape[0]

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        return X, y


def evaluate_net(cpx, net, dataset):
    obj_vals = []

    # from IPython import embed; import sys; embed(); sys.exit(1)
    for X, y in dataset:
        c_true = y
        pred_coefs = net(X[None, :])[0]
        numpy_pred_coefs = pred_coefs.detach().numpy()
        pred_cpx = cplex_utils.set_cplex_objective(cpx,
                                                   -numpy_pred_coefs)
        pred_cpx.objective.set_sense(cpx.objective.sense.minimize)
        pred_cpx.solve()
        true_obj_val = np.array(pred_cpx.solution.get_values()) @ c_true.detach().numpy()
        obj_vals.append(true_obj_val)

    pred_y = net(dataset.X)

    y_pred = pred_y.detach().numpy().flatten()
    y_true = dataset.y.flatten()

    corr = np.corrcoef(y_true, y_pred)[1, 0]
    mse = metrics.mean_squared_error(y_true, y_pred)

    return obj_vals, corr, mse


def scatter_predictions(net, dataset):
    y_pred = net(dataset.X).detach().numpy().flatten().ravel()
    y_true = dataset.y.numpy().flatten().ravel()
    fig = plt.figure()
    plt.hist2d(y_true, y_pred, (50, 50), range=[[0, 800], [0, 800]], norm=colors.LogNorm(), cmap=plt.cm.jet)
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title("pred vs actual")
    #
    # fig2 = plt.figure()
    # ax = fig2.add_subplot(111)
    # plt.scatter(y_true, y_pred, alpha=0.1, marker="+")
    # ax.set_xlim(0, 800)
    # ax.set_ylim(0, 800)
    # plt.xlabel("y_true")
    # plt.ylabel("y_pred")
    # plt.title("pred vs actual")

    return fig


def create_data_generators(kp_file: str, batch_size: int):
    npz_file = np.load(kp_file, allow_pickle=True)
    # from IPython import embed; import sys; embed(); sys.exit(1)
    X_tr, X_val, X_te, y_tr, y_val, y_te = [torch.from_numpy(npz_file[f]) for f in npz_file.files]

    training_set = KPDataset(X_tr, y_tr)
    train_generator = data.DataLoader(training_set, shuffle=True, batch_size=batch_size)

    validation_set = KPDataset(X_val, y_val)
    validation_generator = data.DataLoader(validation_set, batch_size=1)

    test_set = KPDataset(X_te, y_te)
    test_generator = data.DataLoader(test_set, batch_size=1)

    return train_generator, validation_generator, test_generator
