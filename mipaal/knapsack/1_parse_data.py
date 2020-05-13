import os
import pandas as pd
import numpy as np
from sklearn import preprocessing

script_dir = os.path.dirname(os.path.realpath(__file__))


def parse_kp_data(filename):
    """
    Parse file from Emir's knapsack instances
    problems each have 48 items, and 8 features
    first column is the index of the problem
    last column is the final value
    intermediate columns are the features (unnormalized)
    :param filename:
    :return: feature_mat:Txnxf , target_mat:Txnxc
    """
    raw_data = pd.read_csv(filename, skiprows=5, header=None, sep=" ").values

    instance_ids = sorted(np.unique(raw_data[:, 0]))

    scaler = preprocessing.MinMaxScaler()

    raw_data[:,1:-1] = scaler.fit_transform(raw_data[:, 1:-1])

    # stack data so it is in the shape of
    # instances, number of items, number of entries per item
    stacked_data = np.array([raw_data[raw_data[:, 0] == i] for i in instance_ids])

    # split into feature matrix and coefficient (target) matrix
    feature_mat = stacked_data[:, :, 1:-1]
    target_mat = stacked_data[:, :, -1:]
    print(f"instances:{feature_mat.shape[0]}, items:{feature_mat.shape[1]}, features:{feature_mat.shape[2]}")
    return feature_mat, target_mat


def partition_data(X, y, train_pct, val_pct, test_pct):
    """
    partition features and targets into training, validation, test
    :param X: feature matrix
    :param y: target matrix
    :param train_pct: percentage of data for training
    :param val_pct: percentage of data for validation
    :param test_pct: percentage of data for testing
    :return: X_tr, X_val, X_te, y_tr, y_val, y_te
    """
    num_instances = len(X)
    train_stop = int(train_pct * num_instances)
    val_stop = int((train_pct + val_pct) * num_instances)
    test_stop = int((train_pct + val_pct + test_pct) * num_instances)

    X_tr = X[:train_stop]
    X_val = X[train_stop:val_stop]
    X_te = X[val_stop:test_stop]

    y_tr = y[:train_stop]
    y_val = y[train_stop:val_stop]
    y_te = y[val_stop:test_stop]

    return X_tr, X_val, X_te, y_tr, y_val, y_te


if __name__ == "__main__":
    kp_file = os.path.join(script_dir, "data", "energy_data.txt")
    save_file = os.path.join(script_dir, "data", "energy_data.npz")

    X, y = parse_kp_data(kp_file)
    train_pct, val_pct, test_pct = (0.6, 0.2, 0.2)

    X_tr, X_val, X_te, y_tr, y_val, y_te = partition_data(X, y, train_pct, val_pct, test_pct)

    np.savez(save_file, X_tr, X_val, X_te, y_tr, y_val, y_te)
    print(f"saved to {save_file}")


