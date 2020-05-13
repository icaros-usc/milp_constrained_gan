import torch
import numpy as np
from knapsack import knapsack_utils
import os
from utils import experiment_utils
import pickle
import argparse
from torch.utils.tensorboard import SummaryWriter
import shutil
import time
script_dir = os.path.dirname(os.path.realpath(__file__))


# run with
# python -m knapsack.2_train_baseline

# from
# from IPython import embed; import sys; embed(); sys.exit()


if __name__ == "__main__":

    global_start = time.time()

    print("knapsack_dir:", script_dir)

    # set up problem read from experiment file
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir",
                        default=os.path.join(script_dir, "experiments", "example_experiment"),
                        help="experiment directory that specifies setup in params.json and where data is written")

    args = parser.parse_args()
    param_file = os.path.join(args.experiment_dir, "params.json")
    assert os.path.isfile(param_file), "No json configuration file found at {}".format(param_file)
    params = experiment_utils.Params(param_file)
    print("starting experiment with {}".format(params.dict))

    num_epochs = params.num_epochs
    torch.manual_seed(params.seed)
    patience = params.patience

    shutil.rmtree(os.path.join(args.experiment_dir, "runs"), ignore_errors=True)

    writer = SummaryWriter(log_dir=os.path.join(args.experiment_dir, "runs"))
    models_dir = os.path.join(args.experiment_dir, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    metrics_dir = os.path.join(args.experiment_dir, "metrics")
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    # initialize datasets
    data_file = os.path.join(script_dir, "data", "energy_data.npz")
    train_generator, validation_generator, test_generator = knapsack_utils.create_data_generators(data_file,
                                                                                                  batch_size=params.batch_size)
    # get shapes of data
    num_items = train_generator.dataset.X.shape[1]
    num_features = train_generator.dataset.X.shape[2]

    # initialize kp cplex instance
    cpx = knapsack_utils.make_knapsack_problem(num_items=num_items, budget=5)
    cpx.cleanup(epsilon=0.0001)

    # initialize network
    net = knapsack_utils.Net(train_generator.dataset.X, train_generator.dataset.y,
                             hidden_layer_sizes=params.layer_sizes,
                             dropout=params.dropout)

    optimizer = torch.optim.Adam(net.parameters(), lr=params.learning_rate, weight_decay=0.1)

    loss_fn = torch.nn.MSELoss()

    best_validation_loss = np.inf
    iters_unchanged = 0
    best_epoch = -1

    all_train_mip_loss = []
    all_validation_mip_loss = []

    total_time_running_eval = 0

    for epoch in range(num_epochs):

        # perform one epoch of training
        net.train(True)
        for local_X, local_y in train_generator:
            optimizer.zero_grad()
            c_true = local_y
            pred_coefs = net(local_X)
            loss = loss_fn(c_true, pred_coefs)

            loss.backward()
            optimizer.step()
            # print("gradients:", [sum(abs(i.grad)) for i in net.parameters() if i.grad is not None])

        # from IPython import embed; import sys; embed(); sys.exit(1)

        # evaluate network
        net.eval()

        time_running_eval_start = time.time()

        # from IPython import embed; import sys; embed(); sys.exit(1)
        training_mip_loss, training_corr, training_mse = knapsack_utils.evaluate_net(cpx, net, train_generator.dataset)
        validation_mip_loss, validation_corr, validation_mse = knapsack_utils.evaluate_net(cpx, net, validation_generator.dataset)

        total_time_running_eval += time.time() - time_running_eval_start

        writer.add_scalars("loss/mip_mean", {"train_mip_mean": np.mean(training_mip_loss),
                                             "val_mip_mean": np.mean(validation_mip_loss)},
                           epoch)

        writer.add_scalars("loss/mip_std", {"train_mip_std": np.std(training_mip_loss),
                                             "val_mip_std": np.std(validation_mip_loss)},
                           epoch)

        writer.add_scalars("loss/corr", {"train_corr": training_corr,
                                             "val_corr": validation_corr},
                           epoch)

        writer.add_scalars("loss/mse", {"train_mse": training_mse,
                                             "val_mse": validation_mse},
                           epoch)

        writer.add_figure("figs/train_pred_vs_actual", knapsack_utils.scatter_predictions(net, train_generator.dataset),
                          epoch)

        writer.add_figure("figs/val_pred_vs_actual", knapsack_utils.scatter_predictions(net, validation_generator.dataset),
                          epoch)

        all_train_mip_loss.append(training_mip_loss)
        all_validation_mip_loss.append(validation_mip_loss)

        if validation_mse < best_validation_loss:
            best_validation_loss = validation_mse
            print(f"{epoch} new best model found, {validation_mse}")
            with open(os.path.join(models_dir, f"{params.method}_best_model.pt"), "wb") as f:
                torch.save(net, f)
            best_epoch = epoch
            iters_unchanged = 0
        else:
            iters_unchanged += 1

        with open(os.path.join(metrics_dir, f"{params.method}_train_obj_val.p"), "wb") as f:
           pickle.dump(all_train_mip_loss, f)
        with open(os.path.join(metrics_dir, f"{params.method}_validation_obj_val.p"), "wb") as f:
            pickle.dump(all_validation_mip_loss, f)

        with open(os.path.join(models_dir, f"{params.method}_it_{epoch}.pt"), "wb") as f:
            torch.save(net, f)

        if iters_unchanged >= patience:
            break

    global_end = time.time()
    total_time = global_end - global_start
    time_no_eval = total_time - total_time_running_eval
    with open(os.path.join(args.experiment_dir, "result.txt"), "w") as f:
        print("=====================================================", file=f)
        print("finished", file=f)
        print("exp", args.experiment_dir, file=f)
        print("total time:", total_time, file=f)
        print("time no eval:", time_no_eval, file=f)
        print("num epochs:", epoch, file=f)
        print("best validation:", best_validation_loss, file=f)
        print("best epoch:", best_epoch, file=f)
        print("=====================================================", file=f)
        print("train_corr", training_corr, file=f)
        print("val_corr", validation_corr, file=f)
        print("train_mse", training_mse, file=f)
        print("val_corr", validation_mse, file=f)
        print("train_mip_mean", np.mean(training_mip_loss), file=f)
        print("val_mip_mean", np.mean(validation_mip_loss), file=f)
        print("train_mip_std", np.std(training_mip_loss), file=f)
        print("val_mip_std", np.std(validation_mip_loss), file=f)