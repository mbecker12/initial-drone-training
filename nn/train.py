import os, sys

sys.path.append(os.getcwd())
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

# from controller.parameters import N_THRUSTERS
N_THRUSTERS = 4
from create_dataset.create_pid_output import generate_ground_truth
import matplotlib.pyplot as plt
from nn.drone_network import DroneBrain
from copy import deepcopy

# loss = nn.MSELoss()
# l = loss(input, target)


def train_batch(model, optimizer, loss_fn, batch_size, n_inputs=12):

    # Compute the output for all samples in the batch and the average loss
    # 3 extra inputs for separate lab_pos and coin_position, which in
    # the nn are taken as the distance between coin and drone
    # (specifically coin_position - drone_position)
    X_numpy = np.empty((batch_size, n_inputs), dtype=np.float64)
    Y_numpy = np.empty((batch_size, N_THRUSTERS), dtype=np.float64)

    for i in range(batch_size):
        x, y = generate_ground_truth()

        # distance: coin_position - drone_position
        distance = np.array([x[i] - x[i + 3] for i in range(3)])

        X_numpy[i] = np.concatenate((distance, x[6:])).reshape(-1)
        Y_numpy[i] = y.reshape(-1)

    X = torch.tensor(X_numpy, dtype=torch.float32)
    Y = torch.tensor(Y_numpy, dtype=torch.float32)

    pred = model(X)

    # print(f"pred: {pred}, Y: {Y}")

    # TODO: look up how loss and loss_fn works
    # on which object do I call backward?
    loss = loss_fn(pred, Y)
    # print(f"loss: {loss}")
    # Backward-propagation
    loss.backward()

    # Perform one step of optimization
    optimizer.step()

    # Zero gradients before computing backward-propagation
    optimizer.zero_grad()

    return pred, loss


def val_batch(model, loss_fn, batch_size, n_inputs=12):
    with torch.no_grad():
        X_numpy = np.empty((batch_size, n_inputs), dtype=np.float64)
        Y_numpy = np.empty((batch_size, N_THRUSTERS), dtype=np.float64)

        for i in range(batch_size):
            x, y = generate_ground_truth()

            # distance: coin_position - drone_position
            distance = np.array([x[i] - x[i + 3] for i in range(3)])

            X_numpy[i] = np.concatenate((distance, x[6:])).reshape(-1)
            Y_numpy[i] = y.reshape(-1)

        X = torch.tensor(X_numpy, dtype=torch.float32)
        Y = torch.tensor(Y_numpy, dtype=torch.float32)
        pred = model(X)

        loss = loss_fn(pred, Y)

    return pred, loss


def test_batch(model, loss_fn, batch_size, n_inputs=12):
    with torch.no_grad():
        X_numpy = np.empty((batch_size, n_inputs), dtype=np.float64)
        Y_numpy = np.empty((batch_size, N_THRUSTERS), dtype=np.float64)

        for i in range(batch_size):
            x, y = generate_ground_truth()

            # distance: coin_position - drone_position
            distance = np.array([x[i] - x[i + 3] for i in range(3)])

            X_numpy[i] = np.concatenate((distance, x[6:])).reshape(-1)
            Y_numpy[i] = y.reshape(-1)

        X = torch.tensor(X_numpy, dtype=torch.float32)
        Y = torch.tensor(Y_numpy, dtype=torch.float32)
        pred = model(X)

        loss = loss_fn(pred, Y)

    return pred, Y, loss


def eval_prediction(pred, Y, loss, max_prints=1):
    batch_size = Y.shape[0]

    for i in range(batch_size):
        if i >= max_prints:
            break

        outstring = f"Prediction:  {pred[i]}".ljust(70)
        outstring += f"GroundTruth: {Y[i]}".rjust(70)
        outstring += f"\nLoss: {loss}".ljust(50)
        print(outstring)


# TODO: could compute the cosine distance between ground truth and predicted
# thrust vector as scoring metric

# def compute_metrics_on_validation_set(rnn, val_dataset):
#     # Get all the input and labels in the training set.
#     x_val, y_val = val_dataset[:]

#     # Perform forward-prop in the entire validation set, with autograd disabled
#     with torch.no_grad():
#         val_output, val_loss = batch_forward_prop(rnn, x_val, y_val)

#     # Get numpy arrays for the true labels and the predictions
#     y_true = y_val.cpu().numpy()
#     y_pred = val_output.argmax(dim=1).cpu().numpy()

#     return val_loss, precision, recall, fscore


def plot_metrics(fig, ax, ns, train_losses, train_fscores, val_losses, val_fscores):

    # Plot losses
    ax.clear()
    ax.plot(ns, train_losses)
    ax.plot(ns, val_losses)
    ax.set_title("Loss")
    ax.legend(["Train", "Validation"])
    ax.set_xlabel("Number of trained batches")
    ax.grid()

    # # Plot losses
    # ax[0].clear()
    # ax[0].plot(ns, train_losses)
    # ax[0].plot(ns, val_losses)
    # ax[0].set_title('Loss')
    # ax[0].legend(['Train','Validation'])
    # ax[0].set_xlabel('Number of trained batches')
    # ax[0].grid()

    # # Plot F1-scores
    # ax[1].clear()
    # ax[1].plot(ns, train_fscores)
    # ax[1].plot(ns, val_fscores)
    # ax[1].plot(ns, [0.3]*len(ns), 'k--')
    # ax[1].set_title('Macro F1-score')
    # ax[1].legend(['Train','Validation', 'F1-score threshold'])
    # ax[1].set_xlabel('Number of trained batches')
    # ax[1].grid()

    fig.canvas.draw()


def train(model, n_epochs, n_batches_train, n_batches_val, learning_rate, batch_size):

    fig, ax = plt.subplots(ncols=1, figsize=(12, 4))
    plt.ion()
    plot_interval = 100

    # Create arrays to average training metrics across batches
    preds = []
    labels = []
    losses = []

    best_model = None
    best_val_loss = np.Infinity
    # Create dictionaries to hold the computed metrics in
    train_data = {"losses": [], "fscores": []}
    val_data = {"losses": [], "fscores": []}

    print("Set up optimizer and Loss function...")
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss()
    # train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    batch_idxs = []

    # Training loop
    print("Start Training")
    i_batch = 0
    for n in range(n_epochs):
        print(f"Epoch {n} / {n_epochs}")
        for i in range(n_batches_train):
            output, loss = train_batch(model, optimizer, loss_fn, batch_size)
            i_batch += 1

            # Aggregate for later averaging
            # preds += output.argmax(dim=1).cpu().tolist()
            # labels += y_batch.cpu().tolist()
            losses.append(loss)

            # Compute metrics and plot after every `plot_interval` batches
            if i % plot_interval == 0:
                val_output, val_loss = val_batch(
                    model, loss_fn, batch_size, n_inputs=12
                )
                # val_loss, _, _, val_fscore = compute_metrics_on_validation_set(rnn, val_dataset)
                # train_fscore = precision_recall_fscore_support(labels, preds, average='macro')[2]

                val_data["losses"].append(val_loss)
                # val_data['fscores'].append(val_fscore)
                train_data["losses"].append(sum(losses) / len(losses))
                # train_data['fscores'].append(train_fscore)
                batch_idxs.append(i_batch)

                print(
                    f"Train Loss: {sum(losses)/(len(losses) + 1):.5f}, Val Loss: {val_loss:.5f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = deepcopy(model)
                    torch.save(best_model.state_dict(), "pretrained_drone_brain_wide_rot_velocity_lin_noflush")

                preds = []
                labels = []
                losses = []

                plot_metrics(
                    fig,
                    ax,
                    batch_idxs,
                    train_data["losses"],
                    None,
                    val_data["losses"],
                    None,
                )

    return best_model


if __name__ == "__main__":
    print("Initialize Drone Brain...")
    model = DroneBrain(
        n_inputs=12,
        n_neurons1=128,
        n_neurons2=256,
        n_neurons3=128,
        n_neurons4=64,
        n_outputs=4,
    )
    n_epochs = 25
    n_batches_train = 300
    n_batches_val = 100
    learning_rate = 0.0005
    batch_size = 32
    # print("Start Training...")
    train(model, n_epochs, n_batches_train, n_batches_val, learning_rate, batch_size)

    print("Training Finished...")
    print("Show model evaluation...")
    loss_fn = nn.L1Loss()
    pred, Y, loss = test_batch(model, loss_fn, 16)
    eval_prediction(pred, Y, loss, max_prints=4)
