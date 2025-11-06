from re import X
import sys

sys.path.append("../python")
from needle.data import MNISTDataset, DataLoader
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    seq = nn.Sequential(nn.Linear(dim, hidden_dim), norm(hidden_dim), nn.ReLU(), nn.Dropout(drop_prob), nn.Linear(hidden_dim, dim), norm(dim))
    res = nn.Residual(seq)
    return nn.Sequential(res, nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), *[ ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob) for _ in range(num_blocks) ], nn.Linear(hidden_dim, num_classes))
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is None:
      model.eval()
    else:
      model.train()
      opt.reset_grad()

    total_loss, total_error = [], 0.0
    loss_fn = nn.SoftmaxLoss()
    for X, y in dataloader:
      logits = model(X)
      total_error += np.sum(logits.numpy().argmax(axis=1) != y.numpy())
      loss = loss_fn(logits, y)
      total_loss.append(loss.numpy())

      if opt is not None:
        loss.backward()
        opt.step()

    return total_error / len(dataloader.dataset), np.mean(total_loss)
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = MNISTDataset(f"{data_dir}/train-images-idx3-ubyte.gz", f"{data_dir}/train-labels-idx1-ubyte.gz")
    test_dataset = MNISTDataset(f"{data_dir}/t10k-images-idx3-ubyte.gz", f"{data_dir}/t10k-labels-idx1-ubyte.gz")
    train_dataloader = DataLoader(train_dataset, batch_size, True)
    test_dataloader = DataLoader(test_dataset, batch_size)

    resnet = MLPResNet(28*28, hidden_dim, num_classes=10)
    opt = optimizer(resnet.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
      train_loss, train_error = epoch(train_dataloader, resnet, opt)

    test_loss, test_error = epoch(test_dataloader, resnet)
    print( (train_loss, train_error, test_loss, test_error) )
    return (train_loss, train_error, test_loss, test_error)
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(250, 5, ndl.optim.SGD, 0.01, 0.001, 100, data_dir="../data")
