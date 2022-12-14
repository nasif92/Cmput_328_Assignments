# -*- coding: utf-8 -*-
"""Assignment-3 Logistic Regression
CCID: nhossain
Name: Nasif
Student ID: 1545143

"""

import random
import itertools
import torch
import torchvision
from torchvision import transforms, datasets
import numpy as np
import timeit
from collections import OrderedDict
from pprint import pformat
from tqdm import tqdm
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_score(acc, min_thres, max_thres):
    if acc <= min_thres:
        base_score = 0.0
    elif acc >= max_thres:
        base_score = 100.0
    else:
        base_score = float(acc - min_thres) / (max_thres - min_thres) \
            * 100
    return base_score


def run(algorithm, dataset_name, filename):
    start = timeit.default_timer()
    predicted_test_labels, gt_labels = algorithm(dataset_name)
    if predicted_test_labels is None or gt_labels is None:
        return (0, 0, 0)
    stop = timeit.default_timer()
    run_time = stop - start

    np.savetxt(filename, np.asarray(predicted_test_labels))

    correct = 0
    total = 0
    for label, prediction in zip(gt_labels, predicted_test_labels):
        total += label.size(0)
        correct += (prediction.cpu().numpy() == label.cpu().numpy()
                    ).sum().item()   # assuming your model runs on GPU

    accuracy = float(correct) / total

    print('Accuracy of the network on the 10000 test images: %d %%' %
          (100 * correct / total))
    return (correct, accuracy, run_time)


"""
TODO: Finish and submit your code for logistic regression and hyperparameter search.

"""
# some global variables to use throughout the code
learning_rate = 1e-3
lambda_val_Adam = 0.0028
lambda_val_SGD = 0.05
momentum = 0.9
optimizer_val = "ADAM"
epochs = 15


def logistic_regression(dataset_name):

    batch_size_train = 128
    # TODO: implement logistic regression hyper-parameter tuning here
    # 1 load data from dataset

    # for CIFAR
    if dataset_name == 'CIFAR10':
        dims = 3*32*32

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        CIFAR10_training = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                        download=True, transform=transform)

        CIFAR10_training_set, CIFAR10_validation_set = random_split(
            CIFAR10_training, [38000, 12000])

        # CIFAR-10 test set
        CIFAR10_test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                        download=True, transform=transform)

        # # Create data loaders
        trainloader = torch.utils.data.DataLoader(CIFAR10_training_set,
                                                  batch_size=batch_size_train,
                                                  shuffle=True, num_workers=2)

        validation_loader = torch.utils.data.DataLoader(CIFAR10_validation_set,
                                                        batch_size=batch_size_train,
                                                        shuffle=True, num_workers=2)

        testloader = torch.utils.data.DataLoader(CIFAR10_test_set,
                                                 batch_size=1000,
                                                 shuffle=False, num_workers=2)
        logistic_model = LogisticRegression(dims).to(device)
        if optimizer_val == "ADAM":
            optimizer = optim.Adam(
                logistic_model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.SGD(
                logistic_model.parameters(), lr=learning_rate)

    # for MNIST
    else:
        dims = 28*28
        MNIST_training = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                                    transform=torchvision.transforms.Compose([
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

        MNIST_test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                                    transform=torchvision.transforms.Compose([
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

        # create a training and a validation set
        MNIST_training_set, MNIST_validation_set = random_split(
            MNIST_training, [48000, 12000])

        trainloader = torch.utils.data.DataLoader(
            MNIST_training_set, batch_size=128, shuffle=True, num_workers=2)
        validation_loader = torch.utils.data.DataLoader(
            MNIST_validation_set, batch_size=128, shuffle=True)
        testloader = torch.utils.data.DataLoader(
            MNIST_test_set, batch_size=1000, shuffle=False, num_workers=2)

        logistic_model = LogisticRegression(dims).to(device)

        if optimizer_val == "ADAM":
            optimizer = optim.Adam(logistic_model.parameters(
            ), lr=learning_rate, weight_decay=lambda_val_Adam)
        else:
            optimizer = optim.SGD(logistic_model.parameters(
            ), lr=learning_rate, momentum=momentum, weight_decay=lambda_val_SGD)

    # train
    train(logistic_model, trainloader, optimizer, validation_loader)

    # test and output
    return test(logistic_model, testloader)


"""
Logistic Regression Helper classes and functions
"""
# Logistic regression


class LogisticRegression(nn.Module):
    def __init__(self, dims):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(dims, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Following code appears at:  https://lirnli.wordpress.com/2017/09/03/one-hot-encoding-in-pytorch/


class One_Hot(nn.Module):
    def __init__(self, depth):
        super(One_Hot, self).__init__()
        self.depth = depth
        self.ones = torch.sparse.torch.eye(depth).to(device)

    def forward(self, X_in):
        X_in = X_in.long()
        return self.ones.index_select(0, X_in.data)

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)


one_hot = One_Hot(10).to(device)


def train(model, trainloader, optimizer, validation_loader):
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(trainloader):
            data = data.to(device)
            data = data.view(data.size(0), -1)
            target = target.to(device)
            optimizer.zero_grad()
            one_hot = One_Hot(10).to(device)

            loss = torch.nn.functional.cross_entropy(
                model(data), one_hot(target))  # with regularizer on optimizer
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))

        # epoch 5 to validate
        if epoch % 5 == 0:
            validation(model, validation_loader)


def validation(model, validation_loader):
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():  # notice the use of no_grad
        for data, target in validation_loader:
            data = data.to(device)
            data = data.view(data.size(0), -1)
            target = target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            validation_loss += F.cross_entropy(output,
                                               one_hot(target), size_average=False).item()
    validation_loss /= len(validation_loader.dataset)
    print('\nValidation set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(validation_loader.dataset), 100. * correct / len(validation_loader.dataset)))


def test(model, test_loader):
    model.eval()
    y_preds = []
    labels = []
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            data = data.view(data.size(0), -1)
            target = target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output,
                                         one_hot(target), size_average=False).item()
            pred = torch.max(output.data, 1)[1]
            y_preds.append(pred.cpu().numpy())
            labels.append(target.cpu().numpy())
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    return torch.tensor(y_preds), torch.tensor(labels)


'''
TODO: Hyperparameter search.
'''


def tune_hyper_parameter(optimizer_val="ADAM"):
    # TODO: implement logistic regression hyper-parameter tuning here
    best_acc = 0.0
    filenames = {"CIFAR10": "tuning_predictions_cifar10_YourName_IDNumber.txt"}

    parameters = GetParameters()

    best_params = None
    dataset_name = "CIFAR10"
    global lambda_val_Adam, lambda_val_SGD, learning_rate, momentum

    # for both optimizers
    start = timeit.default_timer()

    # Using 4 combinations from the tuning for best parameters
    params_to_tune = random.sample(parameters.get_parameters(optimizer_val), 4)
    for params in params_to_tune:
        if optimizer_val == "ADAM":
            lambda_val_Adam, learning_rate = params[1], params[0]
        else:
            lambda_val_SGD, momentum, learning_rate = params[2], params[1], params[0]

        result, score = run_on_dataset(dataset_name, filenames[dataset_name])
        if result["accuracy"] > best_acc:
            best_acc = result["accuracy"]
            best_params = params
    stop = timeit.default_timer()
    run_time = stop - start

    return best_params, best_acc, run_time


"""
Helper classes and functions for hyperparameter search
"""


class GetParameters():
    def __init__(self):
        self.lr = [0.001, 0.01]
        self.momentum = [0.4, 0.9]
        self.lambda_val = [0.003, 0.002, 0.01]

    def get_parameters(self, optim):
        if optim == "ADAM":
            # return all possible combinations for ADAM
            return list(itertools.product(self.lr, self.lambda_val))

        else:
            # return all possible combinations for SGD
            return list(itertools.product(self.lr, self.momentum, self.lambda_val))


"""Main loop. Run time and total score will be shown below."""


def run_on_dataset(dataset_name, filename):
    if dataset_name == "MNIST":
        min_thres = 0.82
        max_thres = 0.92

    elif dataset_name == "CIFAR10":
        min_thres = 0.28
        max_thres = 0.38

    correct_predict, accuracy, run_time = run(
        logistic_regression, dataset_name, filename)

    score = compute_score(accuracy, min_thres, max_thres)
    result = OrderedDict(correct_predict=correct_predict,
                         accuracy=accuracy, score=score,
                         run_time=run_time)
    return result, score


"""
TODO: TUNE HYPERPARAMETER SEARCH
"""
# Tune Hyperparameter here
# test hyperparameters here


def tune_parameters():
    epochs = 10
    paramsADAM, best_accADAM, run_timeADAM = tune_hyper_parameter()

    paramsSGD, best_accSGD, run_timeSGD = tune_hyper_parameter(
        optimizer_val="SGD")

    print("Best Parameters for ADAM: ", "Learning rate:",
          paramsADAM[0], "Lambda Value(Regularizer):", paramsADAM[1], "Best Accuracy:", best_accADAM, "Total time:", run_timeADAM)

    print("Best Parameters for SGD: ", "Learning rate:", paramsSGD[0],
          "Momentum:", paramsSGD[1], "Lambda Value(Regularizer):", paramsSGD[2], "Best Accuracy:", best_accSGD, "Total time:", run_timeSGD)


def main():
    filenames = {"MNIST": "predictions_mnist_YourName_IDNumber.txt",
                 "CIFAR10": "predictions_cifar10_YourName_IDNumber.txt"}
    result_all = OrderedDict()
    score_weights = [0.5, 0.5]
    scores = []
    for dataset_name in ["MNIST", "CIFAR10"]:
        result_all[dataset_name], this_score = run_on_dataset(
            dataset_name, filenames[dataset_name])
        scores.append(this_score)
    total_score = [score * weight for score,
                   weight in zip(scores, score_weights)]
    total_score = np.asarray(total_score).sum().item()
    result_all['total_score'] = total_score
    with open('result.txt', 'w') as f:
        f.writelines(pformat(result_all, indent=4))
    print("\nResult:\n", pformat(result_all, indent=4))

    # TODO: For tuning hyperparameters uncomment this line of code.
    tune_parameters()


main()
