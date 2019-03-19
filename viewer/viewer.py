import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import main


def help():
    print('\nthe following functions and variables are available:\n')
    print('use_cuda:')
    print('\tboolean variable default is True if cuda is available')
    print('\tcan be modified directly')
    print('\nexample modification viewer.use_cuda=False')
    print('load_model(path)')
    print('\treturns the model stored in the specified path')
    print('\tplease provide the extension (normally .model or .model.part)')
    print('\nexample call: viewer.load_model(\'SGD.model\')\n')
    print('load result(path)')
    print('\treturns the result vector stored in the specified path')
    print('\tplease also provide the extension')
    print('\tthe result is a tupel with the first component the number of testing images')
    print('\t\tand the second component a tuple with the params (lr),\n\t\tthrd component a vector'+
          ' with the number of correct predictions\n\t\tand the forth a vector with the loss for each' +
          ' epoch')
    print('\nexample call: viewer.load_result(\'SGD.result\')\n')
    print('load_mnist(batch_size=64, test_batch_size=1000, use_cuda=True, seed=1)')
    print('\tthe meaning of the parameters is the same as for the torch dataloader')
    print('\nexample call: viewer.load_mnist()\n')
    print('plot_result(result, plot=None, loss=False, correct=True, fraction=True, show=True)')
    print('\tplots the result info and returns the matplotlib object')
    print('\t\tplot=a previously constructed plot with which to combine the current plot')
    print('\t\tloss=True if the plot should contain the loss, False for the number of correct predictions')
    print('\t\tcorrect=True for num of correct prediction false for number of wrong predictions')
    print('\t\tfraction=True divides num correct by num total')
use_cuda = torch.cuda.is_available()


def load_model(path):
    device = torch.device('cuda' if use_cuda else 'cpu')
    net = main.Net()
    net.load_state_dict(torch.load(path))
    net.to(device)
    return net


def load_result(path):
    with open(path, 'rb') as f:
        res = dill.load(f)
    return res

def load_mnist(batch_size=64, test_batch_size=1000, use_cuda=True, seed=1):

    use_cuda = use_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader

def plot_result(result, plot=None, loss=False, correct=True, fraction=True, show=True):

    global plt
    if plot is not None:
        plt = plot

    if loss:
        data = np.array(result[3])
        ymax = np.max(data) + 0.1

    else:
        if correct:
            data = np.array(result[2])
        else:
            data = result[0] - np.array(result[2])

        ymax = result[0]

        if fraction:
            data = data / result[0]
            ymax = 1

    epochs = np.linspace(0, len(result[2]), len(result[2]))

    plt.ylim(0, ymax)
    print(data)
    plt.plot(epochs, data)
    if show:
        plt.show()

    return plt


def model_statistics(model, loader):

    device = torch.device('cuda' if use_cuda else 'cpu')
    df = np.zeros((10))
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            for i in pred[pred.eq(target.view_as(pred))]:
                df[int(i)] += 1
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))

    return df

help()
