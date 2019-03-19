import matplotlib.pyplot as plt
import numpy as np

def gfx(save_result, save_name):
    args = save_result['args']
    n = args.epochs
    xaxis = np.linspace(1, n, n)

    if (args.fash and args.cifar10):
        dataset = 'MNIST'
    elif args.fash:
        dataset = 'MNISTFashion'
    else:
        dataset = 'Cifar10'

    plt.plot(xaxis, save_result['train_correct'], 'r', xaxis, save_result['test_correct'], 'b')
    plt.legend(['Train Correct', 'Test Correct'])
    plt.title(dataset + ' ' + save_result['args'].optim)
    plt.show()
    plt.plot(xaxis, save_result['train_loss'], 'r', xaxis, save_result['test_loss'], 'b')
    plt.legend(['Train Loss', 'Test Loss'])
    plt.title(dataset + ' ' + save_result['args'].optim)
    plt.show()
