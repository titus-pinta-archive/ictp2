#! /usr/python-pytorch/bin/python
import torch
import torch.nn.functional as F

import argparse
import optim

import datetime
import dill
import sys

import models
import dataloader
from test import *
from train import *

import view

def main():
    #Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 0.00001)')
    parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                        help='SGD momentum (default: 0.0)')
    parser.add_argument('--q', type=float, default=2, metavar='Q',
                        help='q parameter for A5 algorithm (default: 2)')
    parser.add_argument('--k', type=float, default=5, metavar="K",
                        help='k parameter for A3/A5 algorithm (default: 5)')
    parser.add_argument('--optim', default='SGD', help='Optimiser to use (default: SGD)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--stoch', action='store_true', default=False,
                        help='use stochastic gradient computation')
    parser.add_argument('--cifar10', action='store_true', default=False, help='Use Cifar10 not MNIST')
    parser.add_argument('--fash', action='store_true', default=False, help='Use MNIST fashion not MNIST')
    parser.add_argument('--optimhelp', default=False, action='store_true', help='Print optim options')

    args = parser.parse_args()

    if args.optimhelp:
        optims = dir(optim)
        for option in optims:
            if '_' not in option:
                print(option)
        exit()
    print('Gradient is computed {}'.format('stochastically' if args.stoch else 'non stochastically'))
    print('Will train for {} epochs with a batch size of {}'.format(args.epochs, args.batch_size))
    if(args.stoch):
        train = train_stoch
    else:
        train = train_non_stoch
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print('Computing on {}'.format('cuda' if use_cuda else 'cpu'))
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader, test_loader = dataloader.dataloader(args.fash, args.cifar10,
                                                      args.batch_size, args.test_batch_size,
                                                     kwargs)


    #create model
    if args.cifar10:
        model = models.cifar10.GoogLeNet()
        model.load_state_dict(torch.load('cifar10.init'))
        print('Using GoogLeNet model')
    else:
        model = models.mnist.Net()
        model.load_state_dict(torch.load('mnist.init'))
        print('Using MNIST model')
    print("Initial model: {}".format(str(model)))
    train_correct = []
    test_correct = []
    train_loss = []
    test_loss = []


    model.to(device)
    try:
        extra_params = {}
        if args.optim == 'SGD' and args.momentum != 0.0:
            extra_params = {'momentum': args.momentum, 'nesterov': True}
        if args.optim == 'A3':
            extra_params = {'k': args.k}
        if args.optim == 'A5':
            extra_params = {'k': args.k, 'q': args.q}

        optim_class = getattr(optim, args.optim)
        optimizer = optim_class(model.parameters(), lr=args.lr, **extra_params)
        print(optimizer)
    except Exception as e:
        print(e)
        raise ValueError('Undefined Optimiser: {}'.format(args.optim))

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, train_correct, train_loss)
        test(args, model, device, test_loader, test_correct, test_loss)


    save_result = {'optim': optimizer, 'model': model.state_dict, 'args': args,
                   'train_loss': train_loss, 'test_loss': test_loss,
                   'train_correct': train_correct, 'test_correct': test_correct}


    save_dir = 'results/{}'.format(str(datetime.datetime.now()).replace(' ', '-')
                                    .replace('.', '-').replace(':', '-'))
    view.gfx(save_result, save_name)

    save_option = input('Save data? (y)es/(n)o ')

    if save_option == 'y':
        os.mkdirs(save_dir)
        save_name = save_dir + '/data.result'
        with open(save_name, 'wb') as f:
            dill.dump(save_result, f)


if __name__ == '__main__':
    main()
