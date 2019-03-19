import torch
import torch.nn.functional as F


def train_stoch(args, model, device, train_loader, optimizer, epoch, result_correct, result_loss):

    model.train()
    train_loss = 0;
    train_correct = 0;

    for batch_idx, (data, target) in enumerate(train_loader):
        loss = None

        def closure():
            nonlocal data
            nonlocal target
            nonlocal loss
            nonlocal train_loss
            nonlocal train_correct

            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct  += pred.eq(target.view_as(pred)).sum().item()


        optimizer.step(closure)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    result_loss.append(train_loss)
    result_correct.append(train_correct)

def train_non_stoch(args, model, device, train_loader, optimizer, epoch, result_correct, result_loss):
    closure_calls = 0
    train_loss = 0
    train_correct = 0

    def closure():
        nonlocal closure_calls
        nonlocal train_loss
        nonlocal train_correct
        closure_calls += 1
        print('\nNumber of closure calls: {}\n'.format(closure_calls))
        optimizer.zero_grad()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct  += pred.eq(target.view_as(pred)).sum().item()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    model.train()
    optimizer.step(closure)
    result_loss.append(train_loss)
    result_correct.append(train_correct)
