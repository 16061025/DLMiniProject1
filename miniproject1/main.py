import torch
import sys
import pickle


import dataprocessor
import ResNetmodel
import netconfig
import torchsummary

def train(dataloader, model, loss_fn, optimizer):

    size = len(dataloader.dataset)

    model.train()
    for batch, (x, y) in enumerate(dataloader):

        x = x.to(netconfig.device)
        y = y.to(netconfig.device)

        # forward
        y_predict = model(x)
        loss = loss_fn(y_predict, y)

        # bp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:

            x, y = x.to(netconfig.device), y.to(netconfig.device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size

    global  max_accuracy, total_accuracy, t
    if (100 * correct) > max_accuracy:
        max_accuracy = (100 * correct)
    if t>=10:
        total_accuracy += (100 * correct)
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if "cuda" in sys.argv:
    netconfig.device = 'cuda'

dp = dataprocessor.DataProcessor()

if "aug" in sys.argv:
    dp = dataprocessor.DataProcessor(aug=True)
else:
    dp = dataprocessor.DataProcessor(aug=False)

train_dataloader, test_dataloader = dp.train_dataloader, dp.test_dataloader

model = ResNetmodel.ResNet(ResNetmodel.BasicBlock).to(netconfig.device)
loss_fn = torch.nn.CrossEntropyLoss()

if "L2" in sys.argv:
    optimizer = torch.optim.Adam(model.parameters(), lr=netconfig.lr, weight_decay=1e-4)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=netconfig.lr)

print("number of parameters", count_parameters(model))

epochs = 30

max_accuracy = -1
total_accuracy = 0

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Done!")
print("max accuracy", max_accuracy)
print("average accuracy", total_accuracy/(epochs-10))








