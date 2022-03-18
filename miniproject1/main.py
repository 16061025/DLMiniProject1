import torch
import matplotlib.pyplot as plt
import sys
import pickle


import dataprocessor
import ResNetmodel
import netconfig
import torchsummary

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batchs = len(dataloader)
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x = x.to(netconfig.device)
        y = y.to(netconfig.device)
        #print("forward")
        # forward
        y_predict = model(x)
        loss = loss_fn(y_predict, y)
        #print("bp")
        # bp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            # global train_batch_loss, train_batch_cnt, train_batch, axs
            # train_batch.append(train_batch_cnt)
            # train_batch_loss.append(loss)
            # train_batch_cnt += 1
            #
            # axs['up'].plot(train_batch, train_batch_loss, color="blue")
            # plt.pause(0.001)
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

    # global test_avg_batch_loss, test_accurary, test_epoch, test_epoch_cnt, axs
    #
    # test_epoch.append(test_epoch_cnt)
    # test_avg_batch_loss.append(test_loss)
    # test_accurary.append(correct*100)
    #
    # test_epoch_cnt += 1
    #
    # axs['lowleft'].plot(test_epoch, test_avg_batch_loss, color="red")
    # axs['lowright'].plot(test_epoch, test_accurary, color="green")
    #
    # plt.pause(0.001)

    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

dp = dataprocessor.DataProcessor()
train_dataloader, test_dataloader = dp.train_dataloader, dp.test_dataloader

model = ResNetmodel.ResNet(ResNetmodel.BasicBlock).to(netconfig.device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=netconfig.lr)

epochs = 50

# num_batch = int(len(train_dataloader)*epochs/100)
#
# fig, axs = plt.subplot_mosaic([['up','up'],
#                                ['lowleft','lowright']])
#
#
# axs['up'].set_title("train batch loss")
# axs['up'].set_xlim(0, num_batch)
# axs['up'].set_ylim(0, 10)
# axs['up'].set_xlabel("batch")
# axs['up'].set_ylabel("loss")
#
# axs['lowleft'].set_title("test avg batch loss")
# axs['lowleft'].set_xlim(0,epochs)
# axs['lowleft'].set_ylim(0, 10)
# axs['lowleft'].set_xlabel("per 100 epoch")
# axs['lowleft'].set_ylabel("loss")
#
# axs['lowright'].set_title("test accuracy")
# axs['lowright'].set_xlim(0,epochs)
# axs['lowright'].set_ylim(0,100)
# axs['lowright'].set_xlabel("epoch")
# axs['lowright'].set_ylabel("accuracy %")
#
# fig.tight_layout()
#
# train_batch_loss = []
# train_batch = []
#
# test_avg_batch_loss = []
# test_accurary = []
# test_epoch = []
#
# train_batch_cnt = 1
# test_epoch_cnt = 1
#
# plt.ion()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("number of parameters", count_parameters(model))


# torchsummary.summary(model, (3,32,32))

f = open(sys.argv[1], 'rb')
configs = pickle.load(f)
config = configs[int(sys.argv[2])]
netconfig.setconfig(config)
while(True):
    continue
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(train_dataloader, model, loss_fn, optimizer)
#     test(test_dataloader, model, loss_fn)
# print("Done!")

# plt.ioff()
# plt.show()







