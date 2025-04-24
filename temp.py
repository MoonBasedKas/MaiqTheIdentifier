import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import sys
import net
import faceData
import os 

def train_epoch():
    # Begin learning
    net.train(True)
    runningLoss = 0.0
    runningAccuracy = 0.0

    # Iterate over the trained data
    for batch, data  in enumerate(trainloader):
        inputs, labels = data[0].to(device), data[1].to(device)

        # Reset the gradents
        optimizer.zero_grad()
        outputs = net(inputs) # Gets our shape, size, shape(2)
        # The highest value of this will be the most likely, it'll be the index of it.
        # Gets us the labels of the images where it matches
        correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
        runningAccuracy += correct / batch_size

        # This is kind of magic but for back propagation
        loss = criterion(outputs, labels)
        runningLoss += loss.item()
        loss.backward()
        optimizer.step()

        if batch % 2 == 1:
            avgLossAcrossBatches = runningLoss/500
            avgAcrossBatches = (runningAccuracy/500)*100
            print("Batch", batch)
            print("Loss:", avgLossAcrossBatches)
            print("Accuracy:",avgAcrossBatches)
            runningAccuracy = 0
            runningLoss = 0

    print("-")

def validate_epoch():
    # Disable
    net.train(False)
    runningLoss = 0.0
    runningAccuracy = 0.0

    # Iterate over the test data
    for batch, data  in enumerate(testloader):
        inputs, labels = data[0].to(device), data[1].to(device)

        with torch.no_grad():
            outputs = net(inputs)
            correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()

        # The highest value of this will be the most likely, it'll be the index of it.
        # Gets us the labels of the images where it matches
            correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
            runningAccuracy += correct / batch_size

        # This is kind of magic but for back propagation
            loss = criterion(outputs, labels)
            runningLoss += loss.item()


        # 4 is batches
        avgLossAcrossBatches = runningLoss/4
        avgAcrossBatches = (runningAccuracy/4)*100
        print("Batch", batch)
        print("Loss:", avgLossAcrossBatches)
        print("Accuracy:",avgAcrossBatches)
        runningAccuracy = 0
        runningLoss = 0



# Very slow
device = 'cpu'

def main():
    

    pass

transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Resize((28,28)),
    transforms.Resize((128,128)),
    transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
     )

# transform = transforms.ToTensor()

batch_size = 4

# This gets the data set?

trainset = faceData.CustomImageDataset(f"data{os.sep}labels.csv", f"data{os.sep}images", transform=transform)
# This loads it into torch to train?
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
# num_workers=2


# Tests the accuracy
# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)

testset = faceData.CustomImageDataset(f".{os.sep}data{os.sep}train.csv", f".{os.sep}data{os.sep}training", transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

classes = ('Not Steve', 'Steve')


# Colored images have 3 channels
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)
# Gets me my shape fo
print(images.shape)
# tset, vset = torch.utils.random_split(trainset, [5,3]) # Splits the dataset

print(torch.min(images), torch.max(images))

# show images
print(len(trainset))
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# Creates the neural network.


net = net.neuralNet()
# net.to(device) # Enables gpu


criterion = nn.CrossEntropyLoss()
# Erik said this adam guy is like magic, lr: learning rate
optimizer = optim.Adam(net.parameters(), lr=0.001)
for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        print(inputs.shape)
        print(net(inputs).shape)
# Our params
z = 0
for x in net.parameters():
     z += len(torch.flatten(x))
print("Params", z)

# Training loop
epic = 10
for epoch in range(epic):
    print("epoch:", epoch)

    train_epoch()
    validate_epoch()



#         # zero the parameter gradients
#         # optimizer.zero_grad()

#         # forward + backward + optimize
#         # outputs = net(inputs)
#         # loss = criterion(outputs, labels)
#         # loss.backward()
#         # optimizer.step()

#         # # print statistics
#         # running_loss += loss.item()
#         # if i % 2000 == 1999:    # print every 2000 mini-batches
#         #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
#         #     running_loss = 0.0
