import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torchvision import transforms


class RNN(nn.Module):
    def __init__(self, name):
        super(RNN, self).__init__()
        self.name = name
        # TODO modify the layers
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=0)

        # self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(11 * 11 * 32, 256)
        self.fc2 = nn.Linear(256, 6)
        # self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # TODO modify the layers
        x = F.tanh(self.conv1(x))
        #print("after first",x.shape)
        x = F.tanh(self.conv2(x))
        #print("after second",x.shape)
        # x = self.pool(x)
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        #x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        #print("after 3rd",x.shape)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def eval_net(dataloader):
    correct = 0
    total = 0
    total_loss = 0
    net.eval()  # Why would I do this?
    criterion = nn.CrossEntropyLoss(size_average=False)
    for data in dataloader:
        images, labels = data
        images, labels = Variable(images.float()).cuda(), Variable(labels).cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        loss = criterion(outputs, labels)
        total_loss += loss.item()
    net.train()  # Why would I do this?
    print("number of correct for this epoch is", correct.item())
    return total_loss / total, correct.item() / total


if __name__ == "__main__":

    files = os.listdir('../data')

    count = 0
    pos_rewards = 0
    train_set = []
    lf = None
    print("length of files is", len(files))
    for i, file in enumerate(files):
        path = "../data/" + str(file)

        with open(path, 'rb') as f:
            f.seek(0)
            # print(i)
            try:
                data = pickle.load(f)
            except pickle.UnpicklingError:
                continue
            # print(data.keys())
            count = count + len(data['frames'])
            frames = data['frames']
            actions = np.array(data['actions'])
            # actions = np.newaxis(actions)
            for i, frame in enumerate(frames):
                frame = np.swapaxes(frame, 0, 2)
                train_set.append((frame, actions[i]))
                lf = frame
            if data['reward'] > 0:
                pos_rewards = pos_rewards + 1
                # print(data['reward'])

    print(len(train_set))
    # print(type(frame))
    # train_CNN(train_set)
    BATCH_SIZE = 10  # mini_batch size
    MAX_EPOCH = 10  # maximum epoch to train
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # torchvision.transforms.Normalize(mean, std)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=1)

    classes = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']

    print('Building model...')
    # net = Net().cuda()
    net = RNN("cnn").cuda()
    net.train()  # Why would I do this?

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    train_accuracy_array = []
    print('Start training...')
    for epoch in range(MAX_EPOCH):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            inputs, labels = data

            # print(type(inputs),type(labels),type(inputs[i]))
            # print(labels.shape,inputs.shape)
            # print(labels)
            inputs = inputs.float()
            inputs = Variable(inputs)
            # labels = labels.long()
            labels = Variable(labels)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:  # print every 2000 mini-batches
                print('    Step: %5d avg_batch_loss: %.5f' %
                      (i + 1, running_loss / 500))
                running_loss = 0.0
        print('    Finish training this EPOCH, start evaluating...')
        train_loss, train_acc = eval_net(trainloader)
        train_accuracy_array.append(train_acc)
        # all_info['train']['acc_y'].append(train_acc)
        # all_info['train']['loss_y'].append(train_loss)

    # plot accuracy
    plt.clf()
    plt.plot(list(range(1, MAX_EPOCH + 1)), train_accuracy_array)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.savefig('./%s/accuracy-%s.png' % ('plots', datetime.now().strftime('%m%d%Y-%H%M%S')))

    print('Finished Training')
    print('Saving model...')

    torch.save(net.state_dict(), 'myatari.pth')

    # print(count,pos_rewards)

