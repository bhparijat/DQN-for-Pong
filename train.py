import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt

from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler

from nets.cnn import CNN
from utils.runs import load_runs


parser = argparse.ArgumentParser(description="Run commands",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--save_dir', type=str, default="data", help="Directory with uct data")
parser.add_argument('--in_model', type=str, default=None, help="Saved model filename")
parser.add_argument('--out_model', type=str, default='models/model_%s' % datetime.now().strftime("%Y%m%d-%H%M%S"),
                    help="Filename for trained model")
parser.add_argument('--save_period', type=int, default=10, help="Interval between checkpoints")
parser.add_argument('--network', type=str, default='cnn', help="Network architecture")
parser.add_argument('--n_frames', type=int, default=2, help="Number of frames to stack")
parser.add_argument('--width', type=int, default=84, help="Width of frame")
parser.add_argument('--height', type=int, default=84, help="Height of frame")
parser.add_argument('--downsample', type=float, default=None, help="Factor of downsampling image")
parser.add_argument('--loss', type=str, default='cross_entropy', help="Type of loss function: [cross_entropy]")
parser.add_argument('--optim', type=str, default='adam', help="Type of advancing learning function: [adam]")
parser.add_argument('--batch', type=int, default=32, help="Number of samples per batch")
parser.add_argument('--samples_per_epoch', type=int, default=1000, help="Number of samples per epoch")
parser.add_argument('--epochs', type=int, default=100, help="Number of epochs")
parser.add_argument('--augment', action='store_true', help="Augment images")
parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
parser.add_argument('--momentum', type=float, default=0.9, help="Momentum")
parser.add_argument('--weight_runs', action='store_true', help="Weight runs according to reward obtained in run")
parser.add_argument('--norm', action='store_true', help="Do value normalization per state or not")
parser.add_argument('--norm_coeff', type=float, default=1, help="Normalization coefficient")
parser.add_argument('--entropy', type=float, default=0.001, help="Entropy coefficient for policy loss")
parser.add_argument('--flip', action='store_true', help="Flip image and action vertically")
parser.add_argument('--color', action='store_true', help="Process color images instead of grayscale")
parser.add_argument('--min_run_score', type=float, default=None, help="Minimum score in run to process run")
parser.add_argument('--generator_workers', type=int, default=1, help="Number of workers to generate data.")


def eval_net(dataloader, net):
    correct = 0
    total = 0
    total_loss = 0
    net.eval()  # Why would I do this?
    criterion = nn.CrossEntropyLoss(reduction='sum')
    for data in dataloader:
        inputs, labels = data
        inputs, labels = Variable(inputs.float()).cuda(), Variable(labels).cuda()
        # inputs, labels = Variable(inputs.float()), Variable(labels)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum().item()
        loss = criterion(outputs, labels)
        total_loss += loss.item()
    print("number of correct for this epoch is", correct)
    return total_loss / total, correct / total


def train():
    # TODO have to match the inputs and label format
    train_accuracy_array = []
    test_accuracy_array = []
    print('Start training...')

    print('Start training...')
    for epoch in range(MAX_EPOCH):  # loop over the dataset multiple times

        running_loss = 0.0
        # enumeration is sufficient because of which we initialize the batch_size when creating the loader
        # for i, data in enumerate(trainloader, 0): removed 0 because by default it starts from 0
        # also here enumerate returns the index of the starting data that is being returned.
        for i, data in enumerate(trainloader):
            # get the inputs
            inputs, labels = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            # inputs, labels = Variable(inputs.float()), Variable(labels)

            net.train()  # Why would I do this? Sets the module in training mode to allow training in the next batch

            optimizer.zero_grad()
            outputs = net(inputs)  # forward
            loss = criterion(outputs, labels)  # loss
            loss.backward()  # accumulated gradient
            optimizer.step()  # performs the parameter update based on the current gradient in the previous line
            running_loss += loss.item()
            if i % 500 == 499:
                print('Step: %5d avg_batch_loss: %.5f' % (i + 1, running_loss / 500))
                running_loss = 0.0
        print('Finish training this EPOCH, start evaluating...')
        train_loss, train_acc = eval_net(trainloader, net)
        test_loss, test_acc = eval_net(testloader, net)
        print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
              (epoch+1, train_loss, train_acc, test_loss, test_acc))

        train_accuracy_array.append(train_acc)
        test_accuracy_array.append(test_acc)

    # plot accuracy
    plt.clf()
    plt.plot(list(range(1, MAX_EPOCH + 1)), train_accuracy_array, label='Train')
    plt.plot(list(range(1, MAX_EPOCH + 1)), test_accuracy_array, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs Epochs [%s]' % net.name)
    plt.savefig('./%s/accuracy-%s.png' % (folders['plots'], net.name))

    print('Finished Training')


if __name__ == "__main__":
    args = parser.parse_args()

    # constant values
    folders = {
        "plots": "plots",
        "models": "models"
    }

    # generate required folders
    for key in folders.keys():
        try:
            os.makedirs(folders[key])
        except FileExistsError:
            # if file exists, pass
            pass

    validation_split = 0.1
    random_seed = 0

    BATCH_SIZE = 10  # mini_batch size
    MAX_EPOCH = 10  # maximum epoch to train
    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # torchvision.transforms.Normalize(mean, std)

    dataset = load_runs(args.save_dir)
    indices = list(range(len(dataset)))
    split = int(np.floor(validation_split * len(dataset)))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    test_indices, train_indices = indices[split:], indices[:split]

    trainset = [dataset[train_index] for train_index in train_indices]  # dataset[train_indices]
    testset = [dataset[test_index] for test_index in test_indices]  # dataset[test_indices]

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    if train_indices is not None and len(train_indices) > 0:
        if args.network == "cnn":
            net = CNN("cnn").cuda()
            # net = CNN("cnn")
        else:
            net = CNN("cnn").cuda()
            # net = CNN("cnn")
        if args.optim == "adam":
            optimizer = optim.Adam(net.parameters(), lr=args.lr)
        else:
            optimizer = optim.Adam(net.parameters(), lr=args.lr)
        if args.loss == "cross_entropy":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        train()
        torch.save(net.state_dict(), './%s/training-%s.pth' % (folders['models'], net.name))
