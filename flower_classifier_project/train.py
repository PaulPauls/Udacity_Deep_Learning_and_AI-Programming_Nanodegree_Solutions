import json
import torch
import PIL
import argparse
import matplotlib
import numpy as np
import torchvision as tv
import matplotlib.pyplot as plt
from torch import nn
from collections import OrderedDict

def main():
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--save_dir', dest="save_dir", type=str, default=".")
    parser.add_argument('--arch', dest="arch", type=str, default="vgg16")
    parser.add_argument('--learning_rate', dest="learning_rate", type=float, default="0.001")
    parser.add_argument('--hidden_units', dest="hidden_units", nargs='*', type=int, default=[6200, 1600, 400])
    parser.add_argument('--epochs', dest="epochs", type=int, default=6)
    parser.add_argument('--gpu', dest="gpu", type=bool, default=True)
    args = parser.parse_args()
    
    print('Chosen configuration:')
    for arg in vars(args):
        print("{} = {}".format(arg, getattr(args, arg)))
        
    train_dir = args.data_dir + '/train'
    valid_dir = args.data_dir + '/valid'
    test_dir = args.data_dir + '/test'
    save_dir = args.save_dir
    arch = args.arch
    input_size = 25088
    hidden_sizes = args.hidden_units
    output_size = 102
    drop_p = 0.5
    learning_rate = args.learning_rate
    epochs = args.epochs
    print_every = 64
    steps = 0
       
    # Get Datasets and Dataloaders
    train_loader, valid_loader, test_loader = get_data_loaders(train_dir, valid_dir, test_dir)

    # Set-up neural network
    model, criterion, optimizer = setup_nn(input_size, hidden_sizes, output_size, drop_p, learning_rate)
    
    # Train neural network
    train_nn(model, criterion, optimizer, epochs, print_every, steps, train_loader)
    
    # validate neural network
    valid_nn(model, test_loader)
    
    # save model
    save_model(model, input_size, hidden_sizes, output_size, drop_p, learning_rate, save_dir)
    
def get_data_loaders(train_dir, valid_dir, test_dir):
    train_transforms = tv.transforms.Compose([tv.transforms.RandomRotation(30),
                                              tv.transforms.RandomResizedCrop(224),
                                              tv.transforms.RandomHorizontalFlip(),
                                              tv.transforms.RandomVerticalFlip(),
                                              tv.transforms.ToTensor(),
                                              tv.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                              ])
    valid_transforms = tv.transforms.Compose([tv.transforms.Resize(255),
                                              tv.transforms.CenterCrop(224),
                                              tv.transforms.ToTensor(),
                                              tv.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                              ])
    test_transforms = tv.transforms.Compose([tv.transforms.Resize(255),
                                              tv.transforms.CenterCrop(224),
                                              tv.transforms.ToTensor(),
                                              tv.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                              ])

    train_dataset = tv.datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = tv.datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_dataset = tv.datasets.ImageFolder(test_dir, transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
    
    return train_loader, valid_loader, test_loader


def setup_nn(input_size, hidden_sizes, output_size, drop_p, learning_rate):
    model = tv.models.vgg16(pretrained=True)    

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('dropout',nn.Dropout(drop_p)),
        ('fc1', nn.Linear(input_size, hidden_sizes[0])),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
        ('relu2',nn.ReLU()),
        ('fc3',nn.Linear(hidden_sizes[1], hidden_sizes[2])),
        ('relu3',nn.ReLU()),
        ('fc4',nn.Linear(hidden_sizes[2], output_size)),
        ('output', nn.LogSoftmax(dim=1))
        ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    return model, criterion, optimizer


def train_nn(model, criterion, optimizer, epochs, print_every, steps, train_loader):
    # change to cuda
    model.to('cuda')

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1

            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every))

                running_loss = 0

                
def valid_nn(model, test_loader):
    correct = 0
    total = 0
    model.to('cuda')
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
                

def save_model(model, input_size, hidden_sizes, output_size, drop_p, learning_rate, save_dir):
    checkpoint = {'input_size': input_size,
                  'hidden_sizes': hidden_sizes,
                  'output_size': output_size,
                  'drop_p': drop_p,
                  'learning_rate': learning_rate,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, save_dir + '/checkpoint.pth')
        
        
if __name__ == '__main__':
    main()
