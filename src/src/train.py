#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/8/28 11:04
# @File     : train.py

"""
import argparse
import sys

import torch
import torchvision

from models.lenet import LeNet
from utils import pre_process
from matplotlib import pyplot as plt
from collections import Counter
from PIL import Image

def get_data_loader(batch_size, augment=[None, False, False]):
    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='src/data/',
                                               train=True,
                                               transform=pre_process.data_augment_transform(augment[0], augment[1], augment[2]),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='src/data/',
                                              train=False,
                                              transform=pre_process.normal_transform())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)


    return train_loader, test_loader


def evaluate(model, test_loader, device, show_error=False, output_path=None):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        
        epoch_error_dict: dict = {}
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 2.7 save false prediction
            for i in range(len(predicted)):
                pre = predicted.tolist()[i]
                lab = labels.tolist()[i]
                if pre != lab:
                    if (pre, lab) not in epoch_error_dict:
                        epoch_error_dict[(pre, lab)] = 1
                        if show_error:
                            plt.imsave(output_path + "images/errors/" + str(pre) + str(lab) + ".png", images[i][0].numpy())
                    else:
                        epoch_error_dict[(pre, lab)] += 1

        if output_path is not None:
            sys.stdout = open(output_path + "output.log", "a+")
        print('Test Accuracy of the model is: {} %'.format(100 * correct / total))
        if output_path is not None:
            sys.stdout.close()
    
    # 2.5 2.7 return the accuracy and the error
    return 100 * correct / total, epoch_error_dict


def save_model(model, save_path='lenet.pth'):
    ckpt_dict = {
        'state_dict': model.state_dict()
    }
    torch.save(ckpt_dict, save_path)


def train(epochs=10, batch_size=256, learning_rate=0.01, num_classes=10, optim='Adam', add_conv_layer=False, show_loss=False, augment=[None, False, False], show_accuracy=False, show_error=False, output_path=""):

    # fetch data
    train_loader, test_loader = get_data_loader(batch_size, augment)
    
    # Loss and optimizer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = LeNet(num_classes, add_conv_layer=add_conv_layer).to(device)
    
    # 2.6 model visualization
    if add_conv_layer:
        data = torch.randn(1, 1, 28, 28)
        net = LeNet()
        torch.onnx.export(net, data, 'LeNet.onnx', export_params=True, opset_version=8)
        
    criterion = torch.nn.CrossEntropyLoss()

    # 2.4 select the optimizer
    if optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError('Invalid optimizer')

    # start train
    loss_list: list = []
    accuracy_list: list = []
    error_dict: dict = {}
    
    total_step = len(train_loader)
    for epoch in range(epochs):
        epoch_loss_list: list = []
        for i, (images, labels) in enumerate(train_loader):
            
            # get image and label
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                if output_path is not None:
                    sys.stdout = open(output_path + "output.log", "a+")
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}'
                      .format(epoch + 0, epochs, i + 1, total_step, loss.item()))
                if output_path is not None:
                    sys.stdout.close()
            epoch_loss_list.append(loss.item())
            
        # evaluate after epoch train
        accuracy, epoch_error_dict = evaluate(model, test_loader, device, show_error=show_error, output_path=output_path)
        accuracy_list.append(accuracy)
        error_dict = dict(Counter(error_dict) + Counter(epoch_error_dict))
        
        # 2.3 save and analyse losses
        loss_list.append(sum(epoch_loss_list)/len(epoch_loss_list))
    if show_loss:
        plt.plot(range(1, epochs + 1), loss_list)
        plt.title("Loss against epoch(" + str(optim)+")")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(output_path + "images/loss.png")
        plt.clf()
        
    # show accuracy
    if show_accuracy:
        plt.plot(range(1, epochs + 1), accuracy_list)
        plt.title("Accuracy against epoch(" + str(optim)+")")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.savefig(output_path + "images/accuracy.png")
        plt.clf()
        
    # 2.7 analyse errors
    if show_error:
        false_count_list: list= sorted(error_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        plt.bar([4*i for i in range(5)], [false_count[1] for false_count in false_count_list], tick_label=[str(false_count[0]) for false_count in false_count_list])
        plt.title("5 Most Frequent False Predictions")
        plt.xlabel("(predicted, actual)")
        plt.ylabel("Count")
        plt.savefig(output_path + "images/errors.png")
        plt.show()
        concat_image = Image.new('RGB', (140, 28))
        for i in range(5):
            false_count = false_count_list[i]
            img = Image.open(output_path + "images/errors/" + str(false_count[0][0]) + str(false_count[0][1]) + ".png")
            concat_image.paste(img, (28*i, 0))
            plt.imshow(concat_image)
        plt.suptitle("Images of 5 Most Frequent False Predictions")
        plt.title("Predictions: " + str([false_count[0][0] for false_count in false_count_list]) +"\nLabels: " + str([false_count[0][1] for false_count in false_count_list]))
        plt.axis('off')
        plt.xlabel([str(false_count[0]) for false_count in false_count_list])
        plt.savefig(output_path + "images/errors_concat.png")
        plt.show()
        

    # save the trained model
    save_model(model, save_path=output_path + 'lenet.pth')
    return model, loss_list, accuracy_list


def parse_args():
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--add_conv_layer', type=bool, default=False)
    parser.add_argument('--show_loss', type=bool, default=False)
    parser.add_argument('--augment', type=list, default=[None, False, False])
    parser.add_argument('--show_accuracy', type=bool, default=False)
    parser.add_argument('--show_error', type=bool, default=False)
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    args = parse_args()
    
    train(args.epochs, args.batch_size, args.lr, args.num_classes, args.optim, args.add_conv_layer, args.show_loss, args.augment, args.show_accuracy, args.show_error, args.output_path)
