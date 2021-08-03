from __future__ import print_function, division

import random
from sys import prefix
from numpy import core

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
from meter import AverageMeter, ProgressMeter

plt.ion()  # interactive mode
TYPE = "dataparallel"


def set_seed(seed=202012):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def imshow(inp, title=None):
    """Imshow for Tensor"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)  # 将inp的元素转化为0-1之间
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

    plt.pause(0.001)


def train_model(model, criterion, optimizer, scheduler, gpus, num_epochs=25):
    

    best_model_wts = copy.deepcopy(model.state_dict())
    global best_acc

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('Learning rage: ', optimizer.state_dict()['param_groups'][0]['lr'])
        print('-' * 10)

        end = time.time()
        for phase in ['train', 'val']:
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')
            top1 = AverageMeter('Acc', ':6.2f')
            progress = ProgressMeter(len(dataloaders[phase]), [batch_time, data_time, 
            losses, top1], prefix="Epoch: [{}]".format(epoch))

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            pbar = tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase]))
            # 迭代数据
            for step, (inputs, lables) in pbar:

                data_time.update(time.time() - end)

                # inputs = inputs.to(device)
                # lables = lables.to(device)
                inputs = inputs.cuda(non_blocking=True)
                lables = lables.cuda(non_blocking=True)

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, lables)

                    # 只在训练模式下进行反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                acc1 = accuracy(outputs, lables)
                losses.update(loss.item(), inputs.size(0))
                top1.update(acc1[0], inputs.size(0))
                

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == lables.data)

                if (step + 1) == len(dataloaders[phase]):
                    description = f'epoch {epoch} loss: {loss.item() * inputs.size(0):.4f} ' \
                                  f'acc: {torch.sum(preds == lables.data):.4f}'

                    pbar.set_description(description)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60
    ))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return

        model.train(model=was_training)


def accuracy(output, label, topk=(1, )):
    """计算topk的准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = label.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()  # 转置
        correct = pred.eq(label.view(1, -1).expand_as(pred))

        res = []

        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



if __name__ == "__main__":

    set_seed()

    gpus = [0, 1, 2, 3]
    batch_size = 8

    data_transforms = {
        "train": torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(3),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ]),
        "val": torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(3),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ])
    }

    image_datasets = {
        "train": torchvision.datasets.MNIST('./data/MNIST', train=True, download=True,
                                            transform=data_transforms["train"]),
        "val": torchvision.datasets.MNIST('./data/MNIST', train=False, download=True,
                                          transform=data_transforms["val"]),
    }

    dataloaders = { x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=batch_size * len(gpus),
                                                  shuffle=True,
                                                  num_workers=4) for x in ['train', 'val']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 微调网络
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    model_ft.cuda()

    # model_ft = model_ft.to(device)
    model_ft = nn.DataParallel(model_ft.cuda(), device_ids=gpus, output_device=gpus[0])

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, gpus, num_epochs=10)

    torch.save(model_ft.state_dict(), "./models/" + TYPE + "_resnet18.pkl")



