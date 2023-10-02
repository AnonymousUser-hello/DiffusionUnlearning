import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from timm.utils import accuracy, AverageMeter
from torchvision.models import resnet18

import gc
import sys
import time
import argparse
from pathlib import Path

from utils.util import (
    system_startup,
    set_random_seed,
    set_deterministic,
    Logger
)
from data.loadUTKFace import UTKFaceDataset
from data.loadCelebA_attr import CelebADataset
from data.loadVGGFace2 import VGG_Faces2HQ
from utils.resnetc import resnet20


data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def getUnlDevNum(unl_dev):
    if unl_dev == '':
        return []
    unl_dev_list = []
    for dev in unl_dev.split('+'):
        unl_dev_list.append(int(dev))
    return unl_dev_list


# def predict(net, imgs, unl_cls):
#     net.eval()
#     unl_clses = getUnlDevNum(unl_cls)

#     total = 0
#     correct = 0
#     with torch.no_grad():
#         outputs = net(imgs)
#         _, predicts = torch.max(outputs.data, 1)
#         for unl_cls in unl_clses:
#             correct += (predicts == unl_cls).sum().item()
#         total += imgs.size(0)
#     return correct, total
def predict(net, imgs, unl_cls):
    net.eval()

    total = 0
    correct = 0
    with torch.no_grad():
        outputs = net(imgs)
        _, predicts = torch.max(outputs.data, 1)
        correct += (predicts == unl_cls).sum().item()
        total += imgs.size(0)
    return correct, total


def load_data(args):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    if args.dataset == 'UTKFace':
        attr = 'age_gender_race'.split('_')
        num_classes = 4
        dataset = UTKFaceDataset(root=args.data_path + '/UTKFace/', attr=attr, transform=transform, myfilter_attr=2)
        train_length = int(len(dataset)*0.8)
        test_length = len(dataset) - int(len(dataset)*0.8)
        trainset, testset= torch.utils.data.random_split(dataset, [train_length, test_length])
    elif args.dataset == 'CIFAR10':
        num_classes = 10
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        trainset = CIFAR10(root=args.data_path + '/CIFAR10', train=True, download=False, transform=transform)
        testset = CIFAR10(root=args.data_path + '/CIFAR10', train=False, download=False, transform=transform)
    elif args.dataset == 'CelebA':
        num_classes = 1
        dataset = CelebADataset(root=args.data_path + '/CelebA', train=True,
                                transform=transform,
                                identity=(range(0, 20)))
        train_length = int(len(dataset)*0.8)
        test_length = len(dataset) - int(len(dataset)*0.8)
        trainset, testset= torch.utils.data.random_split(dataset, [train_length, test_length])
    elif args.dataset == 'VGGFace2':
        num_classes = 10
        dataset = VGG_Faces2HQ(root=args.data_path + '/VGGFace2',
                               transform=transform,
                               identity=(range(0, 10)))
        train_length = int(len(dataset)*0.8)
        test_length = len(dataset) - int(len(dataset)*0.8)
        trainset, testset= torch.utils.data.random_split(dataset, [train_length, test_length])
    else:
        assert False, 'not support the dataset yet.'

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return num_classes, trainloader, testloader, num_examples


def train(loss_fn, optimizer, model, trainloader, device, scheduler=None):
    """Train the network on the training set."""
    train_losses = AverageMeter()
    for batch_idx, (gt_imgs, gt_labs) in enumerate(trainloader):
        gt_imgs = gt_imgs.to(device)
        gt_labels = gt_labs.to(device)

        optimizer.zero_grad()
        out = model(gt_imgs)
        loss = loss_fn(out, gt_labels)
        loss.backward()

        optimizer.step()
        train_losses.update(loss, gt_imgs.size(0))
        torch.cuda.empty_cache()
        gc.collect()

    if scheduler is not None:
        scheduler.step()
    torch.cuda.empty_cache()
    gc.collect()
    return train_losses.avg.item()


def test(criterion, net, testloader, device):
    """Validate the network on the entire test set."""
    net.eval()
    top1 = AverageMeter()
    test_losses = AverageMeter()
    with torch.no_grad():
        for images, gt_labs in testloader:
            images = images.to(device)
            labels = gt_labs.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            acc1 = accuracy(outputs, labels, topk=(1,))
            top1.update(acc1[0], images.size(0))
            test_losses.update(loss, images.size(0))

    test_loss = test_losses.avg.item()
    acc = top1.avg.item()
    return test_loss, acc


def get_args_parser():
    parser = argparse.ArgumentParser(description='train a classifier.')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--data_path', type=str, default='/data/datasets')
    parser.add_argument('--dataset', type=str, default='UTKFace')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--save_dir', type=str, default='./logs/GT')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args_parser()
    args.save_dir = args.save_dir + '/' + args.dataset
    save_path = args.save_dir + '/classifier'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    sys.stdout = Logger(save_path + '/log.csv' , sys.stdout)
    print(args)

    # Choose GPU device and print status information
    device, setup = system_startup()
    set_random_seed(args.seed)
    set_deterministic()

    # Data
    num_classes, trainloader, testloader, num_examples = load_data(args)

    # Model
    # model = resnet18(pretrained=True)
    # model.fc = nn.Linear(512, num_classes)
    model = resnet20(num_classes=num_classes)
    model = model.to(**setup)
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 50], gamma=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    lr_scheduler = None

    # Train
    best_acc = 0.
    for epoch in range(args.epochs):
        ori_time = time.time()
        train_loss = train(loss_fn, optimizer, model, trainloader, device, lr_scheduler)
        test_loss, acc = test(loss_fn, model, testloader, device)
        if best_acc < acc:
            best_acc = acc
            ckp_path = save_path + '/classifier.pth'
            torch.save({
                'model': model.state_dict(),
                'acc': best_acc,
                'epoch': epoch,
            }, ckp_path)
        print('Epoch-{}, best_acc: {:.4f}, test_acc: {:.4f}, test_loss: {:.4f}, train_loss: {:.4f}, time: {:.4f}'.format(
               epoch + 1, best_acc, acc, test_loss, train_loss, time.time()-ori_time))
    torch.cuda.empty_cache()
    gc.collect()
    print('Done!')
    print()

