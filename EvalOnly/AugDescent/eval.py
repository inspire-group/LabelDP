from warnings import filterwarnings

filterwarnings("ignore")

import os
import random
import sys
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.mixture import GaussianMixture

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

import sys
sys.path.append("./../")
sys.path.append("./../dataset")
from cinic10 import CINIC10
from mypath import MyPath
cinic_mean = [0.47889522, 0.47227842, 0.43047404]
cinic_std = [0.24205776, 0.23828046, 0.25874835]
cifar10_mean = [0.4914, 0.4822, 0.4465]
cifar10_std = [0.2023, 0.1994, 0.2010]
cifar100_mean = [0.5071, 0.4867, 0.4408]
cifar100_std = [0.2675, 0.2565, 0.2761]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch LabelDP')
    parser.add_argument('--epsilon', type=float, default=3,
                        help='random respond epsilon')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')    
    parser.add_argument('--noisemode', type = str, 
                        choices=['sym', 'asym', 'randres', 'pate', 'ndp'],
                        help='noise type')
    parser.add_argument('--dataset', type = str, 
                        choices=['cifar10', 'cifar100', 'cinic10'],)
    parser.add_argument("--preset", required=True, type=str)
    parser.add_argument("--arch", type = str, choices = ['resnet18', 'vgg', 'wideresnet28'], default = 'resnet18')
    
    args= parser.parse_args()
    print(dict(args._get_kwargs()))

    if args.epsilon == int(args.epsilon):
        args.epsilon = int(args.epsilon)

    subpresets = args.preset.split(".")
    new_subpresets = [args.noisemode, str(args.epsilon),subpresets[0]]

    db_root_dir = MyPath.db_root_dir(args.dataset)
    args.data_path = db_root_dir  
    ckpt_root_dir = os.path.join(MyPath.ckpt_root_dir(), 'AugDescent', args.dataset)
    if args.noisemode == 'ndp':
        args.checkpoint_path = os.path.join(ckpt_root_dir, args.arch, args.noisemode, "0sym",  new_subpresets[2])
    else:
        args.checkpoint_path = os.path.join(ckpt_root_dir, args.arch, args.noisemode, str(args.epsilon),  new_subpresets[2])
    
    if args.dataset == 'cifar10' or args.dataset =='cinic10':
        args.num_class=10
    else:
        args.num_class=100

    print(args)

    def test(testloader, net1, net2):
        net1.eval()
        net2.eval()
        all_targets = []
        all_predicted = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs1 = net1(inputs)
                outputs2 = net2(inputs)
                outputs = outputs1 + outputs2
                _, predicted = torch.max(outputs, 1)

                all_targets += targets.tolist()
                all_predicted += predicted.tolist()

        accuracy = accuracy_score(all_targets, all_predicted)
        precision = precision_score(all_targets, all_predicted, average="weighted")
        recall = recall_score(all_targets, all_predicted, average="weighted")
        f1 = f1_score(all_targets, all_predicted, average="weighted")
        results = "Accuracy: %.3f, Precision: %.3f, Recall: %.3f, F1: %.3f" % (
            accuracy * 100,
            precision * 100,
            recall * 100,
            f1 * 100)
        print(results)
        return accuracy


    def create_model():
        if args.arch == 'resnet18':
            import models.resnet as resnetmodel

        model = resnetmodel.resnet18(num_class=args.num_class)
        model = model.to(device)
        #model = torch.nn.DataParallel(model, device_ids=devices).cuda()
        return model

    print("| Building net")
    #devices = range(torch.cuda.device_count())
    net1 = create_model()
    net2 = create_model()
    cudnn.benchmark = True


    with open(args.checkpoint_path + f"/best/{args.preset}.pth.tar", "rb") as p:
        unpickled = torch.load(p)
    net1.load_state_dict(unpickled["net1"])
    net2.load_state_dict(unpickled["net2"])


    epoch = unpickled["epoch"]

    CE = nn.CrossEntropyLoss(reduction="none")
    CEloss = nn.CrossEntropyLoss()


    print("Saved model at epoch ", epoch)


    if args.dataset == 'cinic10':
        transform_test = transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize(cinic_mean, cinic_std)])
        filepath = os.path.join(args.data_path, 'npy')
        trainset = CINIC10(root=filepath, filen = 'train', transform = transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=1)
        
        valset = CINIC10(root=filepath, filen = 'valid', transform = transform_test)
        valid_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=1)    
    
        testset = CINIC10(root=filepath, filen = 'test', transform = transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=1)    

        print("train set")
        trainset_acc = test(train_loader, net1, net2)
        print("validation set")
        validset_acc = test(valid_loader, net1, net2)
        print("test set")
        testset_acc = test(test_loader, net1, net2)

    elif args.dataset == 'cifar10':
        transform_test = transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize(cifar10_mean, cifar10_std)])
        filepath = args.data_path
        trainset = datasets.CIFAR10(root=filepath, train=True, transform = transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=1)
        
        valset = datasets.CIFAR10(root=filepath, train=False, transform = transform_test)
        valid_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=1)    
    
        print("train set")
        trainset_acc = test(train_loader, net1, net2)
        print("validation set")
        validset_acc = test(valid_loader, net1, net2)

    elif args.dataset == 'cifar100':
        transform_test = transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize(cifar100_mean, cifar100_std)])
        filepath = args.data_path
        trainset = datasets.CIFAR100(root=filepath, train=True, transform = transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=1)
        
        valset = datasets.CIFAR100(root=filepath, train=False, transform = transform_test)
        valid_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=1)    
    
        print("train set")
        trainset_acc = test(train_loader, net1, net2)
        print("validation set")
        validset_acc = test(valid_loader, net1, net2)
