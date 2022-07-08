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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

import sys
sys.path.append("./../")
from torchvision import datasets
from dataset.cinic10 import CINIC10
from mypath import MyPath

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

cinic_mean = [0.47889522, 0.47227842, 0.43047404]
cinic_std = [0.24205776, 0.23828046, 0.25874835]
cifar10_mean = [0.4914, 0.4822, 0.4465]
cifar10_std = [0.2023, 0.1994, 0.2010]
cifar100_mean = [0.5071, 0.4867, 0.4408]
cifar100_std = [0.2675, 0.2565, 0.2761]

def test(args, test_loader, model):

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    test_preds = np.zeros((0, ))

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            test_preds = np.concatenate((test_preds, torch.argmax(outputs,dim=1).detach().cpu().numpy()))
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])

    print("top-1 acc: {:.2f}".format(top1.avg))
    print("top-5 acc: {:.2f}".format(top5.avg))
    return test_preds.astype(np.int32)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='number of workers')
    parser.add_argument('--dataset', default='cinic10', type=str,
                        choices=['cinic10', 'cifar10', 'cifar100'],
                        help='dataset name')
    parser.add_argument('--arch', default='resnet18', type=str,
                        choices=['wideresnet', 'vgg', 'resnet18'],
                        help='dataset name')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--num-teachers', type=int, default=500,
                        help='number of teacher models')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")

    args= parser.parse_args()

    db_root_dir = MyPath.db_root_dir(args.dataset)
    args.data_path = db_root_dir  
    args.root_path = os.path.join(MyPath.ckpt_root_dir(), "PATEFM", args.dataset, str(args.num_teachers)+'teachers')
    
    if args.dataset == 'cifar10' or args.dataset =='cinic10':
        args.num_class=10
        args.len_dataset = 50000        
    else:
        args.num_class=100
        args.len_dataset = 90000


    print(dict(args._get_kwargs()))

    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_class)
        elif args.arch == 'resnet18':
            import models.resnet as resnetmodel
            model = resnetmodel.resnet18(num_class=args.num_class)
        print("Total params: {:.2f}M".format(sum(p.numel() for p in model.parameters())/1e6))
        return model

    print("| Building net")
    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    model.to(args.device)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)


    if args.dataset == 'cinic10':
        transform_test = transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize(cinic_mean, cinic_std)])
        filepath = os.path.join(args.data_path, 'npy')
        trainset = CINIC10(root=filepath, filen = 'train', transform = transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    elif args.dataset == 'cifar10':
        transform_test = transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize(cifar10_mean, cifar10_std)])
        filepath = args.data_path
        trainset = datasets.CIFAR10(root=filepath, train=True, transform = transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=1)
        
    elif args.dataset == 'cifar100':
        transform_test = transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize(cifar100_mean, cifar100_std)])
        filepath = args.data_path
        trainset = datasets.CIFAR100(root=filepath, train=True, transform = transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=1) 

    pred_count = np.zeros((args.len_dataset, args.num_class))

    for i in range(args.num_teachers):

        print("==> Resuming from checkpoint..", str(i))
        #assert os.path.isfile(args.root_path), "Error: no checkpoint directory found!"
        #args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(os.path.join(args.root_path, str(i), 'checkpoint.pth.tar'), map_location='cpu')
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model


        trainset_preds = test(args, train_loader, test_model)

        pred_count[np.arange(len(pred_count)), trainset_preds]+=1
        sys.stdout.flush()

    np.save(os.path.join(MyPath.ckpt_root_dir(), "PATEFM", args.dataset, str(args.num_teachers)+'teachers', 'aggregate_votes.npy'), pred_count)
    