import logging
import math
import os
import numpy as np
from PIL import Image
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from .randaugment import RandAugmentMC
import sys
from utils.misc import cifar100_to_cifar20
from mypath import MyPath

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
cinic10_mean = (0.47889522, 0.47227842, 0.43047404)
cinic10_std = (0.24205776, 0.23828046, 0.25874835)

def majority_vote(preds, targets, num_k):
    res = []
    for i in range(num_k):
        targets_c = targets[preds==i]
        cnt = np.zeros(num_k)
        for j in range(num_k):
            cnt[j] = np.sum(targets_c==j)
        res.append((i, np.argmax(cnt)))
    return res

def noise_x_u_split(args, noise_labels, cluster_labels):
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(noise_labels)))

    all_indices = np.arange(len(noise_labels))
    labeled_idx = all_indices[cluster_labels==noise_labels]

    print("DenoiseSSL:initial selected samples %d"%(len(labeled_idx)))
    args.num_labeled = len(labeled_idx)

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    print("DnoiseSSL:final labeled samples %d; unlabeled samples %d"%(len(labeled_idx), len(unlabeled_idx)))
    return labeled_idx, unlabeled_idx

def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))

    if args.num_labeled == len(labels):
        labeled_idx = np.arange(len(labels))
    else:
        labeled_idx = []
        for i in range(args.num_classes):
            idx = np.where(labels == i)[0]
            idx = np.random.choice(idx, label_per_class, False)
            labeled_idx.extend(idx)
        labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled
    print("LDPSSL:initial selected samples %d"%(len(labeled_idx)))

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    print("LDPSSL: final labeled samples %d; unlabeled samples %d"%(len(labeled_idx), len(unlabeled_idx)))
    return labeled_idx, unlabeled_idx

def label_match(raw_cluster_label, metric, N):
    cluster_label = raw_cluster_label.copy()
    for i in range(N):
        cluster_label[raw_cluster_label==metric[i][0]] = metric[i][1]
    return cluster_label

def get_cifar10_denoisessl(args):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

    base_db_root = MyPath.db_root_dir(args.dataset)
    base_ckpt_root = MyPath.ckpt_root_dir()
    base_dataset = datasets.CIFAR10(base_db_root, train=True, download=True)

    if args.noisemode =='pate':
        noise_label_file = os.path.join(base_db_root, "dplabel", "pate","eps_"+str(args.epsilon))
        print("Loading (noise) DP label by PATE: ", noise_label_file)
        noise_train_label = torch.load(noise_label_file)      
    elif args.noisemode == 'randres':
        noise_label_file =  os.path.join(base_db_root, "dplabel", "rr", "eps"+str(args.epsilon)+".npy")
        print("Loading (noise) DP label by RandRes: ", noise_label_file)
        noise_train_label = np.load(noise_label_file)
    print("Eps: %.2f. Noisy label acc: %.6f"%(args.epsilon, np.mean(base_dataset.targets==noise_train_label)))

    all_train_label = np.array(base_dataset.targets)

    cluster_path = os.path.join(base_ckpt_root, "SCAN", args.dataset, args.arch, "selflabel", "train_cluster_pred.npy")
    print("Loading cluster from %s model: %s"%(args.arch, cluster_path))
    raw_cluster_label = np.load(cluster_path)    

    tran_metric = majority_vote(raw_cluster_label, noise_train_label, args.num_classes)
    cluster_label = label_match(raw_cluster_label, tran_metric, args.num_classes)

    print("Eps: %.2f. Cluster label acc: %.6f. Selected label ratio: %.6f."%(args.epsilon, np.mean(cluster_label==all_train_label), np.mean(cluster_label==noise_train_label)))
    print("Selected label correctness: %4f"%(np.mean(all_train_label[cluster_label==noise_train_label]==noise_train_label[cluster_label==noise_train_label])))

    noise_train_label = (noise_train_label.astype(np.int32)).tolist()
    train_labeled_idxs, train_unlabeled_idxs = noise_x_u_split(args, noise_train_label, cluster_label)

    train_labeled_dataset = CIFAR10SSL(base_db_root, train_labeled_idxs, noise=noise_train_label, train=True, transform=transform_labeled)
    train_unlabeled_dataset = CIFAR10SSL(base_db_root, train_unlabeled_idxs, noise=noise_train_label, train=True, transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))
    test_dataset = datasets.CIFAR10(base_db_root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar10_ldpssl(args):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

    base_db_root = MyPath.db_root_dir(args.dataset)
    base_ckpt_root = MyPath.ckpt_root_dir()
    base_dataset = datasets.CIFAR10(base_db_root, train=True, download=True)

    if args.noisemode == 'pate':
        noise_label_file = os.path.join(base_db_root, "dplabel", "pate","eps_"+str(args.epsilon))
        print("Loading (noise) DP label by PATE:", noise_label_file)
        noise_train_label = torch.load(noise_label_file)      
    elif args.noisemode == 'randres':
        noise_label_file =  os.path.join(base_db_root, "dplabel", "rr", "eps"+str(args.epsilon)+".npy")
        print("Loading (noise) DP label by RandRes: ", noise_label_file)
        noise_train_label = np.load(noise_label_file)
    print("Eps: %.2f. Noisy label acc: %.6f"%(args.epsilon, np.mean(base_dataset.targets==noise_train_label)))

    noise_train_label = (noise_train_label.astype(np.int32)).tolist()
    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, noise_train_label)

    train_labeled_dataset = CIFAR10SSL(base_db_root, train_labeled_idxs, noise=noise_train_label, train=True, transform=transform_labeled)
    train_unlabeled_dataset = CIFAR10SSL(base_db_root, train_unlabeled_idxs, noise=noise_train_label, train=True, transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))
    test_dataset = datasets.CIFAR10(base_db_root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar10(args):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_db_root = MyPath.db_root_dir(args.dataset)
    base_dataset = datasets.CIFAR10(base_db_root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(base_db_root, train_labeled_idxs, train=True, transform=transform_labeled)
    train_unlabeled_dataset = CIFAR10SSL(base_db_root, train_unlabeled_idxs, train=True, transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))
    test_dataset = datasets.CIFAR10(base_db_root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def get_cifar100_denoisessl(args):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_db_root = MyPath.db_root_dir(args.dataset)
    base_ckpt_root = MyPath.ckpt_root_dir()
    base_dataset = datasets.CIFAR100(base_db_root, train=True, download=True)

    if args.noisemode == 'pate':
        noise_label_file = os.path.join(base_db_root, "dplabel", "pate","eps_"+str(args.epsilon))
        print("Loading (noise) DP label by PATE: ", noise_label_file)
        noise_train_label = torch.load(noise_label_file)      
    elif args.noisemode == 'randres':
        noise_label_file =  os.path.join(base_db_root, "dplabel", "rr", "eps"+str(args.epsilon)+".npy")
        print("Loading (noise) DP label by RandRes: ", noise_label_file)
        noise_train_label = np.load(noise_label_file)
    print("Eps: %.2f. Noisy label acc: %.6f"%(args.epsilon, np.mean(base_dataset.targets==noise_train_label)))

    subclass_train_label = []
    subclass_noise_train_label = []
    for i in range(len(base_dataset.targets)):
        subclass_train_label.append(cifar100_to_cifar20._convert(base_dataset.targets[i]))
        subclass_noise_train_label.append(cifar100_to_cifar20._convert(noise_train_label[i]))
    
    subclass_train_label = np.array(subclass_train_label)
    subclass_noise_train_label = np.array(subclass_noise_train_label)
    print("Sub-class noisy label accuracy %.6f"%(np.mean(subclass_train_label==subclass_noise_train_label)))

    cluster_path = os.path.join(base_ckpt_root, "SCAN", "cifar20", args.arch, "selflabel", "train_cluster_pred.npy")
    print("Loading cluster from %s model: %s"%(args.arch, cluster_path))
    raw_cluster_label = np.load(cluster_path)

    tran_metric = majority_vote(raw_cluster_label, subclass_noise_train_label, args.num_classes)
    cluster_label = label_match(raw_cluster_label, tran_metric, args.num_classes)

    all_train_label = np.array(base_dataset.targets)

    print("Eps: %.2f. Cluster label accuracy: %.6f. Selected ratio: %.6f."%(args.epsilon, np.mean(cluster_label==subclass_train_label), np.mean(cluster_label==subclass_noise_train_label)))
    print("Finegrained Label Accuracy by matched sub-class label: %.6f"%(np.mean(noise_train_label[cluster_label==subclass_noise_train_label] == all_train_label[cluster_label==subclass_noise_train_label])))
    noise_train_label = (noise_train_label.astype(np.int32)).tolist()

    train_labeled_idxs, train_unlabeled_idxs = noise_x_u_split(args, subclass_noise_train_label, cluster_label)

    train_labeled_dataset = CIFAR100SSL(base_db_root, train_labeled_idxs, noise = noise_train_label, train=True, transform=transform_labeled)
    train_unlabeled_dataset = CIFAR100SSL(base_db_root, train_unlabeled_idxs, noise = noise_train_label, train=True, transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))
    test_dataset = datasets.CIFAR100(base_db_root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100_ldpssl(args):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_db_root = MyPath.db_root_dir(args.dataset)
    base_ckpt_root = MyPath.ckpt_root_dir()
    base_dataset = datasets.CIFAR100(base_db_root, train=True, download=True)

    if args.noisemode == 'pate':
        noise_label_file = os.path.join(base_db_root, "dplabel", "pate","eps_"+str(args.epsilon))
        print("Loading (noise) DP label by PATE: ", noise_label_file)
        noise_train_label = torch.load(noise_label_file)      
    elif args.noisemode == 'randres':
        noise_label_file =  os.path.join(base_db_root, "dplabel", "rr", "eps"+str(args.epsilon)+".npy")
        print("Loading (noise) DP label by RandRes: ", noise_label_file)
        noise_train_label = np.load(noise_label_file)
    print("Eps: %.2f. Noisy label acc: %.6f"%(args.epsilon, np.mean(base_dataset.targets==noise_train_label)))

    noise_train_label = (noise_train_label.astype(np.int32)).tolist()
    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, noise_train_label)

    train_labeled_dataset = CIFAR100SSL(base_db_root, train_labeled_idxs, noise = noise_train_label, train=True, transform=transform_labeled)
    train_unlabeled_dataset = CIFAR100SSL(base_db_root, train_unlabeled_idxs, noise = noise_train_label, train=True, transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))
    test_dataset = datasets.CIFAR100(base_db_root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(args):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_db_root = MyPath.db_root_dir(args.dataset)
    base_ckpt_root = MyPath.ckpt_root_dir()
    base_dataset = datasets.CIFAR100(base_db_root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(base_db_root, train_labeled_idxs, train=True, transform=transform_labeled)
    train_unlabeled_dataset = CIFAR100SSL(base_db_root, train_unlabeled_idxs, train=True, transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))
    test_dataset = datasets.CIFAR100(base_db_root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def get_cinic10_denoisessl(args):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic10_mean, std=cinic10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic10_mean, std=cinic10_std)
    ])
    base_db_root = MyPath.db_root_dir(args.dataset)
    base_ckpt_root = MyPath.ckpt_root_dir()
    base_dataset = CINIC10(base_db_root, filen = 'train')

    if args.noisemode =='pate':
        noise_label_file = os.path.join(base_db_root, "dplabel", "pate","eps_"+str(args.epsilon))
        print("Loading (noise) DP label by PATE: ", noise_label_file)
        noise_train_label = torch.load(noise_label_file)      
    elif args.noisemode == 'randres':
        noise_label_file =  os.path.join(base_db_root, "dplabel", "rr", "eps"+str(args.epsilon)+".npy")
        print("Loading (noise) DP label by RandRes: ", noise_label_file)
        noise_train_label = np.load(noise_label_file)
    print("Eps: %.2f. Noisy label acc: %.6f"%(args.epsilon, np.mean(base_dataset.targets==noise_train_label)))

    all_train_label = np.array(base_dataset.targets)

    cluster_path = os.path.join(base_ckpt_root, "SCAN", args.dataset, args.arch, "selflabel", "train_cluster_pred.npy")
    print("Loading cluster from %s model: %s"%(args.arch, cluster_path))
    raw_cluster_label = np.load(cluster_path)

    tran_metric = majority_vote(raw_cluster_label, noise_train_label, args.num_classes)
    cluster_label = label_match(raw_cluster_label, tran_metric, args.num_classes)

    print("Eps: %.2f. Cluster label acc: %.6f. Selected label ratio: %.6f."%(args.epsilon, np.mean(cluster_label==all_train_label), np.mean(cluster_label==noise_train_label)))
    print("Selected label correctness: %4f"%(np.mean(all_train_label[cluster_label==noise_train_label]==noise_train_label[cluster_label==noise_train_label])))

    noise_train_label = (noise_train_label.astype(np.int32)).tolist()
    
    train_labeled_idxs, train_unlabeled_idxs = noise_x_u_split(args, noise_train_label, cluster_label, all_train_label)

    train_labeled_dataset = CINICSSL(base_db_root, filen='train', indexs=train_labeled_idxs, noise=noise_train_label, transform=transform_labeled)
    train_unlabeled_dataset = CINICSSL(base_db_root, filen='train', indexs=train_unlabeled_idxs, noise=noise_train_label, transform=TransformFixMatch(mean=cinic10_mean, std=cinic10_std))
    test_dataset = CINICSSL(base_db_root, filen='valid', transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def get_cinic10_ldpssl(args):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic10_mean, std=cinic10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic10_mean, std=cinic10_std)
    ])
    base_db_root = MyPath.db_root_dir(args.dataset)
    base_dataset = CINIC10(base_db_root, filen = 'train')

    if args.noisemode =='pate':
        noise_label_file = os.path.join(base_db_root, "dplabel", "pate","eps_"+str(args.epsilon))
        print("Loading (noise) DP label by PATE: ", noise_label_file)
        noise_train_label = torch.load(noise_label_file)      
    elif args.noisemode == 'randres':
        noise_label_file =  os.path.join(base_db_root, "dplabel", "rr", "eps"+str(args.epsilon)+".npy")
        print("Loading (noise) DP label by RandRes: ", noise_label_file)
        noise_train_label = np.load(noise_label_file)
    print("Eps: %.2f. Noisy label acc: %.6f"%(args.epsilon, np.mean(base_dataset.targets==noise_train_label)))

    noise_train_label = (noise_train_label.astype(np.int32)).tolist()
    
    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, noise_train_label)
    train_labeled_dataset = CINICSSL(base_db_root, filen='train', indexs=train_labeled_idxs, noise=noise_train_label, transform=transform_labeled)
    train_unlabeled_dataset = CINICSSL(base_db_root, filen='train', indexs=train_unlabeled_idxs, noise=noise_train_label, transform=TransformFixMatch(mean=cinic10_mean, std=cinic10_std))
    test_dataset = CINICSSL(base_db_root, filen='valid', transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cinic10(args):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic10_mean, std=cinic10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic10_mean, std=cinic10_std)
    ])
    base_db_root = MyPath.db_root_dir(args.dataset)
    base_dataset = CINIC10(base_db_root, filen = 'train')

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset.targets)

    train_labeled_dataset = CINICSSL(base_db_root, filen='train',indexs=train_labeled_idxs,transform=transform_labeled)
    train_unlabeled_dataset = CINICSSL(base_db_root, filen='train', indexs=train_unlabeled_idxs, transform=TransformFixMatch(mean=cinic10_mean, std=cinic10_std))
    test_dataset = CINICSSL(base_db_root, filen='valid', transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, noise=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if noise is not None:
            self.targets = noise
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, noise = None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if noise is not None:
            self.targets = noise
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CINIC10(Dataset):
    """`adapt from CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    def __init__(self, root, filen=True, transform=None, target_transform=None):

        super(CINIC10, self).__init__()
        self.root = os.path.join(root, 'npy')
        self.transform = transform
        self.target_transform = target_transform
        self.file_n = filen  # training set or test set
        self.classes = ['frog', 'airplane', 'horse', 'truck', 'cat', 'deer', 'automobile', 'dog', 'bird', 'ship']

        self.data = np.load(os.path.join(self.root, filen+"_data.npy"))
        self.targets = np.load(os.path.join(self.root, filen+"_label.npy"))
        self.targets = self.targets.tolist()

        self._load_meta()

    def _load_meta(self):
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """
        img, target = self.data[index], self.targets[index]
        img_size = (img.shape[0], img.shape[1])
        img = Image.fromarray(img)
        class_name = self.classes[target]        

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def get_image(self, index):
        img = self.data[index]
        return img
        
    def __len__(self):
        return len(self.data)


class CINICSSL(CINIC10):
    def __init__(self, root, filen, indexs=None, noise=None,
                 transform=None, target_transform=None):
        super().__init__(root=root, filen=filen, transform=transform,target_transform=target_transform)
        if noise is not None:
            self.targets = noise
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

DATASET_GETTERS = {'cifar10ssl': get_cifar10,
                   'cifar100ssl': get_cifar100,
                   'cifar10denoisessl': get_cifar10_denoisessl,
                   'cifar100denoisessl': get_cifar100_denoisessl,
                   'cifar10ldpssl': get_cifar10_ldpssl,
                   'cifar100ldpssl': get_cifar100_ldpssl,
                   'cinic10ssl': get_cinic10,
                   'cinic10denoisessl': get_cinic10_denoisessl,
                   'cinic10ldpssl': get_cinic10_ldpssl}