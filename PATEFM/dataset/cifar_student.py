import logging
import math
import os
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
import torch
from .randaugment import RandAugmentMC
import sys
from utils.misc import cifar100_to_cifar20
from mypath import MyPath

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

cinic10_mean = [0.47889522, 0.47227842, 0.43047404]
cinic10_std = [0.24205776, 0.23828046, 0.25874835]

def x_u_split(args):
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(args.len_dataset))

    labeled_idx = np.load(os.path.join(MyPath.ckpt_root_dir(), "PATEFM", args.dataset, str(args.num_teachers)+'teachers', "eps"+str(args.epsilon), "queried_inds.npy"))
    args.num_labeled = len(labeled_idx)

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    print("PATE:final labeled samples %d; unlabeled samples %d"%(len(labeled_idx), len(unlabeled_idx)))
    return labeled_idx, unlabeled_idx


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
    labeled_idx = np.load(os.path.join(MyPath.ckpt_root_dir(), "PATEFM", args.dataset, str(args.num_teachers)+'teachers', "eps"+str(args.epsilon), "queried_inds.npy"))
    noise_label = np.load(os.path.join(MyPath.ckpt_root_dir(), "PATEFM", args.dataset, str(args.num_teachers)+'teachers', "eps"+str(args.epsilon), "noise.npy"))
    noise_full_label = np.array(base_dataset.targets)
    noise_full_label[labeled_idx] = noise_label

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args)

    train_labeled_dataset = CIFAR10SSL(base_db_root, train_labeled_idxs, noise=noise_full_label, train=True, transform=transform_labeled)
    train_unlabeled_dataset = CIFAR10SSL(base_db_root, train_unlabeled_idxs, noise=noise_full_label, train=True, transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))
    test_dataset = datasets.CIFAR10(base_db_root, train=False, transform=transform_val, download=False)

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

    labeled_idx = np.load(os.path.join(MyPath.ckpt_root_dir(), "PATEFM", args.dataset, str(args.num_teachers)+'teachers', "eps"+str(args.epsilon), "queried_inds.npy"))
    noise_label = np.load(os.path.join(MyPath.ckpt_root_dir(), "PATEFM", args.dataset, str(args.num_teachers)+'teachers', "eps"+str(args.epsilon), "noise.npy"))
    noise_full_label = np.array(base_dataset.targets)
    noise_full_label[labeled_idx] = noise_label

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args)

    train_labeled_dataset = CINICSSL(base_db_root, filen='train', noise=noise_full_label, indexs=train_labeled_idxs,transform=transform_labeled)
    train_unlabeled_dataset = CINICSSL(base_db_root, filen='train', noise=noise_full_label, indexs=train_unlabeled_idxs, transform=TransformFixMatch(mean=cinic10_mean, std=cinic10_std))
    test_dataset = CINICSSL(base_db_root, filen='valid', transform=transform_val)

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

    labeled_idx = np.load(os.path.join(MyPath.ckpt_root_dir(), "PATEFM", args.dataset, str(args.num_teachers)+'teachers', "eps"+str(args.epsilon), "queried_inds.npy"))
    noise_label = np.load(os.path.join(MyPath.ckpt_root_dir(), "PATEFM", args.dataset, str(args.num_teachers)+'teachers', "eps"+str(args.epsilon), "noise.npy"))
    noise_full_label = np.array(base_dataset.targets)
    noise_full_label[labeled_idx] = noise_label

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args)

    train_labeled_dataset = CIFAR100SSL(base_db_root, train_labeled_idxs, noise=noise_full_label, train=True, transform=transform_labeled)
    train_unlabeled_dataset = CIFAR100SSL(base_db_root, train_unlabeled_idxs, noise=noise_full_label, train=True, transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))
    test_dataset = datasets.CIFAR100(base_db_root, train=False, transform=transform_val, download=False)

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

DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cinic10': get_cinic10,
                   'cifar100': get_cifar100,}