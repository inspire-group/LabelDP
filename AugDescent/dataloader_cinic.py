import json
import os
import pickle
import random

import _pickle as cPickle
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from autoaugment import CIFAR10Policy
from randaugment import *

cinic10_mean = (0.47889522, 0.47227842, 0.43047404)
cinic10_std = (0.24205776, 0.23828046, 0.25874835)

transform_none_10_compose = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(cinic10_mean, cinic10_std),
    ]
)

transform_weak_10_compose = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cinic10_mean, cinic10_std),
    ]
)

transform_strong_10_compose = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize(cinic10_mean, cinic10_std),
    ]
)

transform_strong_randaugment_10_compose = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # RandAugment(1, 6),
        transforms.ToTensor(),
        transforms.Normalize(cinic10_mean, cinic10_std ),
    ]
)


class cinic_dataset(Dataset):
    def __init__(
        self,
        dataset,
        r,
        noise_mode,
        root_dir,
        transform,
        mode,
        noise_file="",
        preaug_file="",
        pred=[],
        probability=[],
    ):
        self.r = r
        self.transform = transform
        self.mode = mode
        self.preaug_file = preaug_file

        print("dataset", noise_mode)

        if self.mode == "test":
            self.test_data = np.load(os.path.join(root_dir, 'npy', 'valid_data.npy'))
            self.test_label = np.load(os.path.join(root_dir, 'npy', 'valid_label.npy'))
            self.test_label = self.test_label.tolist()

        else:
            train_data = np.load(os.path.join(root_dir, 'npy', 'train_data.npy'))
            train_label = np.load(os.path.join(root_dir, 'npy', 'train_label.npy'))
            train_label = train_label.tolist()

            self.train_label = train_label

            if noise_mode == 'pate':
                eps = self.r
                noise_label = torch.load(noise_file)
                print("Loading dataset by PATE EPS:%.2f. Correct data: %d/%.4f"%(eps, np.sum(train_label == np.array(noise_label)), np.mean(train_label == np.array(noise_label))))
            
            elif noise_mode == "randres":
                eps = self.r
                noise_label = np.load(noise_file)
                print("Loading dataset by RandRes EPS:%.2f. Correct data: %d/%.4f"%(eps, np.sum(train_label == np.array(noise_label)), np.mean(train_label == np.array(noise_label))))
            
            elif noise_mode == 'ndp':
                noise_label = train_label
                print("Loading the ground truth label")

            else:
                noise_label = []
                NN = 90000
                idx = list(range(NN))
                random.shuffle(idx)
                num_noise = int(self.r * NN)
                noise_idx = idx[:num_noise]
                for i in range(NN):
                    if i in noise_idx:
                        noiselabel = random.randint(0, 9)
                    else:
                        noise_label.append(train_label[i])
                print(f"saving noisy labels to {noise_file}...")
                json.dump(noise_label, open(noise_file, "w"), indent=4, sort_keys=True)
            
            if self.preaug_file != "":
                all_augmented = torch.load(self.preaug_file)
                train_data = np.concatenate(
                    (
                        train_data,
                        np.array(all_augmented["samples"], dtype=np.uint8).transpose(
                            (0, 2, 3, 1)
                        ),
                    )
                )
                noise_label = np.concatenate(
                    (
                        noise_label,
                        np.array(all_augmented["labels"]),
                    )
                )

            if self.mode == "all":
                self.train_data = train_data
                self.noise_label = noise_label
            else:
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]

                elif self.mode == "unlabeled":
                    pred_idx = (1 - pred).nonzero()[0]

                self.train_data = train_data[pred_idx]

                self.noise_label = [noise_label[i] for i in pred_idx]
                print("%s data has a size of %d" % (self.mode, len(self.train_data)))

    def __getitem__(self, index):
        if self.mode == "labeled":
            img, target, prob = (
                self.train_data[index],
                self.noise_label[index],
                self.probability[index],
            )
            img = Image.fromarray(img)
            img1 = self.transform[0](img)
            img2 = self.transform[1](img)
            if self.transform[2] == None:
                img3 = img1
                img4 = img2
            else:
                img3 = self.transform[2](img)
                img4 = self.transform[3](img)
            return img1, img2, img3, img4, target, prob
        elif self.mode == "unlabeled":
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform[0](img)
            img2 = self.transform[1](img)
            if self.transform[2] == None:
                img3 = img1
                img4 = img2
            else:
                img3 = self.transform[2](img)
                img4 = self.transform[3](img)
            return img1, img2, img3, img4
        elif self.mode == "all":
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target, index
        elif self.mode == "test":
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode != "test":
            return len(self.train_data)
        else:
            return len(self.test_data)


class dataset_dataloader:
    # workaround for windows because
    # python can't pickle lambdas :(

    def prob_transform_10(self, x):
        if random.random() < self.warmup_aug_prob:
            return transform_strong_10_compose(x)
        else:
            return transform_weak_10_compose(x)

    def transform_strong_10(self, x):
        return transform_strong_10_compose(x)

    def transform_weak_10(self, x):
        return transform_weak_10_compose(x)

    def transform_strong_randaugment_10(self, x):
        return transform_strong_randaugment_10_compose(x)

    def transform_none_10(self, x):
        return transform_none_10_compose(x)

    def __init__(
        self,
        dataset,
        r,
        noise_mode,
        batch_size,
        warmup_batch_size,
        num_workers,
        root_dir,
        noise_file="",
        preaug_file="",
        augmentation_strategy={},
    ):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.warmup_batch_size = warmup_batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.noise_file = noise_file
        self.preaug_file = preaug_file
        self.warmup_aug_prob = augmentation_strategy.warmup_aug_probability
        print("loader", noise_mode)
        if "randaugment_params" in augmentation_strategy:
            p = augmentation_strategy["randaugment_params"]
            a = RandAugment(p["n"], p["m"])
            transform_strong_randaugment_10_compose.transforms.insert(2, a)
        self.transforms = {
            "warmup": self.__getattribute__(augmentation_strategy.warmup_transform),
            "unlabeled": [None for i in range(4)],
            "labeled": [None for i in range(4)],
            "test": None,
        }
        # workaround so it works on both windows and linux
        for i in range(len(augmentation_strategy.unlabeled_transforms)):
            self.transforms["unlabeled"][i] = self.__getattribute__(
                augmentation_strategy.unlabeled_transforms[i]
            )
        for i in range(len(augmentation_strategy.labeled_transforms)):
            self.transforms["labeled"][i] = self.__getattribute__(
                augmentation_strategy.labeled_transforms[i]
            )

        self.transforms["test"] = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(cinic10_mean, cinic10_std),
                ]
            )
        if augmentation_strategy.preaugment and not os.path.exists(self.preaug_file):
            print(f"Preaugmenting and saving to {self.preaug_file}...")
            test_dataset = cinic_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,
                noise_file=self.noise_file,
                r=self.r,
                root_dir=self.root_dir,
                transform=self.__getattribute__(
                    augmentation_strategy.preaugment["transform"]
                ),
                mode="all",
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            all_augmented = {"samples": [], "labels": []}
            for i in range(augmentation_strategy.preaugment["ratio"] - 1):
                for img, target, index in test_loader:
                    for j in range(len(img)):
                        all_augmented["samples"].append(img[j].numpy())
                        all_augmented["labels"].append(target[j])
            torch.save(all_augmented, self.preaug_file)

    def run(self, mode, pred=[], prob=[]):
        if mode == "warmup":
            all_dataset = cinic_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,
                r=self.r,
                root_dir=self.root_dir,
                transform=self.transforms["warmup"],
                mode="all",
                noise_file=self.noise_file,
                preaug_file=self.preaug_file,
            )
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.warmup_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
            return trainloader

        elif mode == "train":
            labeled_dataset = cinic_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,
                r=self.r,
                root_dir=self.root_dir,
                transform=self.transforms["labeled"],
                mode="labeled",
                noise_file=self.noise_file,
                pred=pred,
                probability=prob,
                preaug_file=self.preaug_file,
            )
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )

            unlabeled_dataset = cinic_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,
                r=self.r,
                root_dir=self.root_dir,
                transform=self.transforms["unlabeled"],
                mode="unlabeled",
                noise_file=self.noise_file,
                pred=pred,
                preaug_file=self.preaug_file,
            )
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
            return labeled_trainloader, unlabeled_trainloader
        elif mode == "trainall":
            labeled_dataset = cinic_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,
                r=self.r,
                root_dir=self.root_dir,
                transform=self.transforms["labeled"],
                mode="labeled",
                noise_file=self.noise_file,
                pred=pred,
                probability=prob,
                preaug_file=self.preaug_file,
            )
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
            return labeled_trainloader


        elif mode == "test":
            test_dataset = cinic_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,
                r=self.r,
                root_dir=self.root_dir,
                transform=self.transforms["test"],
                mode="test",
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return test_loader

        elif mode == "eval_train":
            eval_dataset = cinic_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,
                r=self.r,
                root_dir=self.root_dir,
                transform=self.transforms["test"],
                mode="all",
                noise_file=self.noise_file,
                preaug_file=self.preaug_file,
            )
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return eval_loader
