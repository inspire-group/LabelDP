"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import torch
import yaml
import numpy as np
import pickle
import json
import sys
import os
sys.path.append("./../")
from utils.common_config import get_val_dataset, get_val_transformations, get_val_dataloader,\
                                get_model
from utils.evaluate_utils import get_predictions, hungarian_evaluate
from utils.memory import MemoryBank 
from utils.utils import fill_memory_bank
from PIL import Image
from scipy.optimize import linear_sum_assignment
from mypath import MyPath

FLAGS = argparse.ArgumentParser(description='Evaluate models from the model zoo')
FLAGS.add_argument('--dataset', type=str, default='cifar10', help='Data')
args = FLAGS.parse_args()


def random_response(clean_label, eps, num_class, rseed):
    noise_label = clean_label.copy()
    p1 = np.exp(eps)/(np.exp(eps)+num_class-1)
    p2 = 1 /(np.exp(eps)+num_class-1)
    np.random.seed(rseed)
    for i in range(len(noise_label)):
        rnd = np.random.random()
        if rnd < p1:
            continue
        else:
            candidates = np.arange(num_class)
            idx = int((rnd-p1)/p2)
            candidates = np.delete(candidates, clean_label[i])
            noise_label[i] = candidates[idx]
    print("Eps", eps, "p1", p1, "acc", np.mean(noise_label==clean_label))

    return noise_label

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def read_cifar100(path):
    tmp = unpickle(os.path.join(path, 'train'))
    X = np.zeros((tmp[b'data'].shape[0], 3, 32, 32))
    X[:,0,:,:] = np.array(tmp[b'data'])[:,:1024].reshape((-1, 32,32))
    X[:,1,:,:] = np.array(tmp[b'data'])[:,1024:1024*2].reshape((-1, 32,32))
    X[:,2,:,:] = np.array(tmp[b'data'])[:,1024*2:].reshape((-1, 32,32))
    Y = np.array(tmp[b'fine_labels'])
    X = np.transpose(X, (0,2,3,1))

    tmp = unpickle(os.path.join(path, 'test'))
    test_X = np.zeros((tmp[b'data'].shape[0], 3, 32, 32))
    test_X[:,0,:,:] = np.array(tmp[b'data'])[:,:1024].reshape((-1, 32,32))
    test_X[:,1,:,:] = np.array(tmp[b'data'])[:,1024:1024*2].reshape((-1, 32,32))
    test_X[:,2,:,:] = np.array(tmp[b'data'])[:,1024*2:].reshape((-1, 32,32))
    test_Y = np.array(tmp[b'fine_labels'])
    test_X = np.transpose(test_X, (0,2,3,1))

    return X, Y.astype(np.int32), test_X, test_Y.astype(np.int32)

def read_cifar10(path):
    data_list = ['data_batch_1','data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    cnt = np.zeros(6)
    for i in range(5):
        tmp = unpickle(os.path.join(path, data_list[i]))
        cnt[i+1] = cnt[i] + len(tmp[b'labels'])
    XX = np.zeros((int(cnt[5]), 1024*3))
    Y = np.zeros(int(cnt[5]))
    for i in range(5):
        tmp = unpickle(os.path.join(path, data_list[i]))
        XX[int(cnt[i]):int(cnt[i+1]), :] = np.array(tmp[b'data'])#.append(tmp[b'data'])
        Y[int(cnt[i]): int(cnt[i+1])] = np.array(tmp[b'labels'])

    X = np.zeros((XX.shape[0], 3, 32, 32))
    X[:,0,:,:] = XX[:,:1024].reshape((-1, 32, 32))
    X[:,1,:,:] = XX[:,1024:1024*2].reshape((-1, 32, 32))
    X[:,2,:,:] = XX[:,1024*2:].reshape((-1, 32, 32))
    X = np.transpose(X, (0,2,3,1))
    tmp = unpickle(os.path.join(path, 'test_batch'))
    test_X = np.zeros((tmp[b'data'].shape[0], 3, 32, 32))
    test_X[:,0,:,:] = np.array(tmp[b'data'])[:,:1024].reshape((-1, 32,32))
    test_X[:,1,:,:] = np.array(tmp[b'data'])[:,1024:1024*2].reshape((-1, 32,32))
    test_X[:,2,:,:] = np.array(tmp[b'data'])[:,1024*2:].reshape((-1, 32,32))
    test_Y = (np.array(tmp[b'labels']))
    test_X = np.transpose(test_X, (0,2,3,1))
    return X, Y.astype(np.int32), test_X, test_Y.astype(np.int32)

def _hungarian_match(flat_preds, flat_targets, preds_k = 10):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res

def majority_voting(preds, targets, num_k = 10):

    res = []
    for i in range(num_k):
        targets_c = targets[preds==i]
        cnt = np.zeros(num_k)
        
        for j in range(num_k):
            cnt[j] = np.sum(targets_c==j)
        res.append((i, np.argmax(cnt)))
    return res

def gaussian_noise_max_voting(preds, targets, sigma, num_k = 10):
    res = []
    for i in range(num_k):
        targets_c = targets[preds==i]
        cnt = np.zeros(num_k)
        
        for j in range(num_k):
            cnt[j] = np.sum(targets_c==j)
        gaussian_noise = np.random.normal(0, sigma, num_k)
        res.append((i, np.argmax(cnt+gaussian_noise)))
    return res

def laplace_noise_max_voting(preds, targets, eps, sst =2, num_k = 10):
    res = []
    for i in range(num_k):
        targets_c = targets[preds==i]
        cnt = np.zeros(num_k)
        
        for j in range(num_k):
            cnt[j] = np.sum(targets_c==j)
        laplace_noise = np.random.laplace(scale = sst/eps, size=num_k)
        res.append((i, np.argmax(cnt+laplace_noise)))
    return res

def label_match(raw_cluster_label, metric, N):
    cluster_label = raw_cluster_label.copy()
    for i in range(N):
        cluster_label[raw_cluster_label==metric[i][0]] = metric[i][1]
    return cluster_label


def main():
    # Read label file
    if args.dataset == 'cifar10':
        num_k = 10
        _, train_label, _, test_label = read_cifar10(os.path.join(MyPath.db_root_dir(args.dataset), 'cifar-10-batches-py'))
        train_cluster_label = np.load(os.path.join(MyPath.ckpt_root_dir(), 'SCAN', args.dataset, 'resnet18', 'selflabel', 'train_cluster_pred.npy'))
        test_cluster_label = np.load(os.path.join(MyPath.ckpt_root_dir(), 'SCAN', args.dataset, 'resnet18', 'selflabel', 'test_cluster_pred.npy'))
    elif args.dataset == 'cifar100':
        num_k = 100
        _, train_label, _, test_label = read_cifar100(os.path.join(MyPath.db_root_dir(args.dataset),'cifar-100-python'))
        train_cluster_label = np.load(os.path.join(MyPath.ckpt_root_dir(), 'SCAN', args.dataset, 'resnet18', 'selflabel', 'train_cluster_pred.npy'))
        test_cluster_label = np.load(os.path.join(MyPath.ckpt_root_dir(), 'SCAN', args.dataset, 'resnet18', 'selflabel', 'test_cluster_pred.npy'))
    elif args.dataset == 'cinic10':
        num_k = 10
        train_label = np.load(os.path.join(MyPath.db_root_dir(args.dataset), 'npy', 'train_label.npy'))
        #valid_label = np.load(os.path.join(MyPath.db_root_dir(args.dataset), 'npy', 'valid_label.npy'))
        test_label = np.load(os.path.join(MyPath.db_root_dir(args.dataset), 'npy', 'test_label.npy'))
        train_cluster_label = np.load(os.path.join(MyPath.ckpt_root_dir(), 'SCAN', args.dataset, 'resnet18', 'selflabel', 'train_cluster_pred.npy'))
        #val_cluster_label = np.load(os.path.join(MyPath.ckpt_root_dir(), 'SCAN', args.dataset, 'resnet18', 'selflabel', 'valid_cluster_pred.npy'))
        test_cluster_label = np.load(os.path.join(MyPath.ckpt_root_dir(), 'SCAN', args.dataset, 'resnet18', 'selflabel', 'test_cluster_pred.npy'))

    print("Clean label")
    match_eval = _hungarian_match(train_cluster_label, train_label, num_k)
    train_convert_label = label_match(train_cluster_label, match_eval, num_k)
    test_convert_label = label_match(test_cluster_label, match_eval, num_k)
    print("Hungarian Match Acc: trainset: %.8f, testset: %.8f"%(np.mean(train_convert_label==train_label), np.mean(test_convert_label==test_label)))

    match_eval = majority_voting(train_cluster_label, train_label, num_k)
    train_convert_label = label_match(train_cluster_label, match_eval, num_k)
    test_convert_label = label_match(test_cluster_label, match_eval, num_k)
    print("Majority Voting Acc: trainset: %.8f, testset: %.8f"%(np.mean(train_convert_label==train_label), np.mean(test_convert_label==test_label)))


    print("\n\nGenerated noisy label")
    if args.dataset == 'cifar10':
        eps_list = [0.006,0.007,0.5,1,2,4]

    elif args.dataset=='cifar100':
        eps_list = [0.4,0.5,1,2,4,6]
    else:
        eps_list = [0.02,0.03,0.5,1,2,4]


    for eps in eps_list:
        rseed = 42#np.random.seed(42)
        if eps == int(eps):
            eps = int(eps)
        noise_label = random_response(train_label, eps, num_k, rseed)
        p1 = np.exp(eps)/(np.exp(eps)+num_k-1)
        p2 = 1/(np.exp(eps)+num_k-1)
        print("epsilon:", eps, "label acc:", np.mean(noise_label==train_label))

        sigma = 2*np.sqrt(np.log(1.25*1e5))/eps
        match_eval = gaussian_noise_max_voting(train_cluster_label, train_label, sigma, num_k)
        train_convert_label = label_match(train_cluster_label, match_eval, num_k)
        test_convert_label = label_match(test_cluster_label, match_eval, num_k)
        print("Fixed Gaussian noise majority voting acc: trainset: %.8f, testset: %.8f\n"%(np.mean(train_convert_label==train_label), np.mean(test_convert_label==test_label)))

if __name__ == "__main__":
    main() 
