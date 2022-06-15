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

from mypath import MyPath

FLAGS = argparse.ArgumentParser(description='Evaluate models from the model zoo')
FLAGS.add_argument('--dataset', type=str, default='cifar10', help='Data')
args = FLAGS.parse_args()

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

def main():
    # Read label file
    if args.dataset == 'cifar10':
        _, train_label, _, _ = read_cifar10(os.path.join(MyPath.db_root_dir(args.dataset), 'cifar-10-batches-py'))
    elif args.dataset == 'cifar100':
        _, train_label, _, _ = read_cifar100(os.path.join(MyPath.db_root_dir(args.dataset),'cifar-100-python'))
    elif args.dataset == 'cinic10':
        train_label = np.load(os.path.join(MyPath.db_root_dir(args.dataset), 'npy', 'train_label.npy'))

    print(args.dataset)
    if args.dataset == 'cifar10':
        eps_list = [0.5, 1, 2, 4]

    elif args.dataset=='cifar100':
        eps_list = [1, 2, 4, 6]
    else:
        eps_list = [0.5, 1, 2, 4]
    db_root_dir = MyPath.db_root_dir(args.dataset)
    
    print("RandRes")
    for eps in eps_list:
        if eps == int(eps):
            eps = int(eps)
        noise_file_path = os.path.join(db_root_dir, 'dplabel', 'rr', 'eps'+str(eps)+'.npy')
        noise_label = np.load(noise_file_path)

        print("Acc: trainset: %.8f."%(np.mean(noise_label==train_label)))

    print("\n\nPATE")
    for eps in eps_list:
        if eps == int(eps):
            eps = int(eps)
        noise_file_path = os.path.join(db_root_dir, 'dplabel', 'pate','eps_'+str(eps))
        noise_label = torch.load(noise_file_path)
        print("Acc: trainset: %.8f."%(np.mean(noise_label==train_label)))


if __name__ == "__main__":
    main() 
