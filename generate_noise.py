import os
import argparse
import sys
import numpy as np
from torchvision import datasets
import math
from mypath import MyPath
sys.path.append("./dataset")
from cinic import CINIC10

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


def main():
    parser = argparse.ArgumentParser(description = 'Unsupervised LabelDP Evaluation')
    parser.add_argument('--dataset', default = 'cifar10', type = str,
                        choices = ['cifar10', 'cifar100', 'cinic10'], help = 'choice of dataset')
    parser.add_argument("--rseed", default= 42, type = int, help = 'random seed')


    args = parser.parse_args()


    db_root_path = MyPath.db_root_dir(args.dataset)
    if args.dataset == 'cifar10':
        trainset = datasets.CIFAR10(db_root_path, train=True)
        args.num_class = 10
        eps_list = [0.5, 1, 2, 3, 4]

    elif args.dataset == 'cifar100':
        trainset = datasets.CIFAR100(db_root_path, train=True)
        args.num_class = 100
        eps_list = [1, 2, 3, 4, 6]

    elif args.dataset == 'cinic10':
        trainset = CINIC10(os.path.join(db_root_path,'npy'), filen='train')
        args.num_class = 100
        eps_list = [0.5, 1, 2, 3, 4]

    clean_train_label = np.array(trainset.targets)

    if not os.path.exists(os.path.join(db_root_path, "labeldptest")):
        os.mkdir(os.path.join(db_root_path, "labeldptest"))

    for eps in eps_list:
        print("\n\nEpsilon", eps)
        print("Noise type: random response")
        noise_label = random_response(clean_train_label, eps, args.num_class, args.rseed)

        np.save(os.path.join(db_root_path, "labeldptest", "eps"+str(eps)+".npy"), noise_label)



if __name__ == "__main__":
    main()