from warnings import filterwarnings

filterwarnings("ignore")

import os
import random
import sys
import argparse
import numpy as np

import sys
sys.path.append("./../")
from mypath import MyPath

from accountant import run_analysis
from torch.distributions import normal

def noisy_threshold_labels(votes, threshold, selection_noise_scale, result_noise_scale):
    """Imported from https://github.com/facebookresearch/label_dp_antipodes/blob/main/lib/pate/utils.py#L139-L153"""
    ####modified for numpy.arrays
    def noise(scale, shape):
        if scale == 0:
            return 0

        return np.random.normal(loc=0, scale=scale, size=(shape))

    noisy_votes = votes + noise(selection_noise_scale, votes.shape)

    over_t_mask = np.max(noisy_votes, axis=1) > threshold
    over_t_labels = (
        votes[over_t_mask] + noise(result_noise_scale, votes[over_t_mask].shape)
    ).argmax(axis=1)

    return over_t_labels, over_t_mask


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cinic10', 'cifar10', 'cifar100'],
                        help='dataset name')
    parser.add_argument('--arch', default='resnet18', type=str,
                        choices=['wideresnet', 'vgg', 'resnet18'],
                        help='dataset name')
    parser.add_argument('--sigma1', default=500, type=float,
                        help='sigma1 in confgnmax')
    parser.add_argument('--sigma2', default=300, type=float,
                        help='sigma2 in confgnmax')
    parser.add_argument('--n_threshold', default=300, type=float,
                        help='threshold in confgnmax')    
    parser.add_argument('--epsilon', default=4, type=float,
                        help='epsilon for differential privacy')
    parser.add_argument('--num-teachers', type=int, default=500,
                        help='number of teacher models')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")

    args= parser.parse_args()
    if args.epsilon == int(args.epsilon):
        args.epsilon = int(args.epsilon)

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


    pred_count = np.load(os.path.join(MyPath.ckpt_root_dir(), "PATEFM", args.dataset, str(args.num_teachers)+'teachers', "aggregate_votes.npy"))
    
    new_inds = np.arange(args.len_dataset)
    np.random.shuffle(new_inds)
    pred_count = pred_count[new_inds]

    eps_total, partition, answered, order_opt = run_analysis(
        pred_count,
        "gnmax_conf",
        args.sigma2,
        {
            "sigma1": args.sigma1,
            "t": args.n_threshold,
        },
    )


    pred_count_label, sigma1_mask = noisy_threshold_labels(pred_count, args.n_threshold, args.sigma1, args.sigma2)

    select_ind_end = (np.sum(eps_total<=args.epsilon)).astype(np.int32)
    assert (select_ind_end>0), "Exceed privacy budget!"

    sigma1_mask_select_ind = sigma1_mask[:select_ind_end]

    new_inds_select = new_inds[:select_ind_end]
    new_inds_select = new_inds_select[sigma1_mask_select_ind==1]

    pred_count_label = pred_count_label[:select_ind_end]
    pred_count_label = pred_count_label[sigma1_mask_select_ind==1]



    if not os.path.exists(os.path.join(MyPath.ckpt_root_dir(), "PATEFM", args.dataset, str(args.num_teachers)+'teachers', "eps"+str(args.epsilon))):
        os.makedirs(os.path.join(MyPath.ckpt_root_dir(), "PATEFM", args.dataset, str(args.num_teachers)+'teachers', "eps"+str(args.epsilon)))

    np.save(os.path.join(MyPath.ckpt_root_dir(), "PATEFM", args.dataset, str(args.num_teachers)+'teachers', "eps"+str(args.epsilon), "noise.npy"), pred_count_label)
    np.save(os.path.join(MyPath.ckpt_root_dir(), "PATEFM", args.dataset, str(args.num_teachers)+'teachers', "eps"+str(args.epsilon), "queried_inds.npy"), new_inds_select)
