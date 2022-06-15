'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
'''
import logging

import torch

logger = logging.getLogger(__name__)

__all__ = ['get_mean_and_std', 'accuracy', 'AverageMeter']

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    logger.info('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


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




class cifar100_to_cifar20(object):
    @staticmethod
    def _convert(target):
        _dict = \
        {0: 4,
        1: 1,
        2: 14,
        3: 8,
        4: 0,
        5: 6,
        6: 7,
        7: 7,
        8: 18,
        9: 3,
        10: 3,
        11: 14,
        12: 9,
        13: 18,
        14: 7,
        15: 11,
        16: 3,
        17: 9,
        18: 7,
        19: 11,
        20: 6,
        21: 11,
        22: 5,
        23: 10,
        24: 7,
        25: 6,
        26: 13,
        27: 15,
        28: 3,
        29: 15,
        30: 0,
        31: 11,
        32: 1,
        33: 10,
        34: 12,
        35: 14,
        36: 16,
        37: 9,
        38: 11,
        39: 5,
        40: 5,
        41: 19,
        42: 8,
        43: 8,
        44: 15,
        45: 13,
        46: 14,
        47: 17,
        48: 18,
        49: 10,
        50: 16,
        51: 4,
        52: 17,
        53: 4,
        54: 2,
        55: 0,
        56: 17,
        57: 4,
        58: 18,
        59: 17,
        60: 10,
        61: 3,
        62: 2,
        63: 12,
        64: 12,
        65: 16,
        66: 12,
        67: 1,
        68: 9,
        69: 19,
        70: 2,
        71: 10,
        72: 0,
        73: 1,
        74: 16,
        75: 12,
        76: 9,
        77: 13,
        78: 15,
        79: 13,
        80: 16,
        81: 19,
        82: 2,
        83: 4,
        84: 6,
        85: 19,
        86: 5,
        87: 5,
        88: 8,
        89: 19,
        90: 18,
        91: 1,
        92: 2,
        93: 15,
        94: 6,
        95: 0,
        96: 17,
        97: 8,
        98: 14,
        99: 13}

        return _dict[target]