# DenoiseSSL|LDPSSL (FixMatch)
This repo is built based on [source repo](https://github.com/kekmodel/FixMatch-pytorch), which is an unofficial PyTorch implementation of paper:

> [**FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence**](https://arxiv.org/abs/2001.07685).  NeurIPS 2020
>
> Kihyuk Sohn\*, David Berthelot\*, Chun-Liang Li, Zizhao, Zhang, Nicholas Carlini, Ekin D. Cubuk, Alex Kurakin, Han Zhang, Colin Raffel [`*`: equal contribution].


The official Tensorflow implementation of FixMatch is [here](https://github.com/google-research/fixmatch).

For more details, please check the source repo.

Before training, you should prepare the noise label file. either by randres or pate. As mentioned in ```main/README.md```, we also provide a copy of the noise file we used:
- You can download datasets including dp labels from this Google Drive [link](https://drive.google.com/drive/folders/1Z9o0ESF-2V_4WjMJ_LvKpuniNDn20FxM?usp=sharing) and untar it as your ```data_path```.

To run ```denoisessl``` learning mode, you need to prepare ```train_cluster_pred.npy``` by running code in ```SCAN```  for the corresponding dataset and save to path: ```ckpt_path/datasetname/SCAN/archname/selflabel/train_cluster_pred.npy```

```datasetname```options:cifar10|cifar20|cinic10 (cifar20 is for cifar100 dataset). ```archname```options:resnet18
- You can also get train_cluster_pred.npy by downloading from this Google Drive [link](https://drive.google.com/drive/folders/15VewPtwAQclHZLZ0RLlCWVmiz-dQnclp?usp=sharing) and untar it as your ```ckpt_path```(The checkpoints in ```ckpts_path/SCAN/cifar10``` and ```ckpts_path/SCAN/cifar20``` are from the official repo of [SCAN](https://github.com/wvangansbeke/Unsupervised-Classification#clustering)).

## DenoiseSSL on CIFAR10
```
python train.py --dataset cifar10 --batch-size 64 --arch resnet18 --lr 0.03 --expand-labels --noisemode randres --learningmode denoisessl --no-progress --epsilon 4 --seed 5
```
```--epsilon```options: 0.5|1|2|4
```--noisemode```options: randres|pate


## LDPSSL on CIFAR10
```
python train.py --dataset cifar10 --batch-size 64 --arch resnet18 --lr 0.03 --expand-labels  --noisemode randres --learningmode ldpssl --no-progress --epsilon 4 --seed 5
```
```--epsilon```options: 0.5|1|2|4
```--noisemode```options: randres|pate

## DenoiseSSL on CIFAR100
```
python train.py --dataset cifar100 --batch-size 64 --arch resnet18 --lr 0.03  --wdecay 0.001 --expand-labels --noisemode randres --learningmode denoisessl --no-progress --epsilon 4 --seed 5
```
```--epsilon```options: 0.5|1|2|4
```--noisemode```options: randres|pate


## LDPSSL on CIFAR100
```
python train.py --dataset cifar100 --batch-size 64 --arch resnet18 --lr 0.03  --wdecay 0.001 --expand-labels  --noisemode randres --learningmode ldpssl --no-progress --epsilon 4 --seed 5
```
```--epsilon```options: 0.5|1|2|4
```--noisemode```options: randres|pate


## DenoiseSSL on CINIC10
```
python train.py --dataset cinic10 --batch-size 64 --arch resnet18 --lr 0.03 --expand-labels --noisemode randres --learningmode denoisessl --no-progress --epsilon 4 --seed 5
```
```--epsilon```options: 0.5|1|2|4
```--noisemode```options: randres|pate


## LDPSSL on CINIC10
```
python train.py --dataset cinic10 --batch-size 64 --arch resnet18 --lr 0.03 --expand-labels --noisemode randres --learningmode denoisessl --no-progress --epsilon 4 --seed 5
```
```--epsilon```options: 0.5|1|2|4
```--noisemode```options: randres|pate


## Citations
```
@article{sohn2020fixmatch,
    title={FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence},
    author={Kihyuk Sohn and David Berthelot and Chun-Liang Li and Zizhao Zhang and Nicholas Carlini and Ekin D. Cubuk and Alex Kurakin and Han Zhang and Colin Raffel},
    journal={arXiv preprint arXiv:2001.07685},
    year={2020},
}
```
