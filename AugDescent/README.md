# Augmentation-for-LNL

This repo is built based on [source repo](https://github.com/KentoNishi/Augmentation-for-LNL), which contains the code of paper: 

> [**Augmentation Strategies for Learning with Noisy Labels**](https://arxiv.org/pdf/2103.02130).  CVPR 2021
>
>Kento Nishi\*, Yi Ding*, Alex Rich, Tobias Höllerer [`*`: equal contribution].

For more details, please check the source repo.

Before training, you should prepare the noise label file. either by randres or pate. As mentioned in ```main/README.md```, we also provide a copy of the noise file we used:
- You can download datasets including dp labels from this Google Drive [link](https://drive.google.com/drive/folders/1Z9o0ESF-2V_4WjMJ_LvKpuniNDn20FxM?usp=sharing) and untar it as your ```data_path```.

## CIFAR10
```
python train_cifar.py --dataset cifar10 --noisemode pate --preset AugDesc-WS --epsilon 4
```
```--epslion``` options: 0.5|1|2|4 ```--noisemode``` options: randres|pate

## CIFAR100

```
python train_cifar.py --dataset cifar100 --noisemode pate --preset AugDesc-WS --epsilon 4
```
```--epslion``` options: 1|2|4|6 ```--noisemode``` options: randres|pate

## CINIC10
```
python train_cinic.py --dataset cinic10 --noisemode pate --preset AugDesc-WS --epsilon 4
```
```--epsilon``` options: 0.5|1|2|4 ```--noisemode``` options: randres|pate


## Citations
```
@InProceedings{Nishi_2021_CVPR,
    author    = {Nishi, Kento and Ding, Yi and Rich, Alex and {H{\"o}llerer, Tobias},
    title     = {Augmentation Strategies for Learning With Noisy Labels},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {8022-8031}
}
```

