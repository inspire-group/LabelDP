# EvalOnly
- We provide evaluation of results in our paper (including generated labels and checkpoints).


## Setup

- Speicify your ```ckpt_path``` and ```data_path``` in ```EvalOnly/mypath.py```
- Download checkpoints from this Google Drive [link](https://drive.google.com/drive/folders/15VewPtwAQclHZLZ0RLlCWVmiz-dQnclp?usp=sharing) and untar it as your ```ckpt_path``` (The checkpoints in ```ckpts_path/SCAN/cifar10``` and ```ckpts_path/SCAN/cifar20``` are from the official repo of [SCAN](https://github.com/wvangansbeke/Unsupervised-Classification#clustering)).

- Download datasets including dp labels from this Google Drive [link](https://drive.google.com/drive/folders/1Z9o0ESF-2V_4WjMJ_LvKpuniNDn20FxM?usp=sharing) and untar it as your ```data_path```.

The following commands assume the current folder is ```EvalOnly```.

## Table 2 

```
python evaldplabel.py --dataset cifar10
python evaldplabel.py --dataset cifar100
python evaldplabel.py --dataset cinic10
```

## Table 3 (CIFAR10)

#### Non-private baseline
```
cd AugDescent
python eval.py --dataset cifar10 --noisemode ndp --preset AugDesc-WS 
```

#### NoiseCluster
```
cd SCAN
python NoiseCluster.py --dataset cifar10
```
#### RandRes+DenoiseSSL
```
cd FixMatch
python eval.py --dataset cifar10 --seed 5 --noisemode randres --learningmode denoisessl --epsilon 1
```
```--epsilon``` options: 0.5|1|2|4.

#### RandRes+LDPSSL
```
cd FixMach
python eval.py --dataset cifar10  --seed 5 --noisemode randres --learningmode ldpssl --epsilon 1
```
```--epsilon``` options: 0.5|1|2|4.

#### RandRes+AugDescent
```
cd AugDescent
python eval.py --dataset cifar10 --noisemode randres --preset AugDesc-WS --epsilon 1
```
```--epsilon``` options: 0.5|1|2|4.

#### PATE+DenoiseSSL
```
cd FixMatch
python eval.py --dataset cifar10 --seed 5 --noisemode pate --learningmode denoisessl --epsilon 1
```
```--epsilon``` options: 0.5|1|2|4.

#### PATE+LDPSSL
```
cd FixMatch
python eval.py --dataset cifar10 --seed 5 --noisemode pate --learningmode ldpssl --epsilon 1
```
```--epsilon``` options: 0.5|1|2|4.

#### PATE+AugDescent
```
cd AugDescent
python eval.py --dataset cifar10 --noisemode pate --preset AugDesc-WS --epsilon 1
```
```--epsilon``` options: 0.5|1|2|4.

## Table 4 (CIFAR100)
Change all --dataset cifar10 in Table 3(CIFAR10) to --dataset cifar100

```--epsilon``` options: 1|2|4|6.

## Table 5 (CINIC10)
Change all --dataset cifar10 in Table 3(CIFAR10) to --dataset cinic10

```--epsilon``` options: 0.5|1|2|4.

## Table 6

$\varepsilon=0.5|1|2|4$ for CIFAR10/CINIC10  and  $\varepsilon=1|2|4|6 $ for CIFAR100 are from Table3/4/5.

$\varepsilon=3$ for CIFAR10:
```
cd AugDescent
python eval.py --dataset cifar10 --noisemode randres --preset AugDesc-WS --epsilon 3
```

$\varepsilon=3$ for CIFAR100:
```
cd AugDescent
python eval.py --dataset cifar100 --noisemode pate --preset AugDesc-WS --epsilon 3
```

$\varepsilon=3$ for CINIC10:
```
cd AugDescent
python eval.py --dataset cinic10 --noisemode randres --preset AugDesc-WS --epsilon 3
```
