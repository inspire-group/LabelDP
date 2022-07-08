# PATEFM

Train the student model in PATEFM under label differential privacy and get the differentially private labels on training set.

This repo is built based on [FixMatch source repo](https://github.com/kekmodel/FixMatch-pytorch) and [accountant PATE-FM source repo](https://github.com/facebookresearch/label_dp_antipodes/blob/main/lib/pate/accountant.py) (which is based on [PATE source repo](https://github.com/tensorflow/privacy/tree/master/research/pate_2018/ICLR2018))

### Train teacher model
```
python train_teacher.py --dataset cifar10 --batch-size 64 --arch resnet18 --lr 0.03 --expand-labels --no-progress --seed 5 --num-teachers 500 --teacher_id 0
```
You may need to assign ```--num-teachers``` and ```--teacher_id``` (from 0 to num-teachers-1).

### Get raw teacher vote 
```
python teacher_vote.py --dataset cifar10  --batch-size 256 --num-teachers 500
```

### Get noisy label vote under desired label privacy
```
python teacher_vote_add_noise.py --dataset cifar10 --num-teachers 500 --n_threshold 300 --sigma1 500 --sigma2 300 --epsilon 4
```
### Train student model (under desired label privacy)
```
python train_student.py --dataset cifar10 --batch-size 64 --arch resnet18 --lr 0.03 --expand-labels --no-progress --seed 5 --num-teachers 500 --epsilon 4 
```
### Evaluate the performance of final student model (Pipeline 3)
```
python eval_student.py --dataset cifar10 --batch-size 256
```
You can uncomment lines 224-228 in eval_student.py to generate differentially private labels on the whole set (For pipeline 2).
