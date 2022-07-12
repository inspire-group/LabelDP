# LabelDP

Code for "Machine Learning with Differentially Private Labels: Mechanisms and Frameworks" in PoPETs 2022.

## Files
```shell
├── LabelDP
|   ├── SCAN					#unsupervised learning based
|   |	   ├── simiclr.py                  
|   |	   ├── scan.py                  
|   |	   ├── selflabel.py                 
|   |	   ├── eval.py
|   |	   ├── agm.py                           #analytical calibration gaussian mechanism
|   |	   └── NoiseCluster.py			#add gaussian noise to clusters
|   ├── FixMatch				#semi-supervised learning based       
|   |	   └── train.py
|   ├── AugDescent				#learning with noise label based
|   |	   ├── train_cifar.py                  
|   |	   └── train_cinic.py
|   ├── PATEFM				        #pipeline 3
|   |	   ├── accoutant.py
|   |	   ├── train_teacher.py
|   |	   ├── teacher_vote.py
|   |	   ├── teacher_vote_add_noise.py
|   |	   ├── train_student.py
|   |	   └── eval_student.py
|   ├── mypath.py				#specify ckpt_path and data_path in this file. 
|   ├── generate_noise.py			#generate randres noise
|   ├── requirement.txt				#required package
|   ├── cinic10_conver_img2npy.py		#convert cinic10 imgs to npy.     
|   └── EvalOnly  
|	   ├── SCAN                  
|	   ├── FixMatch
|	   ├── AugDescent
|	   ├── mypath.py
|	   └── evaldplabel.py
├── ckpt_path                  
|   ├── cifar10                  
|   ├── cifar100                
|   └── cinic10                 
└── data_path                  
    ├── cifar10
    |	   ├── cifar-10-batches-py                        
    |	   └── dplabel            
    |		   ├── pate                
    |		   └── rr            
    ├── cifar100       
    |	   ├── cifar-100-python
    |	   └── dplabel            
    |		   ├── pate                
    |		   └── rr                       
    └── cinic10                 
	   ├── train   
	   ├── test   
	   ├── valid   	   	   
	   ├── npy     
	   └── dplabel 
	       	   ├── pate                
    		   └── rr                                     

```
Speicify your ```ckpt_path``` and ```data_path``` in ```LabelDP/mypath.py```

## Our Contributions
We leverage advancements in machine learning including unsupervised learning ([SCAN source repo](https://github.com/wvangansbeke/Unsupervised-Classification)) and semi-supervised learning ([FixMatch source repo](https://github.com/kekmodel/FixMatch-pytorch)), learning with noisy labels ([Aug-Descent source repo](https://github.com/KentoNishi/Augmentation-for-LNL)) to improve utility for machine learning models under label differential privacy. Specifically we propose [NoiseCluster](./SCAN/NoiseCluster.py) and [DenoiseSSL](./FixMatch/dataset/cifar.py#L32-L48) to improve the utility.

## Datasets

- [CIFAR10/CIFAR100](http://www.cs.toronto.edu/~kriz/cifar.html)

  - CIFAR10/CIFAR100 will be automatically downloaded when generated differentially private labels by [Randomized Response](#prepare-noise-file). Other scripts will assume CIFAR10/CIFAR100 to be in the expected folder (see [Files](#Files))without specifying ```download=True```.

- [CINIC10](https://datashare.ed.ac.uk/handle/10283/3192)
  - download and save original dataset split ```train/valid/test``` to ```data_path/cinic10```, then run `python cinic10_convert_img2npy.py`  will save npy format of cinic10 dataset in ```data_path/cinic10/npy```

- You can also directly prepare the datasets by downloading the file we provide (Google drive [link](https://drive.google.com/drive/folders/1Z9o0ESF-2V_4WjMJ_LvKpuniNDn20FxM?usp=sharing)) and untar it as ```data_path```.

## Prepare noise file:
- RandRes
```
python generate_noise.py --dataset cifar10
python generate_noise.py --dataset cifar100
python generate_noise.py --dataset cinic10
```

- PATE-FM

Please see [```PATEFM```](./PATEFM#evaluate-the-performance-of-final-student-model-pipeline-3)
- Generated RandRes and PATE label.

As mentioned above, you can also directly prepare the datasets including label files (dplabel folder under each dataset) by downloading the file we provide (Google drive [link](https://drive.google.com/drive/folders/1Z9o0ESF-2V_4WjMJ_LvKpuniNDn20FxM?usp=sharing)) and untar it as ```data_path```.

## Requirements
The code is tested with Python 3.8.5 and PyTorch 1.11.0. The complete list of required packages are available in `requirement.txt`, and can be installed with `pip install -r requirement.txt`.


## Usage
- See [```SCAN```](./SCAN)/[```FixMatch```](./FixMatch)/[```AugDescent```](./AugDescent) for further instructions.

## EvalOnly
- We also provide evaluation of results in our paper (including generated labels and checkpoints). See [```EvalOnly```](./EvalOnly) for further instructions.