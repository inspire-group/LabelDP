# LabelDP

Code for "Machine Learning with Differentially Private Labels: Mechanisms and Frameworks" in PoPETs 2022.

## Files
```shell
├── LabelDP
|   ├── SCAN					#unsupervised Learning based
|   |	   ├── simiclr.py                  
|   |	   ├── scan.py                  
|   |	   ├── selflabel.py                 
|   |	   ├── eval.py                   
|   |	   └── NoiseCluster.py			#add gaussian noise to clusters
|   ├── FixMatch				#semi-Supervised learning based
|   |	   ├── train_cifar.py                  
|   |	   └── train_cinic.py
|   ├── AugDescent				#learning with noise label based
|   |	   ├── train_cifar.py                  
|   |	   └── train_cinic.py
|   ├── mypath.py				#specify ckpt_path and data_pah in this file. 
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

## Datasets

- [CIFAR10/CIFAR100](http://www.cs.toronto.edu/~kriz/cifar.html)

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
- Generated RandRes and PATE label.

As mentioned above, you can also directly prepare the datasets including label files (dplabel folder under each dataset) by downloading the file we provide (Google drive [link](https://drive.google.com/drive/folders/1Z9o0ESF-2V_4WjMJ_LvKpuniNDn20FxM?usp=sharing)) and untar it as ```data_path```.

## Requirements
The code is tested with python 3.8.5 and PyTorch 1.11.0. The complete list of required packages are available in `requirement.txt`, and can be installed with `pip install -r requirement.txt`.


## Usage
- See ```SCAN/FixMatch/AugDescent``` for further instructions.

## EvalOnly
- We also provide evaluation of results in our paper (including generated labels and checkpoints). See ```EvalOnly``` for further instructions.