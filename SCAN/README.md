# NoiseCluster (SCAN: Learning to Classify Images without Labels)


This repo is built based on [source repo](https://github.com/wvangansbeke/Unsupervised-Classification), which contains the official Pytorch implementation of paper:
> [**SCAN: Learning to Classify Images without Labels**](https://arxiv.org/pdf/2005.12320)  ECCV 2020
>
> Wouter Van Gansbeke, Simon Vandenhende, Stamatios Georgoulis, Marc Proesmans, Luc Van Gool.

For more details, please check the source repo.

Please follow the instructions underneath to perform semantic clustering with SCAN.

## Training

### Train model
The configuration files can be found in the `configs/` directory. The training procedure consists of the following steps:
- __STEP 1__: Solve the pretext task i.e. `simclr.py`
- __STEP 2__: Perform the clustering step i.e. `scan.py`
- __STEP 3__: Perform the self-labeling step i.e. `selflabel.py`

For example, run the following commands sequentially to perform our method on CIFAR10:
```shell
python simclr.py --config_exp configs/pretext/simclr_cifar10.yml
python scan.py --config_exp configs/scan/scan_cifar10.yml
python selflabel.py --config_exp configs/selflabel/selflabel_cifar10.yml
```

To train on CIFAR100/CINIC10, the above \*_cifar10.yml needs to be changed to \*_cifar100.yml/*_cinic10.yml

 
### Evaluation
Running `eval.py` script will generate clusters for train set or test set . For example, the model on cifar-10 can be evaluated as follows:
```shell
python eval.py --config_exp configs/scan/scan_cifar10.yml --model $MODEL_PATH 
```

#### Note
- Code line 43-45 and 101 in eval.py are for generating cluster result for test set.
- Code line 46-47 and 103 in eval.py  are for generating cluster result for train set.
- This needs manually commenting and uncommenting eval.py accordingly.

To evaluate NoiseCluster on CIFAR10:

```shell
python NoiseCluster.py --dataset cifar10
```
To train on CIFAR100/CINIC10, ```--dataset cifar10``` needs to be changed to  ```--dataset cifar100``` or  ```--dataset cinic10```

See ```FixMatch``` for the use of clusters in denoisessl. 


## Citation

```bibtex
@inproceedings{vangansbeke2020scan,
  title={Scan: Learning to classify images without labels},
  author={Van Gansbeke, Wouter and Vandenhende, Simon and Georgoulis, Stamatios and Proesmans, Marc and Van Gool, Luc},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2020}
}
```
## License

This software is released under a creative commons license which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary [here](http://creativecommons.org/licenses/by-nc/4.0/).
