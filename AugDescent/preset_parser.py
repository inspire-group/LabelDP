import argparse
import json
import os
from mypath import MyPath

def parse_args(file):
    parser = argparse.ArgumentParser(description="Setting for AugDescent")
    parser.add_argument("--preset", required=True, type=str)
    parser.add_argument('--epsilon', type=float, default=3,
                        help='random respond epsilon')
    parser.add_argument('--noisemode', type = str, 
                        choices=['sym', 'asym', 'randres', 'pate', 'ndp'],
                        help='noise type')
    parser.add_argument('--dataset', type = str, 
                        choices=['cifar10', 'cifar100', 'cinic10'],
                        help='data set')
    parser.add_argument("--arch", type = str, choices = ['resnet18', 'wideresnet28', 'vgg'], default = 'resnet18')
    
    cmdline_args = parser.parse_args()
    if cmdline_args.epsilon == int(cmdline_args.epsilon):
        cmdline_args.epsilon = int(cmdline_args.epsilon)

    with open(file, "r") as f:
        jsonFile = json.load(f)

    class dotdict(dict):
        def __getattr__(self, name):
            return self[name]

        def __setattr__(self, name, value):
            self[name] = value

    args = dotdict()
    args.update(jsonFile)

    args.dataset = cmdline_args.dataset
    args.preset = cmdline_args.preset
    subpresets = cmdline_args.preset.split(".")
    if cmdline_args.noisemode == 'ndp':
        new_subpresets = [cmdline_args.noisemode, "0sym",subpresets[0]]        
    else:
        new_subpresets = [cmdline_args.noisemode, str(cmdline_args.epsilon),subpresets[0]]

    if "configs" in args:
        del args["configs"]
        jsonFile = jsonFile["configs"]

    db_root_dir = MyPath.db_root_dir(args.dataset)
    args.data_path = db_root_dir
    if cmdline_args.noisemode == 'pate':
        args.noise_file_path = os.path.join(db_root_dir, 'dplabel', 'pate','eps_'+str(cmdline_args.epsilon))    
    elif cmdline_args.noisemode == 'randres':
        args.noise_file_path = os.path.join(db_root_dir, 'dplabel', 'rr', 'eps'+str(cmdline_args.epsilon)+'.npy')
    elif cmdline_args.noisemode == 'ndp':
        args.noise_file_path = ""
    else:
        args.noise_file_path = os.path.join(db_root_dir, 'dplabel', cmdline_args.noisemode)        

    ckpt_root_dir = os.path.join(MyPath.ckpt_root_dir(), 'AugDescent', args.dataset)

    for subp in new_subpresets:
        if not (subp in jsonFile):
            continue
        jsonFile = jsonFile[subp]
        args.update(jsonFile)
        if "configs" in args:
            del args["configs"]
        if "configs" in jsonFile:
            jsonFile = jsonFile["configs"]
    args.checkpoint_path = os.path.join(ckpt_root_dir, cmdline_args.arch, new_subpresets[0], new_subpresets[1],  new_subpresets[2])
    
    args.pretrained_path = args.checkpoint_path

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    saved_folder = os.path.join(args.checkpoint_path, "saved")
    if not os.path.exists(saved_folder):
        os.mkdir(saved_folder)
    best_folder = os.path.join(args.checkpoint_path, "best")
    if not os.path.exists(best_folder):
        os.mkdir(best_folder)
    if not os.path.exists(args.pretrained_path + f"/saved/{args.preset}.pth.tar"):
        # if os.path.exists(args.pretrained_path + f"/saved/metrics.log"):
        #     raise AssertionError("Training log already exists!")
        args.pretrained_path = ""
    if cmdline_args.noisemode == "randres" or cmdline_args.noisemode == "pate":
        args.r = cmdline_args.epsilon
    if args.r == int(args.r):
        args.r = int(args.r)
    with open(args.checkpoint_path + "/saved/info.json", "w") as f:
        json.dump(args, f, indent=4, sort_keys=True)
    
    
    print(args)
    return args
