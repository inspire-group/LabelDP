import torchvision
from PIL import Image
import os
import argparse
import numpy as np
from mypath import MyPath

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def main():
	cinic10_db_root_dir = MyPath.db_root_dir('cinic10')
	data_splits = ['train', 'valid', 'test']

	for data_split in data_splits:
		print(data_split)
		image_dir = os.path.join(cinic10_db_root_dir, data_split)

		classes = ['frog', 'airplane', 'horse', 'truck', 'cat', 'deer', 'automobile', 'dog', 'bird', 'ship']

		total_labels = len(classes)

		cnt_label =0

		data = []
		targets = []
		for i in range(len(classes)):
			imgs = os.listdir(os.path.join(image_dir, classes[i]))
			for j in imgs:
				img = Image.open(os.path.join(image_dir, classes[i], j))
				img = img.convert('RGB')
				img_data = np.asarray(img)
				data.append(img_data)
				targets.append(i)

		npy_path = os.path.join(cinic10_db_root_dir, 'npy')
		if not os.path.exists(npy_path):
			mkdir_p(npy_path)
		np.save(os.path.join(npy_path, data_split+'_data.npy'), data)
		np.save(os.path.join(npy_path, data_split+'_label.npy'), targets)

if __name__ == '__main__':
	main()