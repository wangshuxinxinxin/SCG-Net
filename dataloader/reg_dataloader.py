import os
import numpy as np
import pandas as pd

# third party imports
import cv2
import json
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom

from utils import *

class RegDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        
        self.angle = {}
        csvfile = 'angles.json'
        angles = json.load(open(csvfile))
        for iangle in angles:
            self.angle[iangle['name']] = iangle['angle']

        files = os.listdir(os.path.join(self.root_dir, 'data'))
        self.train_data = {}
        self.mask = {}
        self.train_period_files = []
        for sub_dir in files:
            img_path = os.path.join(self.root_dir, 'data', sub_dir)
            img = cv2.imread(img_path).swapaxes(0, 2).swapaxes(1, 2)

            label_path = img_path.replace('data', 'label')
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

            _, x, y = img.shape
            x_crop, y_crop = (x // 2) * 2, (y // 2) * 2
            img = img[:, :x_crop, :y_crop]
            label = label[:x_crop, :y_crop]

            self.train_period_files.append(sub_dir)
            self.train_data[sub_dir] = img.astype(np.int16)
            self.mask[sub_dir] = label.astype(np.int16)

        assert len(self.train_data) == len(self.mask)
        assert len(self.train_data) == len(self.train_period_files)


    def __len__(self):
        return len(self.train_period_files)


    def __getitem__(self, item):
        img_name = self.train_period_files[item]
        img = self.train_data[img_name].astype(np.float64)
        mask = self.mask[img_name].astype(np.float64)
        angle = self.angle[img_name]

        img = img / 255.
        mask = mask[np.newaxis, ...]
        img = np.concatenate((img, mask), axis = 0)

        sample = {'image': img, 'angle': angle, 'name': img_name}

        return sample

