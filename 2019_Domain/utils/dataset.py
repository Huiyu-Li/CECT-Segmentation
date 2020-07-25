import os
import pandas as pd
import torch
import torch.utils.data as data
import mrcfile as mrc
import numpy as np

class CustomDataset(data.Dataset):
    def __init__(self, dir_csv):
        self.image_dirs = pd.read_csv(dir_csv,header=None).iloc[:, :].values#from DataFrame to array

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, item):
        img_name = self.image_dirs[item][0]
        mask_name = self.image_dirs[item][1]
        class_label = self.image_dirs[item][2]

        with mrc.open(img_name, permissive=True) as f:
            img = f.data  # (32, 32, 32)
        with mrc.open(mask_name, permissive=True) as f:
            mask = f.data  # (32, 32, 32)
        img = np.expand_dims(img,0)
        mask = np.expand_dims(mask, 0)
        class_label = np.expand_dims(class_label, 0)
        sample = {'image': img, 'mask': mask, 'class':class_label}
        return sample
