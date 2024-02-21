import os
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


class PretrainDataset(Dataset):
    def __init__(self, root, transform=None):
        self.img_dir = root
        self.transform = transform
        self.file_path_ls =[]
        # Use os.walk to iterate through all subfolders and files
        for folder, _, files in os.walk(root):
            for filename in files:
                if filename.lower().endswith(('.tif')):
                    self.file_path_ls.append(os.path.join(folder, filename))
                    
    def read_img(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_img:
            img = src_img.read()
            return np.transpose(img.astype(np.float32)/255, (1, 2, 0))

    def __len__(self):
        self.len = len(self.file_path_ls)
        return self.len
    
    def __getitem__(self, idx):
        img_path = self.file_path_ls[idx]
        img = self.read_img(img_path)
        if self.transform:
            img = self.transform(img)
        # pretrain dataset does not have annotation
        return img, 0
    
    
    