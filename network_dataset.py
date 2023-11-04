import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps    

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class NetworkDataset(Dataset):
    def __init__(self,imageFolderDataset,transform=None):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        #We need to get a image in the same class and in a different class
        while True:
            #Look untill the same class image is found
            img1_tuple = random.choice(self.imageFolderDataset.imgs) 
            if img0_tuple[1] == img1_tuple[1]:
                break
        while True:
            #Look untill a different class image is found
            img2_tuple = random.choice(self.imageFolderDataset.imgs) 
            if img0_tuple[1] != img2_tuple[1]:
                break

        # open image file
        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img2 = Image.open(img2_tuple[0])

        # to grayscale
        # img0 = img0.convert("L")
        # img1 = img1.convert("L")
        # img2 = img2.convert("L")

        # to RBG
        img0 = img0.convert("RGB")
        img1 = img1.convert("RGB")
        img2 = img2.convert("RGB")


        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img0, img1, img2
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)