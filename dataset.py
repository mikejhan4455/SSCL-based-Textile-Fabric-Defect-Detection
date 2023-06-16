# pylint: disable=too-few-public-methods
import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter, rotate
from sklearn.manifold import TSNE
from torch.autograd import Variable
from tqdm import tqdm

# from utils import get_experience_name

sys.path.append("/workspace/persistent/Projects/CutPaste/script/pytorch-cutpaste/")

import torchvision
from PIL import Image
from torchvision import transforms

class NormalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, extra_transform=None, labled=True):
        self.image_list = glob(dataset_path + "/*")

        if extra_transform is None:
            self.transform = transforms.Compose(
                [
                    # add other transforms here
                    torchvision.transforms.Resize([300, 400]),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    extra_transform,
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ],
            )

        self.labled = labled

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img = Image.open(self.image_list[idx])

        if self.transform is not None:
            img = self.transform(img)
        return img, NO_ANOMALY if self.labled else img
