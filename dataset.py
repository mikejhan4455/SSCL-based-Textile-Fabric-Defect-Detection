# pylint: disable=too-few-public-methods
import sys
from glob import glob
import torch

import torchvision
from PIL import Image
from torchvision import transforms

sys.path.append("pytorch-cutpaste/")

# from cutpaste import CutPaste3Way, CutPasteNormal, CutPasteScar


NO_ANOMALY = 0
ANOMALY = 1


# class CutPasteTransformNormal(CutPasteNormal):
#     """Return transformed image only"""

#     def __init__(self, area_ratio=[0.02, 0.15], aspect_ratio=0.3, **kwags):
#         super(CutPasteNormal, self).__init__(**kwags)
#         self.area_ratio = area_ratio
#         self.aspect_ratio = aspect_ratio

#     def __call__(self, img):
#         return super().__call__(img)[1]


# class CutPasteTransformScar(CutPasteScar):
#     """Return transformed image only"""

#     def __init__(self, area_ratio=[0.02, 0.15], aspect_ratio=0.3, **kwags):
#         super(CutPasteScar, self).__init__(**kwags)
#         self.area_ratio = area_ratio
#         self.aspect_ratio = aspect_ratio

#     def __call__(self, img):
#         return super().__call__(img)[1]


# class CutPasteTransform3Way(CutPaste3Way):
#     """Return transformed image only"""

#     def __init__(self, area_ratio=[0.02, 0.15], aspect_ratio=0.3, **kwags):
#         super(CutPasteScar, self).__init__(**kwags)
#         self.area_ratio = area_ratio
#         self.aspect_ratio = aspect_ratio

#     def __call__(self, img):
#         return super().__call__(img)[1]


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


class AnomalyDataset(torch.utils.data.Dataset):
    def __init__(
        self, dataset_path, extra_transform=None, cutpaste_type="NORMAL", labled=True
    ):
        self.image_list = glob(dataset_path + "/*")

        if extra_transform is None:
            transform = transforms.Compose(
                [
                    # add other transforms here
                    torchvision.transforms.Resize([300, 400]),
                    transforms.ToTensor(),
                ]
            )
        else:
            transform = transforms.Compose([extra_transform])

        self.transform = transforms.Compose([])

        if cutpaste_type == "NORMAL":
            self.transform.transforms.append(
                CutPasteTransformNormal(transform=transform)
            )
        elif cutpaste_type == "SCAR":
            self.transform.transforms.append(CutPasteTransformScar(transform=transform))
        elif cutpaste_type == "3WAY":
            self.transform.transforms.append(CutPasteTransform3Way(transform=transform))

        self.labled = labled

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img = Image.open(self.image_list[idx])

        if self.transform is not None:
            img = self.transform(img)
        return img, ANOMALY if self.labled else img
