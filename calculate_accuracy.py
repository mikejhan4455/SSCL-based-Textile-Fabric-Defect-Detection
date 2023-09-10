# Cauculate model Accuracy(TP+TN)/(TP+TN+FP+FN)

import sys  
import os
import cv2
import numpy as np
import torch
import glob
from torch.utils.data import Dataset
from PIL import Image
from tuning import get_attention_map, plot_attention_map, save_attention_map
import torchvision.transforms as transforms


experience_root = '/workspace/persistent/Projects/SSCL-based Textile Fabric Defect Detection/experiences'
test_data_root = '/workspace/persistent/Datasets/50_600x600_err-image'
# models = glob.glob(os.path.join(experience_root, 'regular*/model/*199.pth'))
models = glob.glob(os.path.join(experience_root, 'regular_*/model/*199.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class  DefectDetectionDataset(Dataset):
    """
    Dataset class for defect detection dataset
    """
    # this dataset is a segment dataset with black and white annotations
    # black means non-defect area and white means defect
    # split the dataset into train and test set
    # dataset hierarchy:
    #   dataset_root
    #   |---image
    #   |   |---image1.png
    #   |   |---image2.png
    #   |   |---...
    #   |---mask
    #   |   |---mask1.png
    #   |   |---mask2.png
    #   |   |---...

    def __init__(self, dataset_path, test=False, transform=None, transform_mask=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.transform_mask = transform_mask
        self.imgs = self._get_imgs()
        self.masks = self._get_masks()
        self.testset_ratio = 0.2
        self.test = test
        self.trainset_size = int(len(self.imgs) * (1 - self.testset_ratio))
        self.testset_size = int(len(self.imgs) * self.testset_ratio)
        if test == 'all':
            self.imgs = self.imgs
            self.masks = self.masks
        elif test:
            self.imgs = self.imgs[self.trainset_size-1:]
            self.masks = self.masks[self.trainset_size-1:]
        else:
            self.imgs = self.imgs[:self.trainset_size+1]
            self.masks = self.masks[:self.trainset_size+1]
            
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_path, "images", self.imgs[idx])
        mask_path = os.path.join(self.dataset_path, "masks", self.masks[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)

        if self.transform_mask:
            mask = self.transform_mask(mask)

        # Return a 1D tensor as the target
        return image, mask
    
    def _get_imgs(self):
        img_path = os.path.join(self.dataset_path, 'image')
        # imgs = [os.path.join(img_path, img) for img in os.listdir(img_path)]
        imgs = sorted(glob.glob(os.path.join(img_path, '*.jpg')))
        return imgs
    
    def _get_masks(self):
        mask_path = os.path.join(self.dataset_path, 'mask')
        masks = sorted(glob.glob(os.path.join(mask_path, '*.png')))
        return masks

def get_transform():
    transform = transforms.Compose(
        [
            # transforms.RandomResizedCrop(384),
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225],
            # ),
        ]
    )

    return transform

def load_model(model_path):
    """
    Load model from model_path
    """
    
    model = torch.load(model_path)
    return model

def load_test_data():
    """
    Load test data
    """
    test_data = DefectDetectionDataset(test_data_root, test='all')
    return test_data

# TODO: WIP
# def calculate_accuracy_resnet(model, test_data, param):


def calculate_accuracy_vit(model, test_data, THRESHOLD_BIAS):
    """
    Calculate accuracy
    """

    model_accuracy = 0
    model_precision = 0
    model_recall = 0

    model_TP = 0
    model_TN = 0
    model_FP = 0
    model_FN = 0

    for i in range(test_data.__len__()):

        img, mask = test_data.__getitem__(i)
        
        # result = get_attention_map(model, img, get_mask=False, get_transform=get_transform)
        result = get_attention_map(model, img, get_transform=get_transform)[:, :, 0]

        # save result
        white = np.ones((600, 600)) * 255
        attention_result = (result * white).astype("uint8")
        # attention_on_image = (result * img).astype("uint8")
        attention_on_image = (cv2.resize(result / result.max(), img.size)[..., np.newaxis] * img).astype("uint8")


        # 二值化
        # threshold = attention_result.mean() + 20
        threshold = np.median(attention_result) + THRESHOLD_BIAS

        _, attention_result = cv2.threshold(attention_result, threshold, 255, cv2.THRESH_BINARY)

        img.save(f'result_{THRESHOLD_BIAS}/img_{i}.png')
        mask.save(f'result_{THRESHOLD_BIAS}/mask_{i}.png')
        cv2.imwrite(f'result_{THRESHOLD_BIAS}/attention_{i}.png', attention_result)
        cv2.imwrite(f'result_{THRESHOLD_BIAS}/attention_on_image_{i}.png', attention_on_image)

        # calculate single image accuracy
        # TP: attention_result == 255 and mask == 255
        # TN: attention_result == 0 and mask == 0
        # FP: attention_result == 255 and mask == 0
        # FN: attention_result == 0 and mask == 255

        mask = np.array(mask)
        
        TP = np.sum((attention_result == 255) & (mask == 255))
        TN = np.sum((attention_result == 0) & (mask == 0))
        FP = np.sum((attention_result == 255) & (mask == 0))
        FN = np.sum((attention_result == 0) & (mask == 255))


        calculate_metrics(TN, FP, FN, TP)

        # calculate model accuracy
        model_TP += TP
        model_TN += TN
        model_FP += FP
        model_FN += FN

    print('-' * 15 + 'END OF MODEL' + '-' * 15)

    # calculate model accuracy
    model_accuracy, model_precision, model_recall = calculate_metrics(model_TN, model_FP, model_FN, model_TP)

    print('-' * 15 + 'END OF MODEL' + '-' * 15)
    # write into file
    with open(f'result_{THRESHOLD_BIAS}/accuracy_median+{THRESHOLD_BIAS}.txt', 'a') as f:
        f.write(f'Accuracy: {model_accuracy}\n')
        f.write(f'Precision: {model_precision}\n')
        f.write(f'Recall: {model_recall}\n')

    return model_accuracy, model_precision, model_recall


def main():
    """
    1. Load model
    2. Load test data
    3. calculate attention map / gradient map
    4. calculate accuracy
    """

    best_model_accuracy, best_model_precision, best_model_recall = 0, 0, 0
    best_model_accuracy_name = ''
    best_model_precision_name = ''
    best_model_recall_name = ''

    print('Loading model...')
    print('Number of models: {}'.format(len(models)))

    for THRESHOLD_BIAS in (0, 10, 20, 30, 40):

        for model_path in models:
            print('Loading model: {}'.format(model_path))
            model = load_model(model_path)
            model.eval()
            model = model.to(device)
            print('Model loaded')

            print('Loading test data...')
            test_data = load_test_data()
            print('Test data loaded')

            print('Calculating accuracy...')
            # write model name
            os.makedirs(f'result_{THRESHOLD_BIAS}', exist_ok=True)
            with open(f'result_{THRESHOLD_BIAS}/accuracy_median+{THRESHOLD_BIAS}.txt', 'a') as f:
                f.write('-' * 15 + 'START OF MODEL' + '-' * 15 + '\n')
                f.write(f'Model: {model_path}\n')
            model_accuracy, model_precision, model_recall = calculate_accuracy_vit(model, test_data, THRESHOLD_BIAS)

            if model_accuracy > best_model_accuracy:
                best_model_accuracy = model_accuracy
                best_model_accuracy_name = model_path
            if model_precision > best_model_precision:
                best_model_precision = model_precision
                best_model_precision_name = model_path
            if model_recall > best_model_recall:
                best_model_recall = model_recall
                best_model_recall_name = model_path

        print('Best model accuracy: {}'.format(best_model_accuracy))
        print('Best model accuracy name: {}'.format(best_model_accuracy_name))
        print('Best model precision: {}'.format(best_model_precision))
        print('Best model precision name: {}'.format(best_model_precision_name))
        print('Best model recall: {}'.format(best_model_recall))
        print('Best model recall name: {}'.format(best_model_recall_name))

        with open(f'result_{THRESHOLD_BIAS}/accuracy_median+{THRESHOLD_BIAS}.txt', 'a') as f:
            f.write('Best model accuracy: {}\n'.format(best_model_accuracy))
            f.write('Best model accuracy name: {}\n\n'.format(best_model_accuracy_name))


            f.write('Best model precision: {}\n'.format(best_model_precision))
            f.write('Best model precision name: {}\n\n'.format(best_model_precision_name))


            f.write('Best model recall: {}\n'.format(best_model_recall))
            f.write('Best model recall name: {}\n\n'.format(best_model_recall_name))
        
        model_accuracy = 0
        model_precision = 0
        model_recall = 0

        model_TP = 0
        model_TN = 0
        model_FP = 0
        model_FN = 0


def calculate_metrics(TN, FP, FN, TP, print=True):
    accuracy, precision, recall = (TP + TN) / (TP + TN + FP + FN), TP / (TP + FP), TP / (TP + FN)

    if print:
        print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}')

        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print('-' * 30)

    return accuracy, precision, recall


if __name__ == '__main__':
    main()
