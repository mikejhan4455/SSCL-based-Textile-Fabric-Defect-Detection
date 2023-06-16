# pylint: disable=missing-function-docstring
# vision_transformer_pytorch version
# https://www.kaggle.com/code/piantic/vision-transformer-vit-visualize-attention-map
# https://pypi.org/project/vision-transformer-pytorch/

import gc
import math
import os
import sys

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from pixel_level_contrastive_learning import PixelCL
from torch.nn import Linear, CrossEntropyLoss
from torch.nn.utils import clip_grad_value_
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm import tqdm
from vision_transformer_pytorch import VisionTransformer
import wandb

sys.path.append("/workspace/persistent/Projects/CutPaste")

from dataset import NormalDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device != torch.device("cuda"):
    raise Exception("Device is not cuda")


WANDB_PROJECT = "vision-trasnformer-ablation-2023-06-14"
ROOT_PATH = "/workspace/persistent/Projects/little-batch-evauation/"
# METHOD = "pixpro"  # "regular"
# METHOD = "regular"
# ENCODER_MODEL = "ViT-B_16"
# EPOCH = 1000
# # LR = 1.0
# LR = 0.0001
# # BATCH_SIZE = 2
# BATCH_SIZE = 16
PRETRAINED = True
# DATASET = "normal"  # 'error' 'mixed'

# Define sweep config
sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": {
        "encoder_model": {"values": ["ViT-B_16"]},
        "optimizer": {"values": ["SGD"]},
        # "lr": {"values": [1, 0.1, 0.01, 0.001, 0.0001]},
        "lr": {"values": [0.001, 0.0001, 0.00001]},
        "epochs": {"values": [200]},
        "method": {"values": ["regular", "pixpro"]},
        "dataset": {"values": ["normal", "error", "mixed"]},
        "pretrained": {"values": [True, False]},
        "argumented": {"values": [True, False]},
    },
}


def get_experience_name(params=None):
    if params is None:
        params = wandb.config

    BATCH_SIZE = 2 if params.method == "pixpro" else 16

    return f"{params.method}_{params.encoder_model}_epoch-{params.epochs}_lr-{params.lr}_batch-{BATCH_SIZE}_pretrained-{params.pretrained}_dataset-{params.dataset}"


def get_dataset():
    # err_image_dataset = PatchedNormalDataset(
    #     "/workspace/persistent/Datasets/50_600x600_err-image",
    #     patch_size=384,
    #     stride=216,
    #     extra_transform=get_transform(),
    #     labled=False,
    # )

    err_image_dataset = NormalDataset(
        "/workspace/persistent/Datasets/50_600x600_err-image",
        extra_transform=get_transform(),
        labled=False,
    )

    fabric_image_dataset = NormalDataset(
        "/workspace/persistent/Datasets/captured_pic",
        extra_transform=get_transform(),
        labled=False,
    )

    if wandb.config.dataset == "normal":
        return fabric_image_dataset
    elif wandb.config.dataset == "error":
        return err_image_dataset
    elif wandb.config.dataset == "mixed":
        return ConcatDataset([err_image_dataset, fabric_image_dataset])


def get_transform(param=None):
    param = wandb.config if param is None else param

    if param.argumented == "True":
        # with argument
        # random crop (with flip and resize), color distortion, and Gaussian blur.

        transform = transforms.Compose(
            [
                # transforms.RandomResizedCrop(384),
                transforms.Resize((384, 384)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
                ),
                transforms.GaussianBlur(kernel_size=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    else:
        # no argument
        transform = transforms.Compose(
            [
                # transforms.RandomResizedCrop(384),
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    return transform


def get_model(classes=2):
    if PRETRAINED:
        vit = VisionTransformer.from_pretrained(wandb.config.encoder_model).to(device)

    else:
        vit = VisionTransformer.from_name(wandb.config.encoder_model).to(device)

    vit.classifier = Linear(in_features=768, out_features=classes)

    if wandb.config.method == "regular":
        return vit

    elif wandb.config.method == "pixpro":
        model = PixelCL(
            net=vit,
            image_size=384,
            # hidden_layer_pixel="layer4",  # leads to output of 8x8 feature map for pixel-level learning
            hidden_layer_instance=-2,  # leads to output for instance-level learning
            # projection_size=256,  # size of projection output, 256 was used in the paper
            projection_size=256,  # size of projection output, 256 was used in the paper
            # projection_hidden_size=2048,  # size of projection hidden dimension, paper used 2048
            projection_hidden_size=256,  # size of projection hidden dimension, paper used 2048
            moving_average_decay=0.99,  # exponential moving average decay of target encoder
            ppm_num_layers=1,  # number of layers for transform function in the pixel propagation module, 1 was optimal
            ppm_gamma=2,  # sharpness of the similarity in the pixel propagation module, already at optimal value of 2
            distance_thres=0.7,  # ideal value is 0.7, as indicated in the paper, which makes the assumption of each feature map's pixel diagonal distance to be 1 (still unclear)
            similarity_temperature=0.3,  # temperature for the cosine similarity for the pixel contrastive loss
            alpha=1.0,  # weight of the pixel propagation loss (pixpro) vs pixel CL loss
            use_pixpro=True,  # do pixel pro instead of pixel contrast loss, defaults to pixpro, since it is the best one
            cutout_ratio_range=(
                0.6,
                0.8,
            ),  # a random ratio is selected from this range for the random cutout
        )

        return model


def get_lr_scheduler(optimizer):
    scheduler = CosineAnnealingLR(optimizer, T_max=wandb.config.epochs, eta_min=0.1)

    # combination of warm-up and cosine annealing
    # Define the number of warm-up epochs and total training epochs
    warmup_epochs = 10

    # Define the learning rate scheduler
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return current_epoch / warmup_epochs

        progress = (current_epoch - warmup_epochs) / (
            wandb.config.epochs - warmup_epochs
        )
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    return scheduler


def get_attention_map(model, img, get_mask=False, param=None):
    """
    Get the attention map of the input image
    """
    transform = get_transform(param)
    x = transform(img)
    # x.size()

    model = model.to(device)
    x = x.to(device)

    logits, att_mat = model(x.unsqueeze(0))

    att_mat = torch.stack(att_mat).squeeze(1)

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat.to(device) + residual_att.to(device)
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(
            aug_att_mat[n].to(device), joint_attentions[n - 1].to(device)
        )

    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    if get_mask:
        result = cv2.resize(mask / mask.max(), img.size)
    else:
        mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]

        # reshape
        result = (mask * img).astype("uint8")

    return result


def plot_attention_map(
    original_img, att_map, epoch=None, cnt=None, output_dir=None, use_wandb=True
):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    ax1.set_title("Original")
    ax2.set_title(f"Attention Map Last Layer, epoch={epoch}")
    _ = ax1.imshow(original_img)
    _ = ax2.imshow(original_img)

    # make a threshold at the average of att_map, 0 if lower than threshold
    # att_map[att_map < att_map.mean()] = 0

    _ = ax2.imshow(att_map[:, :, 0], cmap="hot", alpha=0.3)

    # _ = ax2.imshow(att_map[:, :], cmap="hot", alpha=0.3)

    attention_map_folder = (
        output_dir
        if output_dir
        else os.path.join(
            ROOT_PATH, "experiences", get_experience_name(), "attention_map", cnt
        )
    )
    # mkdir if need
    if not os.path.exists(attention_map_folder):
        os.makedirs(attention_map_folder)

    plt.savefig(
        os.path.join(attention_map_folder, f"attention-e{epoch:03d}-img{cnt}"),
        bbox_inches="tight",
    )

    if use_wandb:
        wandb.log(
            {
                # f"original_{cnt}": wandb.Image(original_img, caption=f"{epoch:03d}"),
                # f"attention_map_{cnt}": wandb.Image(att_map, caption=f"{epoch:03d}"),
                # f"original_{cnt}": ax1,
                f"attention_map_{cnt}": ax2,
            }
        )

    matplotlib.pyplot.close()


def save_attention_map(
    image_path, model, epoch, cnt, param=None, output_dir=None, use_wandb=True
):
    img1 = Image.open(image_path)

    if param is not None:
        if param.method == "pixpro":
            model = model.online_encoder.net

    else:
        if wandb.config.method == "pixpro":
            # vit is wrapped
            model = model.online_encoder.net

    result = get_attention_map(model, img1, param)

    plot_attention_map(
        img1, result, epoch=epoch, cnt=cnt, output_dir=output_dir, use_wandb=wandb
    )


def fit(model, batch, optimizer, scheduler):
    BATCH_SIZE = 2 if wandb.config.method == "pixpro" else 16
    if wandb.config.method == "pixpro":
        loss = model(batch)

        optimizer.zero_grad()
        loss.backward()

        # Clip gradients using a specified value
        max_grad_value = 0.5  # Set the maximum gradient value
        clip_grad_value_(model.parameters(), max_grad_value)

        optimizer.step()
        scheduler.step()
        model.update_moving_average()  # update moving average of target encoder

    elif wandb.config.method == "regular":
        contrastive_loss = CrossEntropyLoss().to(device)

        # data argument
        output, attention = model(torch.cat((batch, batch), dim=0))

        scores = torch.matmul(output, output.t()) / 0.07  # Temperature scaling factor

        labels = torch.arange(BATCH_SIZE).to(device)
        labels = torch.cat((labels, labels), dim=0)

        loss = contrastive_loss(scores, labels)

        optimizer.zero_grad()
        loss.backward()

        # Clip gradients using a specified value
        max_grad_value = 0.5  # Set the maximum gradient value
        clip_grad_value_(model.parameters(), max_grad_value)

        optimizer.step()
        scheduler.step()

    return loss


# deprecated
def wandb_log_attention_map(epoch):
    attention_map_folder = os.path.join(
        ROOT_PATH, "experiences", get_experience_name(), "attention_map"
    )

    wandb.log(
        {
            "original_001": wandb.Image(
                "/workspace/persistent/Datasets/50_600x600_err-image/001.jpg"
            ),
            "attention_map_001": wandb.Image(
                os.path.join(attention_map_folder, f"attention-e{epoch:03d}-img001.png")
            ),
            "original_005": wandb.Image(
                "/workspace/persistent/Datasets/50_600x600_err-image/005.jpg"
            ),
            "attention_map_005": wandb.Image(
                os.path.join(attention_map_folder, f"attention-e{epoch:03d}-img005.png")
            ),
            "original_031": wandb.Image(
                "/workspace/persistent/Datasets/50_600x600_err-image/031.jpg"
            ),
            "attention_map_031": wandb.Image(
                os.path.join(attention_map_folder, f"attention-e{epoch:03d}-img031.png")
            ),
        }
    )


def train():
    gc.collect()
    torch.cuda.empty_cache()

    run = wandb.init(project=WANDB_PROJECT)
    print("=" * 30)
    print(get_experience_name())
    print("=" * 30)

    # get hyperparameters
    BATCH_SIZE = 2 if wandb.config.method == "pixpro" else 16

    dataset = get_dataset()

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
    )

    model = get_model()
    model = model.to(device)

    optimizer = torch.optim.SGD((model.parameters()), lr=wandb.config.lr, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR) # adam makes model error prediction nan values

    scheduler = get_lr_scheduler(optimizer)

    for epoch in tqdm(range(wandb.config.epochs)):
        gc.collect()
        torch.cuda.empty_cache()

        for batch, _ in dataloader:
            model.train()

            # patched image
            if isinstance(batch, list):
                for b in batch:
                    b = b.to(device)

                    loss = fit(model, b, optimizer, scheduler)

            # normal image
            else:
                batch = batch.to(device)

                loss = fit(model, batch, optimizer, scheduler)

        # on batch end
        wandb.log(
            {
                "epoch": epoch,
                "loss": loss.item(),
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if torch.isnan(loss).any():
            raise Exception("loss is nan")

        if epoch % 5 == 0:
            save_attention_map(
                image_path="/workspace/persistent/Datasets/50_600x600_err-image/001.jpg",
                model=model,
                epoch=epoch,
                cnt="001",
            )

            save_attention_map(
                image_path="/workspace/persistent/Datasets/50_600x600_err-image/005.jpg",
                model=model,
                epoch=epoch,
                cnt="005",
            )

            save_attention_map(
                image_path="/workspace/persistent/Datasets/50_600x600_err-image/031.jpg",
                model=model,
                epoch=epoch,
                cnt="031",
            )

            # wandb_log_attention_map(epoch)

        # save model
        model_folder = os.path.join(
            ROOT_PATH, "experiences", get_experience_name(), "model"
        )
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        if epoch % 40 == 0 or epoch == wandb.config.epochs - 1:
            torch.save(
                model,
                os.path.join(
                    model_folder,
                    get_experience_name() + f"_{epoch:03d}.pth",
                ),
            )

def main():
    if not os.path.exists("sweep_id"):
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=WANDB_PROJECT)

        # save sweep_id to file
        with open("sweep_id", "w") as f:
            f.write(sweep_id)

    else:
        with open("sweep_id", "r") as f:
            sweep_id = f.read()

    wandb.agent(project=WANDB_PROJECT, sweep_id=sweep_id, function=train, count=500)


if __name__ == "__main__":
    main()
