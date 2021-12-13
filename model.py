import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import os
from torchvision import datasets, transforms
from pytorch_lightning import Trainer
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image
from torchvision.models.resnet import resnet18 as _resnet18

class MocoNetEncoder(nn.Module):
    def __init__(self,  repo_or_dir='pytorch/vision:v0.10.0', model='wide_resnet50_2', pretrain=True, out_features=128):
        self.encoder = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)  # todo check if to use pretrain
        self.encoder.fc = torch.nn.Linear(in_features=2048, out_features=out_features)

    def forward(self, x):
        x = self.encoder(x)
        return x

class LitMoCo(LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = MocoNetEncoder('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True, out_features=128)
        self.encoder_momentum = MocoNetEncoder('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True, out_features=128)

    def forward(self, x):
        x = self.encoder(x)
        return x

    def forward_momentum(self, x):
        x = self.encoder_momentum(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        # transforms
        transform = transforms.Compose([# transforms.Resize(300),
                                        transforms.RandomSizedCrop(224),
                                        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05), # TODO check unsupervised featuer learning via non parmetric instece discriminition for the values
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomGrayscale(p=0.1),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])  # TODO check
        # data
        imagenette2_train = Imagenette2(dir_path='./datasets/imagenette2-160', transform=transform, mode='train')
        return DataLoader(imagenette2_train, batch_size=64)

    def val_dataloader(self):
        # transforms
        transform = transforms.Compose([transforms.RandomSizedCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])  # TODO check
        # data
        imagenette2_val = Imagenette2(dir_path='./datasets/imagenette2-160', transform=transform, mode='val')
        return DataLoader(imagenette2_val, batch_size=64)

class Imagenette2(Dataset):
    def __init__(self, dir_path, mode, transform=None, target_transform=None):
        self.mode = mode
        dir_path = os.path.join(dir_path, mode)
        class_dirs = [os.path.join(dir_path, cur_path) for cur_path in os.listdir(dir_path) if not cur_path.startswith('.')]
        self.images_paths = []
        self.images_labels = []
        for cur_dir in class_dirs:
            curr_images_paths = [os.path.join(cur_dir, img_name) for img_name in os.listdir(cur_dir)]
            curr_images_labels = [cur_dir.split('/')[-1]] * len(curr_images_paths)
            self.images_paths += curr_images_paths
            self.images_labels += curr_images_labels

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        img_path = self.images_paths[idx]
        img_label = self.images_labels[idx]
        # image = read_image(img_path) / 255
        image = Image.open(img_path)
        # # tmp todo remove
        # # tmp
        # import matplotlib.pyplot as plt
        # plt.imshow(torch.permute(image, (1, 2, 0)))
        # plt.show()
        # # tmp
        # # tmp
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            img_label = self.target_transform(img_label)
        return image, img_label
