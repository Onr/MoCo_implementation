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
from tqdm import tqdm
import itertools

class MocoNetEncoder(nn.Module):
    def __init__(self,  repo_or_dir='pytorch/vision:v0.10.0', model='wide_resnet50_2', pretrained=True, out_features=128):
        super().__init__()
        self.encoder = torch.hub.load(repo_or_dir=repo_or_dir, model=model, pretrained=pretrained)  # todo check if to use pretrain
        self.encoder.fc = torch.nn.Linear(in_features=2048, out_features=out_features)

    def forward(self, x):
        x = self.encoder(x)
        return x
class LinearClassificationNet(LightningModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)


    def forward(self, x):
        out = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        outs = self(batch[0])
        labels = batch[-1]
        loss = nn.CrossEntropyLoss(outs, labels)

    def train_dataloader(self):
        pass

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)




class LitMoCo(LightningModule):
    def __init__(self, channels=128, queue_size=256, temperature=0.1, momentum=0.999, num_of_classes=10):
        super().__init__()
        self.num_of_classes = num_of_classes
        self.C = channels
        self.queue_size = queue_size
        self.temperature = temperature
        self.momentum = momentum
        self.encoder = MocoNetEncoder('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True, out_features=self.C)
        self.encoder_momentum = MocoNetEncoder('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=False, out_features=self.C)
        self.queue = torch.zeros(self.C, self.queue_size, device=self.device)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.cur_dictionary_ind = 0

    def forward(self, x):
        x = self.encoder(x)
        return x

    def forward_momentum(self, x):
        x = self.encoder_momentum(x)
        return x

    def training_step(self, batch, batch_idx):
        N = batch[0].shape[0]
        x_q, x_k, y = batch

        print(f'{x_k.shape = }')
        # momentum update: key network
        for (name_encoder, W_encoder), (name_encoder_momentum, W_encoder_momentum) in zip(self.encoder.named_parameters(), self.encoder_momentum.named_parameters()):
            if 'weight' in name_encoder:
                W_encoder_momentum = self.momentum * W_encoder_momentum + (1 - self.momentum) * W_encoder  # todo check the momentom is on the rigth side


        q = self.forward(x_q)
        k = self.forward_momentum(x_k)
        k = k.detach()

        # positive logits: Nx1
        l_pos = torch.bmm(q.view(N, 1, self.C), k.view(N, self.C, 1)).squeeze(-1)

        # negative logits: NxK
        l_neg = torch.mm(q.view(N, self.C), self.queue.clone().to(self.device))  # todo check if there is an alternative to clone

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # contrastive loss, Eqn. (1)
        labels = torch.zeros(N, dtype=torch.long, device=self.device)  # positives are the 0-th
        loss = self.loss_func(logits/self.temperature, labels)

        # update dictionary # todo add ability for queue to not be multiplication of the minibatch
        start_ind = self.cur_dictionary_ind
        end_ind = self.cur_dictionary_ind + k.T.shape[1]
        if end_ind < self.queue.shape[1]:
            self.queue[:, start_ind: end_ind] = k.T.detach()
        else:
            end_ind = self.queue.shape[1] - 1
            end_len = end_ind - start_ind
            remain_len = k.T.shape[1] - end_len
            self.queue[:, start_ind: end_ind] = k.T.detach()[:, :end_len]
            self.queue[:, :remain_len] = k.T.detach()[:, end_len:]

        self.cur_dictionary_ind = (self.cur_dictionary_ind + k.T.shape[1]) % self.queue.shape[1]

        self.log(name='loss', value=loss)
        return loss

    def on_validation_start(self):
        dataloader = self.val_dataloader()
        embedding_s = []
        label_s = []

        embedding_and_labels = [[self(cur_emb[0].cuda()), cur_emb[1]] for cur_emb in tqdm(dataloader, desc='getting embedings for linear classifier')]
        embedding_s = torch.cat([curr[0] for curr in embedding_and_labels], dim=0)
        label_s = list(itertools.chain.from_iterable([curr[1] for curr in embedding_and_labels]))
        embedding_data = zip(embedding_s, label_s)
        class EmbeddingDataset(Dataset):
            def __init__(self, data):
                super().__init__()
                self.data = data
            def __getitem__(self, idx):
                return self.data[idx]
            def __len__(self):
                return len([1 for _ in self.data])
        embedding_dataset = EmbeddingDataset(embedding_data)

        embedding_dataloader = DataLoader(embedding_dataset)

        net = LinearClassificationNet(in_features=self.C, out_features=self.num_of_classes)
        trainer = Trainer()
        trainer.fit(net, train_dataloader=embedding_dataloader)

    def validation_step(self, batch, batch_idx):

        print('tmp')
        return

    def validation_epoch_end(self, outputs):
        print('tmp')
        pass

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        # transforms
        transform = transforms.Compose([transforms.RandomSizedCrop(224),
                                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomGrayscale(p=0.2),
                                        transforms.GaussianBlur((35, 35)),  # TODO: check
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])
        # data
        imagenette2_train = Imagenette2_dataset(dir_path='./datasets/imagenette2-160', transform=transform, mode='train')
        return DataLoader(imagenette2_train, batch_size=64)

    def val_dataloader(self):
        # transforms
        transform = transforms.Compose([transforms.RandomSizedCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])
        # data
        imagenette2_val = Imagenette2_dataset(dir_path='./datasets/imagenette2-160', transform=transform, mode='val')
        return DataLoader(imagenette2_val, batch_size=64)

class Imagenette2_dataset(Dataset):
    def __init__(self, dir_path, mode, transform=None, target_transform=None):
        self.mode = mode
        dir_path = os.path.join(dir_path, mode)
        class_dirs = [os.path.join(dir_path, cur_path) for cur_path in os.listdir(dir_path) if not cur_path.startswith('.')]
        self.images_paths = []
        self.images_labels = []
        class_name_to_class_id = {
            'chainsaw': 0,
            'church': 1,
            'dog': 2,
            'fish': 3,
            'gas_station': 4,
            'golf': 5,
            'parachute': 6,
            'radio': 7,
            'truck': 8,
            'trumpet': 9,
        }
        for cur_dir in class_dirs:
            curr_images_paths = [os.path.join(cur_dir, img_name) for img_name in os.listdir(cur_dir)]
            curr_images_labels = [cur_dir.split('/')[-1]] * len(curr_images_paths)
            self.images_paths += curr_images_paths
            self.images_labels += [class_name_to_class_id[cur_lab] for cur_lab in curr_images_labels]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        img_path = self.images_paths[idx]
        img_label = self.images_labels[idx]
        image_pil = Image.open(img_path).convert('RGB')

        if self.transform:
            image_q = self.transform(image_pil)
            if self.mode is 'train':
                image_k = self.transform(image_pil)
        if self.target_transform:
            img_label = self.target_transform(img_label)
        if self.mode is 'train':
            return image_q, image_k, img_label
        else:
            return image_q, img_label
