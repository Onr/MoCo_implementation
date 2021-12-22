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
    def __init__(self, config):
        super().__init__()
        self.encoder = torch.hub.load(repo_or_dir=config['repo_or_dir'], model=config['model'],
                                      pretrained=config['pretrained'])  # todo check if to use pretrain
        self.embedding_size = self.encoder.fc.in_features
        self.encoder.fc = torch.nn.Identity()
        self.norm = torch.nn.BatchNorm1d(num_features=self.embedding_size)

        class MLP_encoder(nn.Module):
            def __init__(self, in_features=2048, out_features=128):
                super().__init__()

                self.fc1 = torch.nn.Linear(in_features=in_features, out_features=in_features)
                self.non_lin = nn.ReLU()
                self.fc2 = torch.nn.Linear(in_features=in_features, out_features=out_features)

            def forward(self, x):
                out = self.fc1(x)
                out = self.non_lin(out)
                out = self.fc2(out)
                return out

        self.encoder_end = MLP_encoder(in_features=self.embedding_size, out_features=config['mlp_out_features'])

    def forward(self, x):
        x = self.encoder(x)
        x = self.norm(x)
        if self.training:
            x = self.encoder_end(x)
        return x


class LinearClassificationNet(LightningModule):
    def __init__(self, in_features, out_features, lr):
        super().__init__()
        self.lr = lr
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.norm = nn.Sigmoid()
        # self.norm = nn.Softmax(dim=1)

    def forward(self, x):
        outs = self.fc(x)
        preds = self.norm(outs)
        return preds

    def training_step(self, batch, batch_idx):
        preds = self(batch[0])
        labels = batch[-1]
        loss = nn.CrossEntropyLoss()(preds, labels)
        return loss

    def test_step(self, batch, batch_idx):
        preds = self(batch[0])
        labels = batch[-1]
        loss = nn.CrossEntropyLoss()(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log('test-acc', acc)
        return loss

    # def train_dataloader(self):
    #     pass

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


class EmbeddingDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        # return len([1 for _ in self.data])
        return len(self.data)


class LitMoCo(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_of_classes = config['num_of_classes']
        self.C = config['channels']
        self.queue_size = config['queue_size']
        self.temperature = config['temperature']
        self.momentum = config['momentum']
        self.encoder = MocoNetEncoder(config=config)
        self.encoder_momentum = MocoNetEncoder(config=config)
        self.queue = torch.zeros(self.C, self.queue_size, device=self.device)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.cur_dictionary_ind = 0
        # self.linear_trainer = Trainer(max_epochs=10, devices=1)
        self.linear_net = LinearClassificationNet(in_features=self.encoder.embedding_size, out_features=self.num_of_classes, lr=self.config['linear_lr'])
        self.linear_trainer = Trainer(max_epochs=self.config['linear_max_epoch'], gpus=self.config['gpus'])

    def forward(self, x):
        x = self.encoder(x)
        return x

    def forward_momentum(self, x):
        x = self.encoder_momentum(x)
        return x

    def training_step(self, batch, batch_idx):
        N = batch[0].shape[0]
        x_q, x_k, y = batch

        # momentum update: key network
        for (name_encoder, W_encoder), (name_encoder_momentum, W_encoder_momentum) in zip(
                self.encoder.named_parameters(), self.encoder_momentum.named_parameters()):
            if 'weight' in name_encoder:
                W_encoder_momentum = self.momentum * W_encoder_momentum + (1 - self.momentum) * W_encoder  # todo check the momentom is on the rigth side

        q = self.forward(x_q)
        k = self.forward_momentum(x_k)
        k = k.detach()

        # positive logits: Nx1
        l_pos = torch.bmm(q.view(N, 1, self.C), k.view(N, self.C, 1)).squeeze(-1)

        # negative logits: NxK
        l_neg = torch.mm(q.view(N, self.C),
                         self.queue.clone().to(self.device))  # todo check if there is an alternative to clone

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # contrastive loss, Eqn. (1)
        labels = torch.zeros(N, dtype=torch.long, device=self.device)  # positives are the 0-th
        loss = self.loss_func(logits / self.temperature, labels)

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

    def validation_step(self, batch, batch_idx):
        # todo maybe move get embedding to here
        return

    def validation_epoch_end(self, outputs):
        embedding_and_labels = [[self(cur_emb[0].cuda()), cur_emb[1]] for cur_emb in
                                tqdm(self.val_dataloader(), desc='getting embedings for linear classifier')]
        embedding_s = torch.cat([curr[0] for curr in embedding_and_labels], dim=0)
        label_s = list(itertools.chain.from_iterable([curr[1] for curr in embedding_and_labels]))
        embedding_data = list(zip(embedding_s, label_s))
        embedding_dataset = EmbeddingDataset(embedding_data)
        embedding_dataloader = DataLoader(embedding_dataset, batch_size=self.config['linear_batch_size'])
        self.linear_trainer.fit(model=self.linear_net, train_dataloader=embedding_dataloader)
        test_results = self.linear_trainer.test(model=self.linear_net, test_dataloaders=embedding_dataloader)
        self.log('val_linear-acc', test_results[0]['test-acc'])

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.config['model_lr'])

    def train_dataloader(self):
        # transforms
        transform = transforms.Compose([transforms.RandomSizedCrop(self.config['RandomSizedCrop']),
                                        transforms.ColorJitter(brightness=self.config['ColorJitter_brightness'],
                                                               contrast=self.config['ColorJitter_contrast'],
                                                               saturation=self.config['ColorJitter_saturation'],
                                                               hue=self.config['ColorJitter_hue']),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomGrayscale(p=self.config['RandomGrayscale']),
                                        transforms.RandomApply(torch.nn.ModuleList([
                                            transforms.GaussianBlur(kernel_size=(
                                                1 + self.config['RandomSizedCrop'] // 10, 1 + self.config['RandomSizedCrop'] // 10),
                                                                        sigma=(self.config['GaussianBlur_min'],
                                                                               self.config['GaussianBlur_max'])),
                                        ]), p=0.5),

                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])
        # data
        imagenette2_train = Imagenette2_dataset(dir_path=self.config['dataset'], transform=transform,
                                                mode='train', do_all_dataset_in_memory=self.config['do_all_dataset_in_memory'])
        return DataLoader(imagenette2_train, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'])

    def val_dataloader(self):
        # transforms
        transform = transforms.Compose([transforms.RandomSizedCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])
        # data
        imagenette2_val = Imagenette2_dataset(dir_path=self.config['dataset'], transform=transform, mode='val', do_all_dataset_in_memory=self.config['do_all_dataset_in_memory'])
        return DataLoader(imagenette2_val, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'])


class Imagenette2_dataset(Dataset):
    def __init__(self, dir_path, mode, transform=None, target_transform=None, do_all_dataset_in_memory=True):
        self.mode = mode
        dir_path = os.path.join(dir_path, mode)
        class_dirs = [os.path.join(dir_path, cur_path) for cur_path in os.listdir(dir_path) if
                      not cur_path.startswith('.')]
        self.images_paths = []
        self.images_labels = []
        # class_name_to_class_id = {
        #     'chainsaw': 0,
        #     'church': 1,
        #     'dog': 2,
        #     'fish': 3,
        #     'gas_station': 4,
        #     'golf': 5,
        #     'parachute': 6,
        #     'radio': 7,
        #     'truck': 8,
        #     'trumpet': 9,
        # }
        class_names = [class_name for class_name in os.listdir(dir_path)]

        class_name_to_class_id = {}
        for idx, class_name in enumerate(class_names):
            class_name_to_class_id[class_name] = idx

        for cur_dir in class_dirs:
            curr_images_paths = [os.path.join(cur_dir, img_name) for img_name in os.listdir(cur_dir)]
            curr_images_labels = [cur_dir.split('/')[-1]] * len(curr_images_paths)
            self.images_paths += curr_images_paths
            self.images_labels += [class_name_to_class_id[cur_lab] for cur_lab in curr_images_labels]

        self.do_all_dataset_in_memory = do_all_dataset_in_memory
        if self.do_all_dataset_in_memory:
            self.imgs = []
            self.labels = []
            for img_path, img_label in zip(self.images_paths, self.images_labels):
                image_pil = Image.open(img_path).convert('RGB')
                self.imgs += [image_pil]
                self.labels += [img_label]


        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        if not self.do_all_dataset_in_memory:
            img_path = self.images_paths[idx]
            img_label = self.images_labels[idx]
            image_pil = Image.open(img_path).convert('RGB')
        else:
            image_pil = self.imgs[idx]
            img_label = self.labels[idx]
        if self.transform:
            image_q = self.transform(image_pil)
            if self.mode == 'train':
                image_k = self.transform(image_pil)
        if self.target_transform:
            img_label = self.target_transform(img_label)
        if self.mode == 'train':
            return image_q, image_k, img_label
        else:
            return image_q, img_label
