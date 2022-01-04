from pytorch_lightning.core.lightning import LightningModule
from torch import nn
import torch
from torch.optim import SGD


class LinearClassificationNet(LightningModule):
    def __init__(self, in_features, out_features, cfg):
        super().__init__()
        self.config = cfg
        self.lr = self.config['linear_lr']
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.norm = nn.Softmax(dim=1)

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
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        return acc

    def test_epoch_end(self, outputs) -> None:
        acc = float(torch.tensor(outputs).mean())
        self.log('test-acc', acc)

    # def train_dataloader(self):
    #     pass

    def configure_optimizers(self):
        sgd_optimizer = SGD(self.parameters(),
                            lr=self.lr,
                            momentum=self.config['linear_sgd_momentum'],
                            weight_decay=self.config['linear_sgd_weight_decay'])
        return sgd_optimizer
