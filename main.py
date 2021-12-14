import wandb
from model import LitMoCo
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger


seed_everything(42, workers=True)
wandb_logger = WandbLogger()

# net = LitMoCo()
x = torch.randn(1, 1, 28, 28)

model = LitMoCo()
trainer = Trainer(max_steps=100_000, gpus=1, logger=wandb_logger)
trainer.fit(model)

out = model(x)
print(out.shape)
