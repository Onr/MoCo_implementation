import wandb
from model import LitMoCo
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
import yaml

with open("config.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

seed_everything(config['seed'], workers=True)
wandb_logger = WandbLogger(entity='advanced-topics-in-deep-learning')
wandb_logger.log_hyperparams(config)
model = LitMoCo(config=config)
trainer = Trainer(max_steps=config['max_steps'], gpus=config['gpus'], logger=wandb_logger, check_val_every_n_epoch=config['check_val_every_n_epoch'])
trainer.fit(model)

x = torch.ones(1, 1, 28, 28)
out = model(x)
