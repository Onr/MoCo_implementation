import wandb
from model import LitMoCo
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
import yaml
from typing import Dict, Optional
from pytorch_lightning.callbacks import ModelCheckpoint


def main(config: Optional[Dict] = None, wandb_logger = None):
    if config is None:
        with open("config.yaml", "r") as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    seed_everything(config['seed'], workers=True)
    if wandb_logger is None:
        wandb_logger = WandbLogger(entity='advanced-topics-in-deep-learning')
        wandb_logger.log_hyperparams(config)
        log_func = None
    else:
        def log_func(name, value):
            wandb_logger.log({name: value})

    checkpoint_callback = ModelCheckpoint(
        monitor="val_linear-acc",
        dirpath="./saved_ckpt/",
        filename="moco-{epoch:02d}-{val_linear-acc:.2f}",
        save_top_k=3,
        mode="max",
    )
    model = LitMoCo(config=config, wandb_logger=None, log_func=log_func)
    trainer = Trainer(max_steps=config['max_steps'], gpus=config['gpus'], logger=wandb_logger, check_val_every_n_epoch=config['check_val_every_n_epoch'], callbacks=[checkpoint_callback])
    trainer.fit(model)
    trainer.test(model)

if __name__ == '__main__':
    main()
