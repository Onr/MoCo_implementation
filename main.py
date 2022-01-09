import wandb
from model import LitMoCo
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
import yaml
from typing import Dict, Optional
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import datetime
import argparse

def main(config: Optional[Dict] = None, wandb_logger = None, pre_trained=None):
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

    ckpt_dir_path = os.path.join('./saved_ckpt/', str(datetime.datetime.now()).split('.')[0].replace(':', '_').replace(' ', '_'))
    checkpoint_callback = ModelCheckpoint(
        monitor="val_linear-acc",
        dirpath=ckpt_dir_path,
        filename="moco-{epoch:02d}-{val_linear-acc:.2f}",
        save_top_k=3,
        mode="max",
    )
    model = LitMoCo(config=config, wandb_logger=None, log_func=log_func)
    trainer = Trainer(max_epochs=config['max_epochs'], gpus=config['gpus'], logger=wandb_logger,
                      check_val_every_n_epoch=config['check_val_every_n_epoch'], callbacks=[checkpoint_callback],
                      devices=config['gpus'], accelerator='auto')
    if pre_trained is None:
        trainer.fit(model)
    else:
        model = model.load_from_checkpoint(checkpoint_path=pre_trained, config=config)
    trainer.test(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rcompare.')
    parser.add_argument('--pre_trained', type=str, default=None, help='location of the pre-trained net.')
    args = parser.parse_args()
    main(pre_trained=args.pre_trained)
