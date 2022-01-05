from main import main
import wandb
import argparse
import datetime

from pytorch_lightning.loggers import WandbLogger


def wandb_main() -> None:
    with wandb.init() as wandb_run:
        config = dict(wandb_run.config)
        main(config=config, wandb_logger=wandb_run)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run Knnn compare.')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use (cpu, cuda:0, cuda:1, ...)')
    parser.add_argument('--sweep_id', type=str, help='wandb sweeep id')
    args = parser.parse_args()
    main.device = args.device
    main.start_time = str(datetime.datetime.now()).split('.')[0].replace(':', '_').replace(' ', '_')
    wandb.agent(args.sweep_id, function=wandb_main)
    print('=== ALL DONE ===')