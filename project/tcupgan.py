import argparse
import json
import logging
import logging.config
import os
import sys
from enum import Enum
from typing import List

import torch
from tcupgan.model import LSTMUNet
from tcupgan.disc import PatchDiscriminator
from tcupgan.trainer import TrainerUNet
from tcupgan.io import BraTSDataGenerator
from torchinfo import summary
from torch.utils.data import DataLoader, random_split

logger = logging.getLogger(__name__)


class Task(str, Enum):
    """Define tasks"""

    DownloadData = "download"
    Train = "train"
    Evaluate = "evaluate"


def train(task_args: List[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '--data-dir', type=str, required=True)
    parser.add_argument('--config_file', type=str)
    args = parser.parse_args(args=task_args)

    if os.path.isfile(args.config_file):
        with open(args.config_file, 'r') as infile:
            config = json.load(infile)
    else:
        config = {}

    ckpt_folder = config.get('checkpoint_folder', 'checkpoints/')
    run_name = config.get('run_name', 'TCuPGAN')
    batch_size = config.get('batch_size', 2)
    train_val_split = config.get('train_val_split', 0.9)
    start_from_last = config.get('start_from_last', False)
    transfer_learn_ckpt = config.get('transfer_learn_checkpoint', '')
    gen_lr = config.get('generator_learning_rate', 1e-3)
    dsc_lr = config.get('discriminator_learning_rate', 1e-4)

    assert not (start_from_last and transfer_learn_ckpt != ''), 'cannot load from last save AND transfer learn from a different model'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = BraTSDataGenerator(args.data_dir)

    assert train_val_split < 1, "Training/Validation split must be less than 1!"

    train_datagen, val_datagen = random_split(data, [train_val_split, 1 - train_val_split])

    train_data = DataLoader(train_datagen, batch_size=batch_size, num_workers=5, shuffle=True, pin_memory=True)
    val_data = DataLoader(val_datagen, batch_size=batch_size, num_workers=5, shuffle=True, pin_memory=True)

    hidden = [6, 12, 18, 64, 128]
    generator = LSTMUNet(hidden_dims=hidden, input_channels=4, output_channels=4).to(device)

    discriminator = PatchDiscriminator(input_channels=8, nlayers=5, nfilt=16).to(device)

    summary(generator, input_size=(1, 155, 4, 256, 256), device=device)
    summary(discriminator, input_size=(1, 155, 8, 256, 256), device=device)

    trainer = TrainerUNet(generator, discriminator, os.path.join(ckpt_folder, run_name))
    if start_from_last:
        trainer.load_last_checkpoint()
    if transfer_learn_ckpt != '':
        trainer.load_transfer_data(args.transfer_learn_ckpt)

    gen_learning_rate = gen_lr
    dsc_learning_rate = dsc_lr

    trainer.seg_alpha = 200
    trainer.loss_type = 'tversky'

    trainer.train(train_data, val_data, 50, lr_decay=0.95,
                  dsc_learning_rate=dsc_learning_rate,
                  gen_learning_rate=gen_learning_rate, save_freq=5)


def main():
    """
    tcupgan.py task task_specific_parameters...
    """
    # noinspection PyBroadException
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("mlcube_task", type=str, help="Task for this MLCube.")
        parser.add_argument(
            "--log_dir", "--log-dir", type=str, required=True, help="Logging directory."
        )
        mlcube_args, task_args = parser.parse_known_args()

        os.makedirs(mlcube_args.log_dir, exist_ok=True)
        logger_config = {
            "version": 1,
            "disable_existing_loggers": True,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s"
                },
            },
            "handlers": {
                "file_handler": {
                    "class": "logging.FileHandler",
                    "level": "INFO",
                    "formatter": "standard",
                    "filename": os.path.join(
                        mlcube_args.log_dir,
                        f"tcupgan_{mlcube_args.mlcube_task}.log",
                    ),
                }
            },
            "loggers": {
                "": {"level": "INFO", "handlers": ["file_handler"]},
                "__main__": {"level": "NOTSET", "propagate": "yes"},
            },
        }
        logging.config.dictConfig(logger_config)

        if mlcube_args.mlcube_task == Task.Train:
            train(task_args)
        else:
            raise ValueError(f"Unknown task: {task_args}")
    except Exception as err:
        logger.exception(err)
        sys.exit(1)


if __name__ == "__main__":
    main()
