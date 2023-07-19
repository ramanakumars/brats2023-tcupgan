import torch
from tcupgan.model import LSTMUNet
from tcupgan.disc import PatchDiscriminator
from tcupgan.trainer import TrainerUNet
from tcupgan.io import BraTSDataGenerator
from torchinfo import summary
from torch.utils.data import DataLoader, random_split
import os
import yaml

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def run_train(data_dir: str, config_file: str, ckpt_folder: str) -> None:
    if os.path.isfile(config_file):
        with open(config_file, 'r') as infile:
            config = yaml.safe_load(infile)
    else:
        config = {}

    run_name = config.get('run_name', 'TCuPGAN')
    batch_size = config.get('batch_size', 2)
    train_val_split = config.get('train_val_split', 0.9)
    start_from_last = config.get('start_from_last', False)
    transfer_learn_ckpt = config.get('transfer_learn_checkpoint', '')
    gen_lr = config.get('generator_learning_rate', 1e-3)
    dsc_lr = config.get('discriminator_learning_rate', 1e-4)

    assert not (start_from_last and transfer_learn_ckpt != ''), 'cannot load from last save AND transfer learn from a different model'

    print(f"Using device: {device}")

    data = BraTSDataGenerator(data_dir)

    assert len(data) > 0, "No files detected!"

    assert train_val_split < 1, "Training/Validation split must be less than 1!"

    train_datagen, val_datagen = random_split(data, [train_val_split, 1 - train_val_split])

    train_data = DataLoader(train_datagen, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_data = DataLoader(val_datagen, batch_size=batch_size, shuffle=True, pin_memory=True)

    hidden = [6, 12, 18, 64, 128]
    generator = LSTMUNet(hidden_dims=hidden, input_channels=4, output_channels=4).to(device)

    discriminator = PatchDiscriminator(input_channels=8, nlayers=4, nfilt=16).to(device)

    summary(generator, input_size=(1, 155, 4, 128, 128), device=device)
    summary(discriminator, input_size=(1, 155, 8, 128, 128), device=device)

    checkpoint_path = os.path.join(ckpt_folder, run_name)
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    trainer = TrainerUNet(generator, discriminator, checkpoint_path)
    if start_from_last:
        trainer.load_last_checkpoint()
    if transfer_learn_ckpt != '':
        trainer.load_transfer_data(transfer_learn_ckpt)

    gen_learning_rate = gen_lr
    dsc_learning_rate = dsc_lr

    trainer.seg_alpha = 200
    trainer.loss_type = 'tversky'

    trainer.train(train_data, val_data, 50, lr_decay=0.95,
                  dsc_learning_rate=dsc_learning_rate,
                  gen_learning_rate=gen_learning_rate, save_freq=5)
