import torch
from tcupgan.model import LSTMUNet
from tcupgan.disc import PatchDiscriminator
from tcupgan.trainer import TrainerUNet
from tools.io import BraTSDataGenerator
from torchinfo import summary
from torch.utils.data import DataLoader, random_split
import os
import yaml


def run_train(config_file: str) -> None:
    assert config_file!='', 'provide a valid config file'
    if os.path.isfile(config_file):
        with open(config_file, 'r') as infile:
            config = yaml.safe_load(infile)
    else:
        config = {}
    
    data_dir = config.get('data_dir')
    assert (data_dir!='') and (os.path.exists(data_dir)), 'Provide a valid data directory path'
    
    ckpt_folder = config.get('ckpt_folder', './checkpoints')
    run_name = config.get('run_name', 'TCuPGAN')
    batch_size = config.get('batch_size', 1)
    train_val_split = config.get('train_val_split', 0.9)
    start_from_last = config.get('start_from_last', False)
    gen_transfer_learn_ckpt = config.get(
        'generator_transfer_learn_checkpoint', '')
    disc_transfer_learn_ckpt = config.get(
        'discriminator_transfer_learn_checkpoint', '')
    gen_lr = config.get('generator_learning_rate', 1e-3)
    dsc_lr = config.get('discriminator_learning_rate', 1e-4)
    n_epochs = config.get('n_epochs', 50)
    norm = config.get('normalize', True)
    add_noise = config.get('add_noise', True)

    assert not (start_from_last and gen_transfer_learn_ckpt != '' and gen_transfer_learn_ckpt != ''), 'cannot load from last save AND transfer learn from a different model'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = BraTSDataGenerator(data_dir, normalize=norm, add_noise=add_noise)

    assert len(data) > 0, "No files detected!"

    assert train_val_split < 1, "Training/Validation split must be less than 1!"

    torch_rand_gen = torch.Generator().manual_seed(9999)
    train_datagen, val_datagen = random_split(
        data, [train_val_split, 1 - train_val_split], generator=torch_rand_gen)

    train_data = DataLoader(train_datagen, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_data = DataLoader(val_datagen, batch_size=batch_size, shuffle=True, pin_memory=True)

    hidden = [16, 32, 48, 64, 128]
    generator = LSTMUNet(hidden_dims=hidden, input_channels=4,
                         output_channels=3).to(device)

    discriminator = PatchDiscriminator(
        input_channels=7, nlayers=5, nfilt=16, activation='tanh', kernel_size=3).to(device)

    summary(generator, input_size=(1, 155, 4, 256, 256), device=device)
    summary(discriminator, input_size=(1, 155, 7, 256, 256), device=device)

    checkpoint_path = os.path.join(ckpt_folder, run_name)

    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    trainer = TrainerUNet(generator, discriminator, checkpoint_path)
    if start_from_last:
        trainer.load_last_checkpoint()
    if gen_transfer_learn_ckpt != '' and disc_transfer_learn_ckpt != '':
        print('Transfer learning checkpoints provided')
#         generator.load_transfer_data(gen_transfer_learn_ckpt)
#         discriminator.load_transfer_data(disc_transfer_learn_ckpt)
        trainer.load(gen_transfer_learn_ckpt, disc_transfer_learn_ckpt)
        print(
            f'Loaded {gen_transfer_learn_ckpt} \n {disc_transfer_learn_ckpt}')

    gen_learning_rate = gen_lr
    dsc_learning_rate = dsc_lr

    trainer.seg_alpha = 200
    trainer.loss_type = 'weighted_bce'

    trainer.train(train_data, val_data, n_epochs, lr_decay=1,
                  dsc_learning_rate=dsc_learning_rate,
                  gen_learning_rate=gen_learning_rate, save_freq=5)
