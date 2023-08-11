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
from tcupgan.utils import process_mask_tissue_wise
import nibabel as nib
from einops import rearrange
import numpy as np
import scipy
import cc3d

logger = logging.getLogger(__name__)


class Task(str, Enum):
    """Define tasks"""

    DownloadData = "download"
    Train = "train"
    Evaluate = "infer"
    Metrics = "metrics"


def train(task_args: List[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '--data-dir', type=str, required=True)
    parser.add_argument('--config_file', type=str)
    args = parser.parse_args(args=task_args)

    if os.path.isfile(config_file):
        with open(config_file, 'r') as infile:
            config = yaml.safe_load(infile)
    else:
        config = {}

    ckpt_folder = config.get('checkpoint_folder', 'checkpoints/')
    run_name = config.get('run_name', 'TCuPGAN')
    batch_size = config.get('batch_size', 4)
    train_val_split = config.get('train_val_split', 0.9)
    start_from_last = config.get('start_from_last', False)
    gen_transfer_learn_ckpt = config.get('generator_transfer_learn_checkpoint', '')
    disc_transfer_learn_ckpt = config.get('discriminator_transfer_learn_checkpoint', '')
    gen_lr = config.get('generator_learning_rate', 1e-3)
    gen_lr_decay = config.get('gen_lr_decay', 0.95)
    dsc_lr = config.get('discriminator_learning_rate', 1e-4)
    n_epochs = config.get('n_epochs', 50)
    norm = config.get('normalize', True)
    add_noise = config.get('add_noise', True)

    assert not (start_from_last and gen_transfer_learn_ckpt != '' and gen_transfer_learn_ckpt !=''), 'cannot load from last save AND transfer learn from a different model'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = BraTSDataGenerator(data_dir, normalize=norm, add_noise=add_noise)

    assert len(data) > 0, "No files detected!"

    assert train_val_split < 1, "Training/Validation split must be less than 1!"
    
    torch_rand_gen = torch.Generator().manual_seed(9999)
    train_datagen, val_datagen = random_split(data, [train_val_split, 1 - train_val_split], generator=torch_rand_gen)

    train_data = DataLoader(train_datagen, num_workers=8, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_data = DataLoader(val_datagen,num_workers=8, batch_size=batch_size, shuffle=True, pin_memory=True)

    hidden = [16, 32, 48, 64, 128]
    generator = LSTMUNet(hidden_dims=hidden, input_channels=4, output_channels=3).to(device)

    discriminator = PatchDiscriminator(input_channels=7, nlayers=5, nfilt=16, activation='tanh', kernel_size=3).to(device)

    summary(generator, input_size=(1, 155, 4, 256, 256), device=device)
    summary(discriminator, input_size=(1, 155, 7, 256, 256), device=device)

    checkpoint_path = os.path.join(ckpt_folder, run_name)
    
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    trainer = TrainerUNet(generator, discriminator, checkpoint_path)
    if start_from_last:
        trainer.load_last_checkpoint()
    if gen_transfer_learn_ckpt != '' and disc_transfer_learn_ckpt !='':
        print('Transfer learning checkpoints provided')
#         generator.load_transfer_data(gen_transfer_learn_ckpt)
#         discriminator.load_transfer_data(disc_transfer_learn_ckpt)
        trainer.load(gen_transfer_learn_ckpt, disc_transfer_learn_ckpt)
        print(f'Loaded {gen_transfer_learn_ckpt} \n {disc_transfer_learn_ckpt}')

    gen_learning_rate = gen_lr
    dsc_learning_rate = dsc_lr
    
    trainer.seg_alpha = 200
    trainer.loss_type = 'weigthed_bce'

    trainer.train(train_data, val_data, n_epochs, lr_decay=gen_lr_decay,
                  dsc_learning_rate=dsc_learning_rate,
                  gen_learning_rate=gen_learning_rate, save_freq=5)


def infer(task_args: List[str]) -> None:
    parser.add_argument('--data_dir', '--data-dir', type=str, required=True)
    parser.add_argument('--challenge_name', type=str, required = True)
    parser.add_argument('--ckpt_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args(args=task_args)
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        print('Made missing output directory')
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    list_of_case_ids = ['-'.join(os.path.basename(folder).split('-')[2:4]) for folder in glob.glob('%s/*'%(data_dir))]
    print(f'There are {len(list_of_case_ids)} case folders in the {data_dir} folder')
    
    data = BraTSDataGenerator(data_dir)
    assert len(data) > 0, "No files detected!"
    
    hidden = [16, 32, 48, 64, 128]
    generator = LSTMUNet(hidden_dims=hidden, input_channels=4, output_channels=3).to(device)
    generator.eval()
    
    generator.load_state_dict(torch.load(f'{ckpt_file}', map_location=torch.device(device)))
    print('Loaded Model Weights successfully!')

    resize_fn = Resize((240, 240), interpolation = transforms.InterpolationMode.NEAREST)
    affine_matrix = np.array([[ -1.,  -0.,  -0.,   0.],[ -0.,  -1.,  -0., 239.],[  0.,   0.,   1.,   0.],[  0.,   0.,   0.,   1.]])
    
    if challenge_name == 'PED':
        lesion_threshs = [75, 75, 25]
        lesion_len_thresh = [5,5,5]
    elif challenge_name == 'GLI':
        lesion_threshs = [125, 75, 20]
        lesion_len_thresh = [5,5,5]
    elif challenge_name == 'MEN':
        lesion_threshs = [125, 125, 25]
        lesion_len_thresh = [5,5,5]
    elif challenge_name == 'SSA':
        lesion_threshs = [75, 100, 5]
        lesion_len_thresh = [5,5,5]
    
    for each_sample in tqdm.tqdm(list_of_case_ids):
        ID, timepoint = each_sample.split('-')
        mri_cube, mask = data.get_from_ID(int(ID), int(timepoint))
        with torch.no_grad():
            predicted_mask = torch.squeeze(generator(torch.unsqueeze(torch.Tensor(mri_cube).to(device), 0)),0).cpu().numpy()
        
        background = 1-torch.sum(torch.Tensor(predicted_mask), axis=1, keepdim=True)
        combined_pred = np.concatenate([background.cpu().numpy(), predicted_mask], axis=1)
        normed_combined_pred = torch.Tensor(combined_pred)/torch.sum(torch.Tensor(combined_pred), axis=(1), keepdim=True)
        
        combined_pred_mask = rearrange(resize_fn(torch.argmax(torch.Tensor(normed_combined_pred), dim=1)), 'z h w -> h w z' ).cpu().numpy()
        
        processed_mask = process_mask_tissue_wise(combined_pred_mask, mean_lesion_span_thresh=lesion_threshs, lesion_span_len_thresh=lesion_len_thresh)
        
        nii_file = nib.Nifti1Image(processed_mask, affine=affine_matrix, dtype='uint8')
        nib.save(nii_file, f'{output_dir}/BraTS-{each_sample}.nii.gz')

    

def metrics(task_args: List[str]) -> None:
    parser.add_argument('--data_dir', '--data-dir', type=str, required=True)
    parser.add_argument('--ckpt_file', type=str, required=True)
    parser.add_argument('--challenge_name', type=str, required = True)
    args = parser.parse_args(args=task_args)
    
    assert (data_dir!='') or (ckpt_file!=''), 'One or more input arguments are blank'
    
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    list_of_case_ids = ['-'.join(os.path.basename(folder).split('-')[2:4]) for folder in glob.glob('%s/*'%(data_dir))]
    print(f'There are {len(list_of_case_ids)} case folders in the {data_dir} folder')
    
    data = BraTSDataGenerator(data_dir)
    assert len(data) > 0, "No files detected!"
    
    hidden = [16, 32, 48, 64, 128]
    generator = LSTMUNet(hidden_dims=hidden, input_channels=4, output_channels=3).to(device)

    generator.eval()

    generator.load_state_dict(torch.load(ckpt_file, map_location=torch.device(device)))

    resize_fn = Resize((240, 240), interpolation = transforms.InterpolationMode.NEAREST)
    affine_matrix = np.array([[ -1.,  -0.,  -0.,   0.],[ -0.,  -1.,  -0., 239.],[  0.,   0.,   1.,   0.],[  0.,   0.,   0.,   1.]])
    
    if challenge_name == 'PED':
        lesion_threshs = [75, 75, 25]
        lesion_len_thresh = [5,5,5]
    elif challenge_name == 'GLI':
        lesion_threshs = [125, 75, 20]
        lesion_len_thresh = [5,5,5]
    elif challenge_name == 'MEN':
        lesion_threshs = [125, 125, 25]
        lesion_len_thresh = [5,5,5]
    elif challenge_name == 'SSA':
        lesion_threshs = [75, 100, 5]
        lesion_len_thresh = [5,5,5]
    
    if not os.path.exists(f'./tmp_{challenge_name}'):
        os.mkdir(f'./tmp_{challenge_name}')
    
    metrics_information = {}
    for each_file in tqdm.tqdm(data.IDs):
        ID, timepoint = each_file.split('-')
        image, label = data.get_from_ID(int(ID),int(timepoint))
        assert label!=None, 'ground truth mask should be present for metrics to be computed'
        
        with torch.no_grad():
            pred = torch.squeeze(generator(torch.unsqueeze(image.to(device), 0)), 0).cpu().numpy()
        
        background = 1-torch.sum(torch.Tensor(pred), axis=1, keepdim=True)
        combined_pred = np.concatenate([background.cpu().numpy(), pred], axis=1)
        normed_combined_pred = torch.Tensor(combined_pred)/torch.sum(torch.Tensor(combined_pred), axis=(1), keepdim=True)
        combined_pred = rearrange(resize_fn(torch.argmax(torch.Tensor(normed_combined_pred), dim=1)), 'z h w -> h w z' ).cpu().numpy()

        processed_mask = process_mask_tissue_wise(combined_pred, mean_lesion_span_thresh=lesion_threshs, lesion_span_len_thresh=[5,5,5])
        nii_file = nib.Nifti1Image(processed_mask, affine=affine_matrix, dtype='uint8')
        nib.save(nii_file, f'./tmp_{challenge_name}/BraTS-{each_file}.nii.gz')
        
        original_mask = f'{data_dir}/BraTS-{challenge_name}-{each_file}/BraTS-{challenge_name}-{each_file}-seg.nii.gz'
        results = get_LesionWiseResults(f'./tmp_{challenge_name}/BraTS-{each_file}.nii.gz', original_mask, challenge_name=f'BraTS-{challenge_name}').to_dict()
    
        metrics_information[each_file] = results
    
    shutil.rmtree(f'./tmp_{challenge_name}')
        
    if data_dir.endswith('/'):
        outfile_name = f'{os.path.basename(data_dir[:-1])}-metrics-{challenge_name}.json'
    else:
        outfile_name = f'{os.path.basename(data_dir)}-metrics-{challenge_name}.json'
    with open(f'{os.path.dirname(data_dir[:-1])}/{outfile_name}', 'w') as outJSON:
        json.dump(metrics_information, outJSON, cls=NpEncoder)
    print(f'{os.path.dirname(data_dir[:-1])}/{outfile_name} has been created!')
    
    
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
        elif mlcube_args.mlcube_task == Task.Evaluate:
            infer(task_args)
        elif mlcube_args.mlcube_task == Task.Metrics:
            metrics(task_args)
        else:
            raise ValueError(f"Unknown task: {task_args}")
    except Exception as err:
        logger.exception(err)
        sys.exit(1)


if __name__ == "__main__":
    main()
