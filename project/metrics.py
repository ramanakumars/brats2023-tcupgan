import torch
from tcupgan.model import LSTMUNet
from io import BraTSDataGenerator
from utils import NpEncoder
from torchvision import transforms
from torchvision.transforms import Resize
import os
import glob
import tqdm
from einops import rearrange
import numpy as np
import json
from tcupgan.utils import process_mask_tissue_wise
from brats_val_2023.eval import nib, get_LesionWiseResults
import shutil


def run_metrics(data_dir: str, ckpt_file: str, challenge_name: str) -> None:
    assert (data_dir != '') or (ckpt_file != ''), 'One or more input arguments are blank'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    list_of_case_ids = ['-'.join(os.path.basename(folder).split('-')[2:4]) for folder in glob.glob('%s/*' % (data_dir))]
    print(f'There are {len(list_of_case_ids)} case folders in the {data_dir} folder')

    data = BraTSDataGenerator(data_dir)
    assert len(data) > 0, "No files detected!"

    hidden = [16, 32, 48, 64, 128]
    generator = LSTMUNet(hidden_dims=hidden, input_channels=4, output_channels=3).to(device)

    generator.eval()

    generator.load_state_dict(torch.load(ckpt_file, map_location=torch.device(device)))

    resize_fn = Resize((240, 240), interpolation=transforms.InterpolationMode.NEAREST)
    affine_matrix = np.array([[-1., -0., -0., 0.], [-0., -1., -0., 239.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

    if challenge_name == 'PED':
        lesion_threshs = [75, 75, 25]
    elif challenge_name == 'GLI':
        lesion_threshs = [125, 75, 20]
    elif challenge_name == 'MEN':
        lesion_threshs = [125, 125, 25]
    elif challenge_name == 'SSA':
        lesion_threshs = [75, 100, 5]

    if not os.path.exists(f'./tmp_{challenge_name}'):
        os.mkdir(f'./tmp_{challenge_name}')

    metrics_information = {}
    for each_file in tqdm.tqdm(data.IDs):
        ID, timepoint = each_file.split('-')
        image, label = data.get_from_ID(int(ID), int(timepoint))
        assert label is not None, 'ground truth mask should be present for metrics to be computed'

        with torch.no_grad():
            pred = torch.squeeze(generator(torch.unsqueeze(image.to(device), 0)), 0).cpu().numpy()

        background = 1 - torch.sum(torch.Tensor(pred), axis=1, keepdim=True)
        combined_pred = np.concatenate([background.cpu().numpy(), pred], axis=1)
        normed_combined_pred = torch.Tensor(combined_pred) / torch.sum(torch.Tensor(combined_pred), axis=(1), keepdim=True)
        combined_pred = rearrange(resize_fn(torch.argmax(torch.Tensor(normed_combined_pred), dim=1)), 'z h w -> h w z').cpu().numpy()

        processed_mask = process_mask_tissue_wise(combined_pred, mean_lesion_span_thresh=lesion_threshs, lesion_span_len_thresh=[5, 5, 5])
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


# run_metrics('/home/fortson/manth145/data/BraTS2021_Seg/PED/ASNR-MICCAI-BraTS2023-PED-Challenge-InternalValidationData/', '/home/fortson/manth145/codes/BraTS/mlcube_v2/TCuPGAN/checkpoints/ALL_BCE_2.3M/generator_ep_030.pth', 'PED')
