import torch
from tcupgan.model import LSTMUNet
from tools.io import BraTSDataGenerator
from tools.utils import NpEncoder, process_mask_tissue_wise
from torchvision import transforms
from torchvision.transforms import Resize
import os
import glob
import tqdm
from einops import rearrange
import numpy as np
import json
import shutil
import yaml

try:
    from brats_val_2023.eval import nib, get_LesionWiseResults
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, './brats_val_2023/')
    from brats_val_2023.eval import nib, get_LesionWiseResults


def run_metrics(parameters_file: str) -> None:
    
    assert parameters_file!='', 'Provide a valid parameters file'
    
    with open(parameters_file, 'r') as param_file:
        parameters = yaml.safe_load(param_file)
    
    data_dir = parameters.get('data_dir')
    ckpt_file = parameters.get('ckpt_file')
    assert (data_dir != '') or (ckpt_file != ''), 'One or more input arguments are blank'
    
    challenge_name = parameters.get('challenge_name')
    assert (challenge_name!=None) or (challenge_name!=''), 'A valid challenge name needs to be provided in the parameters_infer.yaml'
    assert (challenge_name=='GLI') or (challenge_name == 'PED') or (challenge_name == 'MEN') or (challenge_name == 'SSA'), 'Challenge name not in one of the accepted list [GLI, MEN, PED, or SSA], consider picking one of this.'


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
    print(f'Created metrics file {os.path.dirname(data_dir[:-1])}/{outfile_name}')