import torch
from tcupgan.model import LSTMUNet
from tcupgan.trainer import TrainerUNet
from tcupgan.io import BraTSDataGenerator
from torchvision import transforms
from torchvision.transforms import Resize
from tcupgan.utils import process_mask_tissue_wise
import os, glob
import tqdm
import nibabel as nib
from einops import rearrange
import numpy as np
import scipy
import cc3d

def run_inference(data_dir: str, challenge_name:str, ckpt_file: str, output_dir: str) -> None:
    assert (data_dir!='') or (ckpt_file!='') or (output_dir!=''), 'One or more input arguments are blank'
    
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
    
    if device == 'cpu':
        generator.load_state_dict(torch.load(f'{ckpt_file}',map_location=torch.device(device)))
        print('Loaded Model Weights successfully!')
    else:
        generator.load_state_dict(torch.load(f'{ckpt_file}'))
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
    
    counter = 1
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
        counter+=1
        

