import json
import numpy as np
import cc3d
import scipy
from copy import deepcopy

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    

    
def process_mask(input_mask, challenge_name):
    if challenge_name == 'GLI':
        mean_lesion_span_thresh = [110, 45, 30]
        lesion_span_len_thresh = [5, 5, 5]
    elif challenge_name == 'MEN':
        mean_lesion_span_thresh = [55, 55, 55]
        lesion_span_len_thresh = [5, 5, 5]
    
    processed_mask = np.zeros_like(input_mask)
    connected_components = cc3d.connected_components(input_mask, connectivity=26)
    for each in np.unique(connected_components)[1:]:
        condition = np.where(connected_components==each)
        lesion_spans = []
        label_in_mask = np.unique(input_mask[condition]).item()
        for each_slice in np.unique(np.where(connected_components==each)[2]):
            lesion_spans.append(len(np.where(connected_components[:,:,each_slice]==each)[0]))
        if (label_in_mask ==1) and (np.mean(lesion_spans)>=mean_lesion_span_thresh[0] and len(lesion_spans)>=lesion_span_len_thresh[0]):
            processed_mask[condition] = input_mask[condition]
        elif label_in_mask == 2 and (np.mean(lesion_spans)>=mean_lesion_span_thresh[1] and len(lesion_spans)>=lesion_span_len_thresh[1]):
            processed_mask[condition] = input_mask[condition]
        elif label_in_mask == 3 and (np.mean(lesion_spans)>=mean_lesion_span_thresh[2] and len(lesion_spans)>=lesion_span_len_thresh[2]):
            processed_mask[condition] = input_mask[condition]
    
    return processed_mask

def process_mask_var(input_mask, mean_lesion_span_thresh = [50, 50, 50], lesion_span_len_thresh = [5, 5, 5] ):
    processed_mask = np.zeros_like(input_mask)
#     print('Started cc')
    connected_components = cc3d.connected_components(input_mask, connectivity=26)
#     print('Done cc')
#     print(np.unique(connected_components)[1:])
    for each in np.unique(connected_components)[1:]:
        condition = np.where(connected_components==each)
        lesion_spans = []
        label_in_mask = np.unique(input_mask[condition]).item()
        for each_slice in np.unique(np.where(connected_components==each)[2]):
            lesion_spans.append(len(np.where(connected_components[:,:,each_slice]==each)[0]))
        if (label_in_mask ==1) and (np.mean(lesion_spans)>=mean_lesion_span_thresh[0] and len(lesion_spans)>=lesion_span_len_thresh[0]):
            processed_mask[condition] = input_mask[condition]
        elif label_in_mask == 2 and (np.mean(lesion_spans)>=mean_lesion_span_thresh[1] and len(lesion_spans)>=lesion_span_len_thresh[1]):
            processed_mask[condition] = input_mask[condition]
        elif label_in_mask == 3 and (np.mean(lesion_spans)>=mean_lesion_span_thresh[2] and len(lesion_spans)>=lesion_span_len_thresh[2]):
            processed_mask[condition] = input_mask[condition]
#         print(each)
    return processed_mask

def get_tissue_split(input_array, tissue_type):
    array = deepcopy(input_array)
    if tissue_type == 'WT':
        np.place(array, (array != 1) & (array != 2) & (array != 3), 0)
        np.place(array, (array > 0), 1)
    
    elif tissue_type == 'TC':
        np.place(array, (array != 1)  & (array != 3), 0)
        np.place(array, (array > 0), 1)
        
    elif tissue_type == 'ET':
        np.place(array, (array != 3), 0)
        np.place(array, (array > 0), 1)
    
    return array

def combine_tissue_segs(tissue_wise_array):
    wt, tc, et = tissue_wise_array
    bkg = np.zeros_like(wt)
    combined_array = np.zeros(shape=(4, 240, 240, 155))
    class1 = np.logical_and(tc, np.logical_not(et))   #tc and not(et)
    class2 = np.logical_and(wt, np.logical_not(tc))#wt and not(tc)
    class3 = et
    
    combined_array[0,:,:,:,] = bkg
    combined_array[1,:,:,:,] = class1
    combined_array[2,:,:,:,] = class2
    combined_array[3,:,:,:,] = class3
#     print(combined_array.shape)
    
    merged_array = np.argmax(combined_array, axis=0)
    return merged_array


def process_mask_tissue_wise(input_array, mean_lesion_span_thresh, lesion_span_len_thresh):
    combined_pred_mask = deepcopy(input_array)
    tissue_wise_splits =[]
    for each_tissue_type, area_thresh, slice_num_thresh in zip(['WT', 'TC', 'ET'], mean_lesion_span_thresh,lesion_span_len_thresh):
        tissue_array = get_tissue_split(combined_pred_mask, each_tissue_type)
        tissue_array_cc = cc3d.connected_components(tissue_array, connectivity=26)
#         print(f'{each_tissue_type}, {np.unique(tissue_array)}, {np.unique(tissue_array_cc)}')
        processed_tissue_array = np.zeros_like(tissue_array)
        for each_cc in np.unique(tissue_array_cc)[1:]:
            condition = np.where(tissue_array_cc==each_cc)
            lesion_spans = []
            for each_slice in np.unique(np.where(tissue_array_cc==each_cc)[2]):
                lesion_spans.append(len(np.where(tissue_array_cc[:,:,each_slice]==each_cc)[0]))
    #         print(lesion_spans)
#             print(np.mean(lesion_spans), len(lesion_spans))
            if np.mean(lesion_spans)>=area_thresh and len(lesion_spans)>=slice_num_thresh:
                processed_tissue_array[condition] = tissue_array[condition]
        tissue_wise_splits.append(processed_tissue_array)
        processed_cc = cc3d.connected_components(processed_tissue_array, connectivity=26)
#         print(f'processed cc {np.unique(processed_cc)}')
#         print('-----')
    recombined_pred_mask = combine_tissue_segs(np.array(tissue_wise_splits)) 
    return recombined_pred_mask