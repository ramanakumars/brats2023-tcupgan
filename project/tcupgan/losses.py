import torch
from torch import nn
import cc3d
import scipy
import numpy as np
from copy import deepcopy

def tversky(y_true, y_pred, beta, batch_mean=True):
    tp = torch.sum(y_true * y_pred, axis=(1, 2, 3, 4))
    fn = torch.sum((1. - y_pred) * y_true, axis=(1, 2, 3, 4))
    fp = torch.sum(y_pred * (1. - y_true), axis=(1, 2, 3, 4))
    # tversky = reduce_mean(tp)/(reduce_mean(tp) +
    #                           beta*reduce_mean(fn) +
    #                           (1. - beta)*reduce_mean(fp))
    tversky = tp /\
        (tp + beta * fn + (1. - beta) * fp)

    if batch_mean:
        return torch.mean((1. - tversky))
    else:
        return (1. - tversky)


def kl_loss(mu, sig):
    kl = 0.5 * torch.mean(-1 - sig + torch.square(mu) + torch.exp(sig), axis=-1)
    return torch.mean(kl)


def fc_tversky(y_true, y_pred, beta=0.7, gamma=0.75, batch_mean=True):
    smooth = 1
    tp = torch.sum(y_true * y_pred, axis=(3, 4))
    fn = torch.sum((1. - y_pred) * y_true, axis=(3, 4))
    fp = torch.sum(y_pred * (1. - y_true), axis=(3, 4))
    tversky = (tp + smooth) /\
        (tp + beta * fn + (1. - beta) * fp + smooth)

    focal_tversky_loss = 1 - tversky

    if batch_mean:
        return torch.pow(torch.mean(focal_tversky_loss), gamma)
    else:
        return torch.pow(focal_tversky_loss, gamma)

# def fc_tversky(y_true, y_pred, weights, beta=0.7, gamma=0.75, batch_mean=True):
#     smooth = 1
#     tp = torch.sum(y_true * y_pred, axis=(3, 4))
#     fn = torch.sum((1. - y_pred) * y_true, axis=(3, 4))
#     fp = torch.sum(y_pred * (1. - y_true), axis=(3, 4))
#     tversky = (tp + smooth) /\
#         (tp + beta * fn + (1. - beta) * fp + smooth)

#     focal_tversky_loss = 1 - tversky

#     if batch_mean:
#         loss = torch.pow(torch.mean(focal_tversky_loss, dim=(0,1)), gamma)
#         print(loss.detach().cpu().numpy(), weights.detach().cpu().numpy(), (loss * weights).detach().cpu().numpy())
#         return torch.mean(loss * weights)
#     else:
#         return torch.pow(focal_tversky_loss, gamma)




def focal_loss(y_pred, y_true, gamma=0.75):
    BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(y_pred, y_true)
    pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
    F_loss = (1 - pt)**gamma * BCE_loss

    return torch.mean(F_loss)


def MSE(y_pred, y_true):
    return torch.mean(torch.sum((y_pred - y_true)**2, axis=(1, 2, 3, 4)))


def MAE(y_pred, y_true):
    return torch.mean(torch.sum(torch.abs(y_pred - y_true), axis=(1, 2, 3, 4)))


mink_power = 1


def mink(y_pred, y_true):
    return torch.mean(torch.sum(torch.abs(y_pred - y_true)**mink_power, axis=(1, 2, 3, 4)))  # **(1./mink_power))


def lw_dice(gt_mat, pred_mat):
    dilation_struct = scipy.ndimage.generate_binary_structure(3, 2)

    gt_mat_cc = cc3d.connected_components(gt_mat, connectivity=26)
    pred_mat_cc = cc3d.connected_components(pred_mat, connectivity=26)

    gt_mat_dilation = scipy.ndimage.binary_dilation(gt_mat, structure = dilation_struct, iterations = 3)
    gt_mat_dilation_cc = cc3d.connected_components(gt_mat_dilation, connectivity=26)

    gt_mat_combinedByDilation = get_GTseg_combinedByDilation(gt_dilated_cc_mat = gt_mat_dilation_cc, gt_label_cc = gt_mat_cc)
    
    ## Performing the Lesion-By-Lesion Comparison

    gt_label_cc = gt_mat_combinedByDilation
    pred_label_cc = pred_mat_cc

    gt_tp = []
    tp = []
    fn = []
    fp = []
    metric_pairs = []
    
    for gtcomp in range(np.max(gt_label_cc)):
        gtcomp += 1

        ## Extracting current lesion
        gt_tmp = np.zeros_like(gt_label_cc)
        gt_tmp[gt_label_cc == gtcomp] = 1

        ## Extracting ROI GT lesion component
        gt_tmp_dilation = scipy.ndimage.binary_dilation(gt_tmp, structure = dilation_struct, iterations = 3)
        
        gt_vol = np.sum(gt_tmp)
        
        ## Extracting Predicted true positive lesions
        pred_tmp = np.copy(pred_label_cc)
        #pred_tmp = pred_tmp*gt_tmp
        pred_tmp = pred_tmp*gt_tmp_dilation
        intersecting_cc = np.unique(pred_tmp) 
        intersecting_cc = intersecting_cc[intersecting_cc != 0] 
        for cc in intersecting_cc:
            tp.append(cc)

        ## Isolating Predited Lesions to calulcate Metrics
        pred_tmp = np.copy(pred_label_cc)
        pred_tmp[np.isin(pred_tmp,intersecting_cc,invert=True)] = 0
        pred_tmp[np.isin(pred_tmp,intersecting_cc)] = 1

        ## Calculating Lesion-wise Dice and HD95
        dice_score = calc_dice(pred_tmp, gt_tmp)
        metric_pairs.append([len(intersecting_cc), 
                            gtcomp, gt_vol, dice_score])
        
#         print(gtcomp, gt_vol,dice_score)
#         print(intersecting_cc)
#         print('----')
        
        
        ## Extracting Number of TP/FP/FN and other data
        if len(intersecting_cc) > 0:
            gt_tp.append(gtcomp)
        else:
            fn.append(gtcomp)

    fp = np.unique(
            pred_label_cc[np.isin(
                pred_label_cc,tp+[0],invert=True)])
    
    return np.array(metric_pairs), len(fp)

def lw_dice_loss(gt_mask,pred_mask, weights):
    labels = ['WT', 'TC', 'ET']
    scores = []
    for label in labels:
        copied_gt_mask = deepcopy(gt_mask)
        copied_pred_mask = deepcopy(pred_mask)
        indiv_pred_mask, indiv_gt_mask = get_TissueWiseSeg(copied_pred_mask, copied_gt_mask, label)
        metrics, fp = lw_dice(indiv_gt_mask, indiv_pred_mask)
#         print(metrics)
        try:
            scores.append(np.sum(metrics[:,3])/(len(metrics[:,0]) + fp))
        except:
            scores.append([0])
    
    try:
        loss = (1 - np.array(scores)) * np.array(weights)
    except:
        loss = 0
    
    return loss


def calc_dice(im1, im2):
    """
    Computes Dice score for two images

    Parameters
    ==========
    im1: Numpy Array/Matrix; Predicted segmentation in matrix form 
    im2: Numpy Array/Matrix; Ground truth segmentation in matrix form

    Output
    ======
    dice_score: Dice score between two images
    """

    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * (intersection.sum()) / (im1.sum() + im2.sum())



def get_TissueWiseSeg(prediction_matrix, gt_matrix, tissue_type):
    """
    Converts the segmentatations to isolate tissue types

    Parameters
    ==========
    prediction_matrix: Numpy Array/Matrix; Predicted segmentation in matrix form 
    gt_matrix: Numpy Array/Matrix; Ground truth segmentation in matrix form
    tissue_type: str; Can be WT, ET or TC

    Output
    ======
    prediction_matrix: Numpy Array/Matrix; Predicted segmentation in matrix form with 
                       just tissue type mentioned
    gt_matrix: Numpy Array/Matrix; Ground truth segmentation in matrix form with just 
                       tissue type mentioned
    """

    if tissue_type == 'WT':
        np.place(prediction_matrix, (prediction_matrix != 1) & (prediction_matrix != 2) & (prediction_matrix != 3), 0)
        np.place(prediction_matrix, (prediction_matrix > 0), 1)
        
        np.place(gt_matrix, (gt_matrix != 1) & (gt_matrix != 2) & (gt_matrix != 3), 0)
        np.place(gt_matrix, (gt_matrix > 0), 1)
    
    elif tissue_type == 'TC':
        np.place(prediction_matrix, (prediction_matrix != 1)  & (prediction_matrix != 3), 0)
        np.place(prediction_matrix, (prediction_matrix > 0), 1)
        
        np.place(gt_matrix, (gt_matrix != 1) & (gt_matrix != 3), 0)
        np.place(gt_matrix, (gt_matrix > 0), 1)
        
    elif tissue_type == 'ET':
        np.place(prediction_matrix, (prediction_matrix != 3), 0)
        np.place(prediction_matrix, (prediction_matrix > 0), 1)
        
        np.place(gt_matrix, (gt_matrix != 3), 0)
        np.place(gt_matrix, (gt_matrix > 0), 1)
    
    return prediction_matrix, gt_matrix


def get_GTseg_combinedByDilation(gt_dilated_cc_mat, gt_label_cc):
    """
    Computes the Corrected Connected Components after combing lesions
    together with respect to their dilation extent

    Parameters
    ==========
    gt_dilated_cc_mat: Numpy Array/Matrix; Ground Truth Dilated Segmentation 
                       after CC Analysis
    gt_label_cc: Numpy Array/Matrix; Ground Truth Segmentation after 
                       CC Analysis

    Output
    ======
    gt_seg_combinedByDilation_mat: Numpy Array/Matrix; Ground Truth 
                                   Segmentation after CC Analysis and 
                                   combining lesions
    """    
    
    
    gt_seg_combinedByDilation_mat = np.zeros_like(gt_dilated_cc_mat)

    for comp in range(np.max(gt_dilated_cc_mat)):  
        comp += 1

        gt_d_tmp = np.zeros_like(gt_dilated_cc_mat)
        gt_d_tmp[gt_dilated_cc_mat == comp] = 1
        gt_d_tmp = (gt_label_cc*gt_d_tmp)

        np.place(gt_d_tmp, gt_d_tmp > 0, comp)
        gt_seg_combinedByDilation_mat += gt_d_tmp
        
    return gt_seg_combinedByDilation_mat
# alias
adv_loss = nn.BCELoss()

ce_loss = nn.CrossEntropyLoss()  # mink
