import os
import sys
path = os.path.dirname(os.path.abspath(""))+"/"
sys.path.append(path)
sys.path.insert(1, path+'cellpose/')
print(path)
from skimage import io
from skimage import color
from skimage import segmentation
from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
from CellAnalysis.utils import *
from CellAnalysis import visualize
from CellAnalysis import evaluation
import pandas as pd
from mAP_3Dvolume import vol3d_eval, vol3d_util
import mAP_3Dvolume as meanap


def get_scores(pred_seg, gt_seg):
    sz_gt = np.array(gt_seg.shape)
    sz_pred = pred_seg.shape
    if np.abs((sz_gt - sz_pred)).max() > 0:
        print('Warning: size mismatch. gt: ', sz_gt, ', pred: ', sz_pred)
    sz = np.minimum(sz_gt, sz_pred)
    pred_seg = pred_seg[:sz[0], :sz[1]]
    gt_seg = gt_seg[:sz[0], :sz[1]]

    ui, uc = np.unique(pred_seg, return_counts=True)
    uc = uc[ui > 0]
    ui = ui[ui > 0]
    pred_score = np.ones([len(ui), 2], int)
    pred_score[:, 0] = ui
    pred_score[:, 1] = uc

    thres = np.fromstring('5e3, 1.5e4', sep=",")
    areaRng = np.zeros((len(thres) + 2, 2), int)
    areaRng[0, 1] = 1e10
    areaRng[-1, 1] = 1e10
    areaRng[2:, 0] = thres
    areaRng[1:-1, 1] = thres

    return pred_score, areaRng


def get_precision(pred, gt):
    pred_score, areaRng = get_scores(pred, gt)
    result_p, result_fn, pred_score_sorted = meanap.vol3d_util.seg_iou3d_sorted(pred, gt, pred_score, areaRng)
    v3dEval = meanap.vol3d_eval.VOL3Deval(result_p, result_fn, pred_score_sorted, output_name='map_output')
    stats = v3dEval.get_stats()
    return stats


# file root
file_root = path + 'data/EM/'
pc_pred_em = io.imread(file_root + 'pc_prediction/cells_em.tif')
sd_pred_em = io.imread(file_root + 'stardist_prediction/EM_multiSEM_thumbnails-cropped_iso_SD.tif').astype(np.uint16)
vol_em = io.imread(file_root + 'EM_multiSEM_thumbnails-cropped_iso.tif')
test_gt_em = io.imread(file_root + 'testing/box7_gt.tif').astype(np.uint16)
cp_cutoff = io.imread(file_root + 'cellpose_prediction/EM_multiSEM_thumbnails-cropped_iso_CP_cutOff.tif')
cp = io.imread(file_root + 'cellpose_prediction/EM_multiSEM_thumbnails-cropped_iso_CP.tif')
test_old_pred_em = io.imread(file_root + 'testing/box7_CP.tif')
test_img_em = io.imread(file_root + 'testing/box7_vol.tif').astype(np.uint8)
#test_old_pyc_pred = np.moveaxis(dataloader(file_root + 'pc_prediction/pred_pyc.h5')['main'], -1, 1)
print(pc_pred_em.shape)
print(sd_pred_em.shape)
#pc_pred = np.moveaxis(pc_pred, -1, 0) #move z-axis to the end
print(vol_em.shape)
print(test_gt_em.shape)
print(test_img_em.shape)
print(test_old_pred_em.shape)
#print(test_old_pyc_pred.shape)
print(type(cp[1,1,1]))
row_start = 320
col_start = 320
slice_start = 140
width = 25
test_pred_pc_em = pc_pred_em[slice_start : slice_start + width, row_start : row_start + width, col_start: col_start + width]
test_pred_sd_em = sd_pred_em[slice_start : slice_start + width, row_start : row_start + width, col_start: col_start + width]
test_cp_cutoff_em = cp_cutoff[slice_start : slice_start + width, row_start : row_start + width, col_start: col_start + width]
test_cp_em = cp[slice_start : slice_start + width, row_start : row_start + width, col_start: col_start + width]
test_vol_em = vol_em[slice_start : slice_start + width, row_start : row_start + width, col_start: col_start + width]
print(test_pred_pc_em.shape)
print(test_pred_sd_em.shape)
print(test_vol_em.shape)

print(test_pred_pc_em.dtype)
print(test_pred_sd_em.dtype)
print(test_cp_cutoff_em.dtype)
print(test_cp_em.dtype)
print(test_gt_em.dtype)
stats_em_pc = get_precision(test_cp_em[0], test_gt_em[0])
recall = stats_em_pc['Recall'][:,0]
precision = stats_em_pc['Precision'][:,0]

fig = plt.figure()
plt.plot(recall, precision)
plt.show()
