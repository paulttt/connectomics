import numpy as np
import glob
from skimage import measure
import pandas as pd
import tifffile
import math
import pdb

def iou(pred, gt):
    intersection = np.logical_and(pred, gt).astype(np.float32).sum()
    union = np.logical_or(pred, gt).astype(np.float32).sum()
    if union == 0:
        iou = 0
    else:
        iou = intersection / union
    return iou

def add_margin_to_bbox(bbox, margin, shape):
    temp = math.floor(bbox.shape[0] / 2)
    for i in range(math.floor(bbox.shape[0] / 2)):
        bbox[i] -= margin
        if bbox[i] < 0:
            bbox[i] = 0
    for i in range(math.floor(bbox.shape[0] / 2), bbox.shape[0]):
        bbox[i] += margin
        if bbox[i] >= shape[i - temp]:
            bbox[i] = shape[i - temp] - 1
    return bbox

def find_candidates(gt, pred, gt_bbox, df_pred, margin=0):
    gt_bbox = add_margin_to_bbox(gt_bbox, margin, pred.shape)
    pred_subvol = crop_volume(pred, gt_bbox)
    values = np.unique(pred_subvol)
    pred_labels = np.delete(values, np.where(values == 0))
    df_pred_cand = df_pred.loc[df_pred['label'].isin(pred_labels)]
    #df_pred_cand = df_pred.loc[[i for i in pred_labels]]
    return df_pred_cand, pred_labels

def find_centroid_matches(centroid, df_segments, thresh):
    label_match = []
    distances = []
    df_closest = pd.DataFrame(columns=df_segments.columns)
    smallest_distance = 200000.0
    closest_idx = -1
    for idx, row in df_segments.iterrows():
        dist = np.linalg.norm(centroid - row['centroid'])
        distances.append(dist)
        if dist <= thresh:
            label_match.append(idx)
            if smallest_distance <= dist:
                closest_idx = idx
    if closest_idx >= 0:
        df_closest = df.closest.append(df_segments.iloc[closest_idx])
    df_match = df_segments.loc[label_match]
    return df_match, label_match, df_closest, distances

def evaluate_overlap(gt, pred, gt_label, pred_label, metric='IoU'):
    gt_bin = binarize_segment(gt, gt_label)
    pred_bin = binarize_segment(pred, pred_label)
    return iou(gt_bin, pred_bin)

def crop_volume(vol, bbox):
    bbox = bbox.astype(int)
    row_start = bbox[0]
    row_end = bbox[3]
    col_start = bbox[1]
    col_end = bbox[4]
    slice_start = bbox[2]
    slice_end = bbox[5]
    return vol[row_start:row_end, col_start:col_end, slice_start:slice_end]

def get_convex_bbox(bbox_1, bbox_2):
    """
    computes the convex bounding box out of two given bounding boxes.

    Attributes
    ----------
    bbox_1 : 1D np.array
        expected to be of type [row_start, col_start, slice_start, row_end, col_end, slice_end]
    bbox_2 : 1D np.array
        expected to be of type [row_start, col_start, slice_start, row_end, col_end, slice_end]
    return
    -------
    convex_bbox: 1D array of same shape as input
    """
    assert len(bbox_1.shape) == 1 & len(bbox_2.shape) == 1, print("Dimensionality must be of shape 1.")
    assert bbox_1.shape == bbox_2.shape, print("bbox must be of same shapes.")
    convex_bbox = np.zeros(bbox_1.shape)
    # for i in range(math.floor(bbox_1.shape[0]/2)):
    convex_bbox[:math.ceil(bbox_1.shape[0] / 2)] = np.minimum(bbox_1, bbox_2)[0:math.floor(bbox_1.shape[0] / 2)]
    # for i in range(math.floor(bbox_1.shape[0]/2), bbox_1.shape[0]):
    convex_bbox[math.floor(bbox_1.shape[0] / 2):bbox_1.shape[0]] = np.maximum(bbox_1, bbox_2)[
                                                                   math.floor(bbox_1.shape[0] / 2):bbox_1.shape[0]]
    return convex_bbox

def binarize_segment(seg, label):
    seg_bin = np.zeros(seg.shape)
    seg_bin[seg == label] = 1
    return seg_bin

def find_segment_differences(pred, gt, centroid_thresh, iou_thresh):
    properties = ['label', 'area', 'bbox', 'centroid']
    df_gt = pd.DataFrame.from_dict(measure.regionprops_table(gt, properties=properties))
    df_pred = pd.DataFrame.from_dict(measure.regionprops_table(pred, properties=properties))
    df_gt['centroid'] = np.nan
    df_pred['centroid'] = np.nan
    df_gt['bbox'] = np.nan
    df_pred['bbox'] = np.nan
    df_gt['match'] = False
    df_pred['match'] = False

    df_gt['centroid'] = df_gt.apply(lambda x: np.array([x['centroid-0'], x['centroid-1'],
                                                        x['centroid-2']]), axis=1)
    df_pred['centroid'] = df_pred.apply(lambda x: np.array([x['centroid-0'], x['centroid-1'],
                                                            x['centroid-2']]), axis=1)
    df_gt['bbox'] = df_gt.apply(lambda x: np.array([x['bbox-0'], x['bbox-1'], x['bbox-2'],
                                                    x['bbox-3'], x['bbox-4'], x['bbox-5']]), axis=1)
    df_pred['bbox'] = df_pred.apply(lambda x: np.array([x['bbox-0'], x['bbox-1'], x['bbox-2'],
                                                        x['bbox-3'], x['bbox-4'], x['bbox-5']]), axis=1)

    df_gt_ext = pd.DataFrame(columns=df_gt.columns)
    df_pred_ext = pd.DataFrame(columns=df_pred.columns)

    for idx_gt, row_gt in df_gt.iterrows():
        match_found = False
        df_pred_cand, pred_labels = find_candidates(gt, pred, row_gt['bbox'], df_pred, margin=20)
        df_match, label_match, df_closest, distances = find_centroid_matches(row_gt['centroid'], df_pred_cand,
                                                                             centroid_thresh)
        iou_max = 0
        iou_match = []
        for idx_pred, row_pred in df_match.iterrows():
            bbox = get_convex_bbox(row_gt['bbox'], row_pred['bbox'])
            gt_roi = crop_volume(gt, bbox)
            pred_roi = crop_volume(pred, bbox)
            iou = evaluate_overlap(gt_roi, pred_roi, idx_gt + 1, idx_pred + 1)

            if iou >= iou_thresh:
                iou_match.append((iou, idx_pred))
                if iou >= iou_max:
                    iou_max = iou
                match_found = True

        if match_found:
            idx_max = max(iou_match, key=lambda i: i[0])[1]
            df_gt['match'] = True
            df_pred['match'] = True
            df_gt_ext = df_gt_ext.append(df_gt.loc[idx_gt], ignore_index=False)
            df_pred_ext = df_pred_ext.append(df_pred.loc[idx_max], ignore_index=False)
        if not match_found:
            df_gt_ext = df_gt_ext.append(df_gt.loc[idx_gt], ignore_index=False)
            df_pred_ext = df_pred_ext.append({'area': 0.0, 'centroid': np.array([0.0, 0.0, 0.0]),
                                              'match': False}, ignore_index=True)

    subframe = df_pred[df_pred['match'] == False]
    if subframe.shape[0] > 0:
        df_pred_ext = df_pred_ext.append(subframe, ignore_index=False)
        n_rows = subframe.shape[0]
        df_gt_ext = df_gt_ext.append({'area': [0.0] * n_rows, 'centroid': np.array([0.0, 0.0, 0.0]) * n_rows,
                                      'match': [False] * n_rows}, ignore_index=True)
    return df_gt_ext, df_pred_ext

def create_tiff_stack(name, path):
    with tifffile.TiffWriter(name) as stack:
        for filename in glob.glob(path + '/*.tif'):
            stack.save(
                tifffile.imread(filename),
                photometric='minisblack',
                contiguous=True
                )