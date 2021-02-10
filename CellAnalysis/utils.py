import numpy as np
import glob
from skimage import measure
import pandas as pd
import tifffile
import math
from scipy.signal import resample_poly
from scipy.ndimage import find_objects
import h5py
import cv2
import matplotlib.colors as mcolors
from random import randint

pd.options.mode.chained_assignment = 'raise'


def write_h5(name, data, name_dataset='main'):
    hf = h5py.File(name + '.h5', 'w')
    hf.create_dataset(name_dataset, data=data)
    hf.close()

def dataloader(path):
    # Read data from HDF5
    content = {}
    with h5py.File(path, 'r') as hdf:
        ls = list(hdf.keys())
        print('list of datasets in this file: \n', ls)
        for key in ls:
            data = hdf.get(key)
            content[key] = np.array(data)
    return content

def iou(pred, gt):
    """
    Calculates Intersection over Union (IoU) metric

    Parameters
    ----------
    pred : array_like(binary)
    gt : array_like(binary)

    Returns
    -------
    iou: float
    """
    intersection = np.logical_and(pred, gt).astype(np.float32).sum()
    union = np.logical_or(pred, gt).astype(np.float32).sum()
    if union == 0:
        iou = 0
    else:
        iou = intersection / union
    return iou


def add_margin_to_bbox(bbox, margin, shape):
    """
    This function computes a new dilated bounding box with a given margin factor.

    Parameters
    ----------
    bbox : array_like
        six dimensional vector with entries [row_start, col_start, slice_start, row_end, col_end, slice_end]
    margin : float
        margin value that gets added on each of the six sides of the bounding cube
    shape : triplet of int
        shape of the full volume

    Returns
    -------
    bbox : array_like
        six dimensional vector with new boundary values
    """
    temp = math.floor(bbox.shape[0] / 2)
    bbox_new = np.zeros(bbox.shape, dtype=int)
    for i in range(math.floor(bbox.shape[0] / 2)):
        bbox_new[i] = bbox[i] - margin
        if bbox_new[i] < 0:
            bbox_new[i] = 0
    for i in range(math.floor(bbox.shape[0] / 2), bbox.shape[0]):
        bbox_new[i] = bbox[i] + margin
        if bbox_new[i] >= shape[i - temp]:
            bbox_new[i] = shape[i - temp] - 1
    return bbox_new


def find_candidates(gt, pred, gt_bbox, df_pred, margin=0):
    """
    This function finds all the cell segments around a given cell with its own bounding box.
    First the search space will be enlarged by adding a margin to the given bounding box. With that we will
    consider more candidate cells in the surrounding neighbourhood. All the contained cells within the enlarged volume a
    and their labels will be returned to the user.
    Parameters
    ----------
    gt : array_like
        multi-instance segmentation mask of ground truth segments
    pred : array_like
        multi-instance segmentation mask of prediction segments
    gt_bbox : pandas DataFrame
    df_pred : pandas DataFrame
    margin : int

    Returns
    -------
    df_pred_cand : pandas DataFrame
        contains all the segment instances that are found within the search volume
    pred_labels: array_like
        contains all the unique labels of cell instances
    """
    if margin > 0:
        gt_bbox = add_margin_to_bbox(gt_bbox, margin, pred.shape)
    pred_subvol = crop_volume(pred, gt_bbox)
    values = np.unique(pred_subvol)
    pred_labels = np.delete(values, np.where(values == 0))
    df_pred_cand = df_pred.loc[df_pred['label'].isin(pred_labels)]
    return df_pred_cand, pred_labels


def find_centroid_matches(centroid, df_segments, thresh):
    """
    Parameters
    ----------
    centroid : array_like
        three dimensional vector representation of cell's centroid location within the volume
    df_segments : pandas DataFrame
        containing candidate cells that lay around the cell of interest
    thresh : float
        hyperparameter defining the threshold of the euclidean distance between pairs of cell centroids

    Returns
    -------
        df_match : pandas DataFrame
            containing all the cells that are considered a match under the centroid difference threshold
        label_match : list
            containing all the unique labels that satisfy the threshold condition
        df_closest : pandas DataFrame
            containing one instance that is the closest to the segment of interest
        distances : list
            list of distance values, where index corresponds with df_match's 0-axis
    """
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
        df_closest = df_closest.append(df_segments.iloc[closest_idx])
    df_match = df_segments.loc[label_match]
    return df_match, label_match, df_closest, distances


def evaluate_overlap(gt, pred, gt_label, pred_label, metric='IoU'):
    """
    computes the overlap between two given labeled segments from two different volumes.
    Parameters
    ----------
    gt : array_like
        multi-instance segmentation mask of ground truth segments
    pred : array_like
        multi-instance segmentation mask of prediction segments
    gt_label : int
        label of instance of interest from ground truth volume
    pred_label : int
        label of instance of interest from prediction volume
    metric : string
        to be implemented: default metric is IoU
    Returns
    -------
        iou: float
            Intersection over Union metric
    """
    gt_bin = binarize_segment(gt, gt_label)
    pred_bin = binarize_segment(pred, pred_label)
    return iou(gt_bin, pred_bin)


def crop_volume(vol, bbox):
    """
    Parameters
    ----------
    vol : array_like
        3 dimensional matrix of arbitrary type
    bbox : array_like
        bounding box with entries [row_start, col_start, slice_start, row_end, col_end, slice_end]

    Returns
    -------
    vol: array_like
        cropped volume
    """
    bbox = bbox.astype(int)
    if len(bbox) == 6:
        row_start = bbox[0]
        row_end = bbox[3]
        col_start = bbox[1]
        col_end = bbox[4]
        slice_start = bbox[2]
        slice_end = bbox[5]
        return vol[row_start:row_end, col_start:col_end, slice_start:slice_end]
    elif len(bbox) == 4:
        row_start = bbox[0]
        row_end = bbox[2]
        col_start = bbox[1]
        col_end = bbox[3]
        return vol[row_start:row_end, col_start:col_end]
    else:
        raise TypeError('Check your bbox array.')


def get_convex_bbox(bbox_1, bbox_2):
    """
    computes the convex bounding box out of two given bounding boxes.

    Attributes
    ----------
    bbox_1 : array_like
        expected to be of type [row_start, col_start, slice_start, row_end, col_end, slice_end]
    bbox_2 : array_like
        expected to be of type [row_start, col_start, slice_start, row_end, col_end, slice_end]
    return
    -------
    convex_bbox: array_like
        bounding box of same type as input
    """
    assert len(bbox_1.shape) == 1 & len(bbox_2.shape) == 1, print("Dimensionality must be of shape 1.")
    assert bbox_1.shape == bbox_2.shape, print("bbox must be of same shapes.")
    convex_bbox = np.zeros(bbox_1.shape)
    convex_bbox[:math.ceil(bbox_1.shape[0] / 2)] = np.minimum(bbox_1, bbox_2)[0:math.floor(bbox_1.shape[0] / 2)]
    convex_bbox[math.floor(bbox_1.shape[0] / 2):bbox_1.shape[0]] = np.maximum(bbox_1, bbox_2)[
                                                                   math.floor(bbox_1.shape[0] / 2):bbox_1.shape[0]]
    return convex_bbox.astype(int)


def binarize_segment(seg, label):
    """
    Parameters
    ----------
    seg : array_like
        segmented volume with label and background values
    label : int
        label of interest

    Returns
    -------
    seg_bin : array_like
        binary volume matrix
    """
    seg_bin = np.zeros(seg.shape)
    seg_bin[seg == label] = 1
    return seg_bin
'''
def find_segment_differences(pred, gt, margin=20, iou_thresh=0.5):
    """
    Function that takes two segmented volumes and returns two corresponding DataFrame instances of same shapes.
    Each unique segmentation instance is represented in either one of the two dataframes. For False Positives and
    False Negatives either one of the dataframes gets filled up with a dummy row like the following:
    {'area': 0.0, 'centroid': np.array([np.nan, np.nan, np.nan]), 'match': False}
    Each column of the dataframe contains information about specific properties of the instance mask, such as...
        - area
        - centroid location
        - bounding box
        - label

    Parameters
    ----------
    pred : array_like
        multi-instance segmentation mask of prediction segments
    pred : array_like
        multi-instance segmentation mask of ground truth segments
    centroid_thresh : float
    iou_thresh : float

    Returns
    -------
    df_gt_ext : pandas DataFrame
        extended DataFrame corresponding to ground truth mask to compare properties with predicted mask
    df_pred_ext : pandas DataFrame
        extended DataFrame corresponding to predicted mask to compare properties with ground truth mask
    """
    properties = ['label', 'area', 'bbox', 'centroid']
    df_gt = pd.DataFrame.from_dict(measure.regionprops_table(gt, properties=properties))
    df_pred = pd.DataFrame.from_dict(measure.regionprops_table(pred, properties=properties))
    df_gt['centroid'] = np.nan
    df_pred['centroid'] = np.nan
    df_gt['bbox'] = np.nan
    df_pred['bbox'] = np.nan
    df_gt['match'] = False
    df_pred['match'] = False
    df_pred['confusion'] = None
    df_gt['confusion'] = None

    df_gt['centroid'] = df_gt['centroid'].apply(lambda x: np.array([x['centroid-0'], x['centroid-1'],
                                                                    x['centroid-2']]), axis=1)
    df_pred['centroid'] = df_pred['centroid'].apply(lambda x: np.array([x['centroid-0'], x['centroid-1'],
                                                                        x['centroid-2']]), axis=1)
    df_gt['bbox'] = df_gt['bbox'].apply(lambda x: np.array([x['bbox-0'], x['bbox-1'], x['bbox-2'],
                                                    x['bbox-3'], x['bbox-4'], x['bbox-5']]), axis=1)
    df_pred['bbox'] = df_pred['bbox'].apply(lambda x: np.array([x['bbox-0'], x['bbox-1'], x['bbox-2'],
                                                        x['bbox-3'], x['bbox-4'], x['bbox-5']]), axis=1)

    df_gt_ext = pd.DataFrame(columns=df_gt.columns)
    df_pred_ext = pd.DataFrame(columns=df_pred.columns)
    for idx_gt, row_gt in df_gt.iterrows():
        match_found = False

        # find neighbouring cells within a dilated bounding cube search space
        df_pred_cand, pred_labels = find_candidates(gt, pred, row_gt['bbox'], df_pred, margin=margin)
        # choose close cells based on centroid distance thresholding
        #df_match, label_match, df_closest, distances = find_centroid_matches(row_gt['centroid'], df_pred_cand,
        #                                                                     centroid_thresh)
        iou_match = []
        for idx_pred, row_pred in df_pred_cand.iterrows():
            # enlarge region of interest such that all the segments of interest are fully contained within the volume
            bbox = get_convex_bbox(row_gt['bbox'], row_pred['bbox'])
            gt_roi = crop_volume(gt, bbox)
            pred_roi = crop_volume(pred, bbox)
            gt_label = df_gt.iloc[idx_gt]['label']
            pred_label = df_pred.iloc[idx_pred]['label']
            iou = evaluate_overlap(gt_roi, pred_roi, gt_label, pred_label)
            if iou >= iou_thresh:
                # store all the instances that fulfill threshold condition as list of tuples
                iou_match.append((iou, idx_pred))
                match_found = True

        if match_found:
            # find idx of segment with biggest iou score
            idx_max = max(iou_match, key=lambda i: i[0])[1]
            df_gt.loc[idx_gt, 'match'] = True
            df_gt.loc[idx_gt, 'confusion'] = 'TP'
            df_pred.loc[idx_max, 'match'] = True
            df_pred.loc[idx_max, 'confusion'] = 'TP' # True Positive
            # fill up each new dataframe with the matching instance rows
            df_gt_ext = df_gt_ext.append(df_gt.loc[idx_gt], ignore_index=False)
            df_pred_ext = df_pred_ext.append(df_pred.loc[idx_max], ignore_index=False)
        if not match_found:
            # fill up each dataframe with a new line, where the dummy row in df_pred_ext corresponds to a
            # FALSE NEGATIVE of the prediction algorithm.
            df_gt.loc[idx_gt, 'confusion'] = 'FN'
            df_gt_ext = df_gt_ext.append(df_gt.loc[idx_gt], ignore_index=False)
            df_pred_ext = df_pred_ext.append({'area': 0.0, 'centroid': np.array([np.nan, np.nan, np.nan]),
                                              'match': False, 'confusion': 'FN'}, ignore_index=True)

    # find all the FALSE POSITIVES in prediction mask and store them in subframe
    subframe = df_pred[(df_pred['match'] == False) & (df_pred['confusion'] != 'FN')]
    if subframe.shape[0] > 0:
        n_rows = subframe.shape[0]
        # fill up each dataframe with a new line, where the dummy row in df_gt_ext corresponds to a
        # FALSE POSITIVE of the prediction algorithm.
        subframe['confusion'].values[:] = 'FP'
        df_pred_ext = df_pred_ext.append(subframe, ignore_index=False)
        df_gt_ext = df_gt_ext.append(pd.DataFrame.from_dict({'area': [0.0] * n_rows,
                                                             'centroid': [[np.nan, np.nan, np.nan]] * n_rows,
                                                             'match': [False] * n_rows,
                                                             'confusion': ['FP'] * n_rows}), ignore_index=True)
    df_gt_ext.reset_index(inplace=True)
    df_pred_ext.reset_index(inplace=True)
    return df_gt_ext, df_pred_ext

def sync_instance_masks(gt, pred, df_gt, df_pred, merged_mask = True):
    """
    Function that assigns pixels a unique color based on TP, FP and FN for visualization purposes. Moreover returning
    a synced label pattern with gt_sync and pred_sync. Cells that seem to match in both masks will have the same
    label values.
    Parameters
    ----------
    gt : array_like
    pred : array_like
    df_gt : pandas DataFrame
    df_pred : pandas DataFrame

    Returns
    -------
    array_like
        gt_sync
        pred_sync
        merged
    """
    assert gt.shape == pred.shape, print("Input volumes must be of same size.")
    assert df_gt.shape[0] == df_pred.shape[0], print("Inputs must be of same shape along axis 0.")

    gt_sync = np.zeros(gt.shape)
    pred_sync = np.zeros(pred.shape)
    shape = list(gt_sync.shape)
    shape.append(4)
    merged = np.zeros(tuple(shape)) #, dtype=np.uint8)
    merged[:, :, :, 3] = 0.9
    #merged = convert_gray2rgb(gt_sync)
    i = 1
    for idx, row in df_gt.iterrows():
        if df_pred.iloc[idx]['confusion'] == 'TP':
            gt_sync[gt == df_gt.iloc[idx]['label']] = i
            pred_sync[pred == df_pred.iloc[idx]['label']] = i
            if merged_mask:
                #merged[(gt_sync == pred_sync) * (gt_sync != 0)] = mcolors.to_rgb('darkgreen')   # darkgreen (TP)
                #merged[(gt_sync == i) & (pred_sync != i) & (pred_sync != 0)] = [0, 128, 0]  # green          (FN)
                #merged[(gt_sync != i) & (pred_sync == i)] = [144, 238, 144]                 # light green    (FP)
                merged[(gt_sync == pred_sync) * (gt_sync != 0)] = mcolors.to_rgba('darkgreen', 1.0)  # darkgreen (TP)
                merged[(gt_sync == i) & (pred_sync != i) & (pred_sync != 0)] = mcolors.to_rgba('darkgreen', 0.6)  # green          (FN)
                merged[(gt_sync != i) & (pred_sync == i)] = mcolors.to_rgba('darkgreen', 0.25)  # light green    (FP)
        elif df_pred.iloc[idx]['confusion'] == 'FP':
            pred_sync[pred == df_pred.iloc[idx]['label']] = i
            if merged_mask:
                #merged[(gt_sync == 0) & (pred_sync == i)] = [220, 20, 60]        # red            (FP - background)
                #merged[(pred_sync == i) & (gt_sync != 0)] = [255, 127, 80]       # orangered      (FP - cell overlap)
                merged[(gt_sync == 0) & (pred_sync == i)] = mcolors.to_rgba('red', 1.0)  # red            (FP - background)
                merged[(pred_sync == i) & (gt_sync != 0)] = mcolors.to_rgba('red', 0.5)  # orangered      (FP - cell overlap)
        elif df_pred.iloc[idx]['confusion'] == 'FN':
            gt_sync[gt == df_gt.iloc[idx]['label']] = i
            if merged_mask:
                #merged[(gt_sync == i) & (pred_sync == 0)] = [218, 165, 32]      # yellow         (FN - background)
                #merged[(gt_sync == i) & (pred_sync != 0)] = [255, 215, 0]        # greenyellow    (FN - cell overlap)
                merged[(gt_sync == i) & (pred_sync == 0)] = mcolors.to_rgba('gold', 1.0)  # yellow         (FN - background)
                merged[(gt_sync == i) & (pred_sync != 0)] = mcolors.to_rgba('gold', 0.5)  # greenyellow    (FN - cell overlap)
        else:
            raise KeyError("Something is wrong with your DataFrame. Entry not found in 'confusion' column." )
        i += 1
    return gt_sync, pred_sync, merged
'''


def find_segment_differences(pred, gt, margin=20, iou_thresh=0.5, centroid_thresh=5, do_centroid_thresh=False):
    """
    Function that takes two segmented volumes and returns two corresponding DataFrame instances of same shapes.
    Each unique segmentation instance is represented in either one of the two dataframes. For False Positives and
    False Negatives either one of the dataframes gets filled up with a dummy row like the following:
    {'area': 0.0, 'centroid': np.array([np.nan, np.nan, np.nan]), 'match': False}
    Each column of the dataframe contains information about specific properties of the instance mask, such as...
        - area
        - centroid location
        - bounding box
        - label

    Parameters
    ----------
    pred : array_like
        multi-instance segmentation mask of prediction segments
    pred : array_like
        multi-instance segmentation mask of ground truth segments
    centroid_thresh : float
    iou_thresh : float

    Returns
    -------
    df_gt_ext : pandas DataFrame
        extended DataFrame corresponding to ground truth mask to compare properties with predicted mask
    df_pred_ext : pandas DataFrame
        extended DataFrame corresponding to predicted mask to compare properties with ground truth mask
    """
    properties = ['label', 'area', 'bbox', 'centroid']
    df_gt = pd.DataFrame.from_dict(measure.regionprops_table(gt, properties=properties))
    df_pred = pd.DataFrame.from_dict(measure.regionprops_table(pred, properties=properties))
    df_gt['bbox'] = np.nan
    df_pred['bbox'] = np.nan
    df_gt['centroid'] = np.nan
    df_pred['centroid'] = np.nan
    df_gt['match'] = False
    df_pred['match'] = False
    df_pred['confusion'] = None
    df_gt['confusion'] = None

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

        # find neighbouring cells within a dilated bounding cube search space
        df_pred_cand, pred_labels = find_candidates(gt, pred, row_gt['bbox'], df_pred, margin=margin)
        df_iter = df_pred_cand
        iou_match = []
        for idx_pred, row_pred in df_iter.iterrows():
            # enlarge region of interest such that all the segments of interest are fully contained within the volume
            bbox = get_convex_bbox(row_gt['bbox'], row_pred['bbox'])
            gt_roi = crop_volume(gt, bbox)
            pred_roi = crop_volume(pred, bbox)
            gt_label = df_gt.iloc[idx_gt]['label']
            pred_label = df_pred.iloc[idx_pred]['label']
            iou = evaluate_overlap(gt_roi, pred_roi, gt_label, pred_label)
            if iou >= iou_thresh:
                # store all the instances that fulfill threshold condition as list of tuples
                iou_match.append((iou, idx_pred))
                match_found = True

        if match_found:
            # find idx of segment with biggest iou score
            idx_max = max(iou_match, key=lambda i: i[0])[1]
            if do_centroid_thresh:
                centroid_gt = df_gt.loc[idx_gt, 'centroid']
                centroid_pred = df_pred.loc[idx_max, 'centroid']
                if np.linalg.norm(centroid_gt - centroid_pred) <= centroid_thresh:
                    df_gt.loc[idx_gt, 'match'] = True
                    df_gt.loc[idx_gt, 'confusion'] = 'TP'
                    df_pred.loc[idx_max, 'match'] = True
                    df_pred.loc[idx_max, 'confusion'] = 'TP' # True Positive
                    # fill up each new dataframe with the matching instance rows
                    df_gt_ext = df_gt_ext.append(df_gt.loc[idx_gt], ignore_index=False)
                    df_pred_ext = df_pred_ext.append(df_pred.loc[idx_max], ignore_index=False)
                else:
                    print('Centroid Threshold criteria failed.')
                    print('gt centroid: {}'.format(centroid_gt))
                    print('pred centroid: {}'.format(centroid_pred))
                    print('difference in centroid position: {}'.format(np.linalg.norm(centroid_gt - centroid_pred)))
                    match_found = False
            else:
                df_gt.loc[idx_gt, 'match'] = True
                df_gt.loc[idx_gt, 'confusion'] = 'TP'
                df_pred.loc[idx_max, 'match'] = True
                df_pred.loc[idx_max, 'confusion'] = 'TP'  # True Positive
                # fill up each new dataframe with the matching instance rows
                df_gt_ext = df_gt_ext.append(df_gt.loc[idx_gt], ignore_index=False)
                df_pred_ext = df_pred_ext.append(df_pred.loc[idx_max], ignore_index=False)
        if not match_found:
            # fill up each dataframe with a new line, where the dummy row in df_pred_ext corresponds to a
            # FALSE NEGATIVE of the prediction algorithm.
            df_gt.loc[idx_gt, 'confusion'] = 'FN'
            df_gt_ext = df_gt_ext.append(df_gt.loc[idx_gt], ignore_index=False)
            df_pred_ext = df_pred_ext.append({'area': 0.0, 'centroid': np.array([np.nan, np.nan, np.nan]),
                                              'match': False, 'confusion': 'FN'}, ignore_index=True)

    # find all the FALSE POSITIVES in prediction mask and store them in subframe
    subframe = df_pred[(df_pred['match'] == False) & (df_pred['confusion'] != 'FN')]
    if subframe.shape[0] > 0:
        n_rows = subframe.shape[0]
        # fill up each dataframe with a new line, where the dummy row in df_gt_ext corresponds to a
        # FALSE POSITIVE of the prediction algorithm.
        subframe['confusion'].values[:] = 'FP'
        df_pred_ext = df_pred_ext.append(subframe, ignore_index=False)
        df_gt_ext = df_gt_ext.append(pd.DataFrame.from_dict({'area': [0.0] * n_rows,
                                                             'centroid': [[np.nan, np.nan, np.nan]] * n_rows,
                                                             'match': [False] * n_rows,
                                                             'confusion': ['FP'] * n_rows}), ignore_index=True)
    df_gt_ext.reset_index(drop=True, inplace=True)
    df_pred_ext.reset_index(drop=True, inplace=True)
    return df_gt_ext, df_pred_ext


def sync_instance_masks(gt, pred, df_gt, df_pred, merged_mask = True):
    """
    Function that assigns pixels a unique color based on TP, FP and FN for visualization purposes. Moreover returning
    a synced label pattern with gt_sync and pred_sync. Cells that seem to match in both masks will have the same
    label values.
    Parameters
    ----------
    gt : array_like
    pred : array_like
    df_gt : pandas DataFrame
    df_pred : pandas DataFrame

    Returns
    -------
    array_like
        gt_sync
        pred_sync
        merged
    """
    assert gt.shape == pred.shape, print("Input volumes must be of same size.")
    assert df_gt.shape[0] == df_pred.shape[0], print("Inputs must be of same shape along axis 0.")

    gt_sync = np.zeros(gt.shape)
    pred_sync = np.zeros(pred.shape)
    shape = list(gt_sync.shape)
    shape.append(4)
    merged_gt = np.zeros(tuple(shape)) #, dtype=np.uint8)
    merged_gt[:, :, :, 3] = 0.9
    merged_pred = np.zeros(tuple(shape))  # , dtype=np.uint8)
    merged_pred[:, :, :, 3] = 0.9
    i = 1
    for idx, row in df_gt.iterrows():
        if df_pred.iloc[idx]['confusion'] == 'TP':
            gt_sync[gt == df_gt.iloc[idx]['label']] = i
            pred_sync[pred == df_pred.iloc[idx]['label']] = i
            if merged_mask:
                merged_gt[(gt_sync == pred_sync) * (gt_sync != 0)] = mcolors.to_rgba('darkgreen', 1.0)  # TP pixel
                merged_gt[(gt_sync == i) & (pred == 0)] = mcolors.to_rgba('darkgreen', 0.6)        # FN pixel
                merged_gt[(gt_sync != i) & (pred_sync == i)] = mcolors.to_rgba('darkgreen', 0.25)       # FP pixel

                merged_pred[(gt_sync == pred_sync) * (gt_sync != 0)] = mcolors.to_rgba('darkgreen', 1.0)  # TP pixel
                merged_pred[(gt_sync == i) & (pred == 0)] = mcolors.to_rgba('darkgreen', 0.6)  # FN pixel
                merged_pred[(gt_sync != i) & (pred_sync == i)] = mcolors.to_rgba('darkgreen', 0.25)  # FP pixel
        elif df_pred.iloc[idx]['confusion'] == 'FP':
            pred_sync[pred == df_pred.iloc[idx]['label']] = i
            if merged_mask:
                merged_pred[(gt == 0) & (pred_sync == i)] = mcolors.to_rgba('red', 1.0)  # FP - background
                merged_pred[(pred_sync == i) & (gt != 0)] = mcolors.to_rgba('red', 0.5)  # FP - cell overlap
        elif df_pred.iloc[idx]['confusion'] == 'FN':
            gt_sync[gt == df_gt.iloc[idx]['label']] = i
            if merged_mask:
                merged_gt[(gt_sync == i) & (pred == 0)] = mcolors.to_rgba('gold', 1.0)  # FN - background
                merged_gt[(gt_sync == i) & (pred != 0)] = mcolors.to_rgba('gold', 0.5)  # FN - cell overlap
        else:
            raise KeyError("Something is wrong with your DataFrame. Entry not found in 'confusion' column." )
        i += 1
    return gt_sync, pred_sync, merged_gt, merged_pred

def gray2rgb(volume):
    shape = list(volume.shape)
    shape.append(3)
    out = np.empty(tuple(shape), dtype=np.uint8)
    out[..., 0] = volume
    out[..., 1] = volume
    out[..., 2] = volume
    return out


def create_cmap(N):
    colors = []
    for i in range(N):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    return colors


def gray2rgba(volume, alpha=1.0):
    shape = list(volume.shape)
    shape.append(4)
    out = np.empty(tuple(shape), dtype=np.float32)
    out[..., 0] = volume
    out[..., 1] = volume
    out[..., 2] = volume
    out[..., 3] = alpha * 255.0
    return out


def label2rgb_3D(label_mask, image, alpha=0.2):
    image = gray2rgb(image).astype(np.float32)
    values = np.unique(label_mask)
    labels = np.delete(values, np.where(values == 0))
    cmap = create_cmap(len(labels))

    for i, label in enumerate(labels):
        color = np.array([255.0 * x for x in mcolors.to_rgb(cmap[i])], dtype=np.float32)
        image[label_mask == label] = (1 - alpha) * image[label_mask == label] + alpha * color
    return image / 255.0


def calc_area_diff(df_gt, df_pred):
    """
    Parameters
    ----------
    df_gt : pandas dataframe
        ground truth data that must contain a column named 'area' with cell values of type float
    df_pred : pandas dataframe
        prediction data that must contain a column named 'area' with cell values of type float

    Returns
    -------
    df: pandas dataframe
        with 'area_diff' column that contains difference in areas and same length as inputs
    """
    if "area" not in df_gt.columns or "area" not in df_pred.columns:
        print("'area' column is required for execution of the function.")
    elif df_gt['area'].shape != df_pred['area'].shape:
        raise KeyError("shape of area column must be of same shape.")
    df = pd.DataFrame({"area_diff": [np.nan] * df_gt.shape[0]})
    df['area_diff'] = df_gt['area'] - df_pred['area']
    return df


def calc_centroid_diff(df_gt, df_pred):
    """
    al lengths along axis 0 and calculates the difference of two 3D centroid vectors
    originating from two different datasets.This function takes two dataframes of equ

    Parameters
    ----------
    df_gt : pandas dataframe
        Contains centroid triplets of ground truth labels
    df_pred : pandas dataframe
        Contains centroid triplets of prediction labels

    Returns
    -------
    df: pandas dataframe
        contains "centroid_diff" column with centroid difference vector
    """
    if "centroid" not in df_gt.columns or "centroid" not in df_pred.columns:
        print("'centroid' column is required for execution of the function.")
    elif df_gt['centroid'].shape[0] != df_pred['centroid'].shape[0]:
        raise KeyError("length of centroid columns must be the same.")
    df = pd.DataFrame({"centroid_diff": [np.array([np.nan, np.nan, np.nan])] * df_gt.shape[0]})
    for i in range(df_gt.shape[0]):
        if np.isnan(df_gt["centroid"].iloc[i]).any() or np.isnan(df_pred["centroid"].iloc[i]).any():
            continue
        else:
            df['centroid_diff'].iloc[i] = df_gt["centroid"].iloc[i] - df_pred["centroid"].iloc[i]
    return df


def get_centroids_from_mask(seg):
    df = pd.DataFrame.from_dict(measure.regionprops_table(seg, properties=['centroid']))
    if len(seg.shape) == 3:
        df['centroid'] = df.apply(lambda r: tuple(r), axis=1).apply(np.array)
    elif len(seg.shape) == 2:
        df['centroid'] = df.apply(lambda r: tuple(r), axis=1).apply(np.array)
    return df


def get_centroid_array(df):
    a = df['centroid'].to_numpy()
    a = np.stack(a, axis=0)
    return a


def ignore_boundary_segments(mask):
    """
    Deletes segments that touch the boundaries of the segmented mask. This will be important for e.g.
    centroid analysis, where it is important to have information about the whole segment.
    Parameters
    ----------
    mask : array_like

    Returns
    -------
    array_like
    """
    rows, cols, slices = mask.shape
    for label in np.delete(np.unique(mask), np.where(mask == 0)):
        if mask[0,:,:] == label or mask[:, 0, :] == label or mask[:, :, 0] == label or mask[rows-1, :, :] == label \
                or mask[:, cols-1, :] == label or mask[:, :, slices-1] == label:
            np.delete(mask, np.where(mask == label))
    return mask

def cast_indices_to_int(df):
    df['centroid'] = df['centroid'].apply(lambda x: list(map(int, map(round, x))))
    return df


# Source: https://github.com/MouseLand/cellpose/blob/master/cellpose/utils.py
def distance_to_boundary(masks):
    """ get distance to boundary of mask pixels

    Parameters
    ----------------
    masks: int, 2D or 3D array
        size [Ly x Lx] or [Lz x Ly x Lx], 0=NO masks; 1,2,...=mask labels
    Returns
    ----------------
    dist_to_bound: 2D or 3D array
        size [Ly x Lx] or [Lz x Ly x Lx]
    """
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError('distance_to_boundary takes 2D or 3D array, not %dD array' % masks.ndim)
    dist_to_bound = np.zeros(masks.shape, np.float64)

    if masks.ndim == 3:
        for i in range(masks.shape[0]):
            dist_to_bound[i] = distance_to_boundary(masks[i])
        return dist_to_bound
    else:
        slices = find_objects(masks)
        for i, si in enumerate(slices):
            if si is not None:
                sr, sc = si
                mask = (masks[sr, sc] == (i + 1)).astype(np.uint8)
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                pvc, pvr = np.concatenate(contours[-2], axis=0).squeeze().T
                ypix, xpix = np.nonzero(mask)
                min_dist = ((ypix[:, np.newaxis] - pvr) ** 2 +
                            (xpix[:, np.newaxis] - pvc) ** 2).min(axis=1)
                dist_to_bound[ypix + sr.start, xpix + sc.start] = min_dist
        return dist_to_bound


# Source: https://github.com/MouseLand/cellpose/blob/master/cellpose/utils.py
def masks_to_edges(masks, threshold=1.0):
    """ get edges of masks as a 0-1 array

    Parameters
    ----------------
    masks: int, 2D or 3D array
        size [Ly x Lx] or [Lz x Ly x Lx], 0=NO masks; 1,2,...=mask labels
    Returns
    ----------------
    edges: 2D or 3D array
        size [Ly x Lx] or [Lz x Ly x Lx], True pixels are edge pixels
    """
    dist_to_bound = distance_to_boundary(masks)
    edges = (dist_to_bound < threshold) * (masks > 0)
    return edges


def highlight_boundary(img, seg, mode='gt', thickness=1.0):
    """
    Highlights the boundaries of a multi instance segmentation mask in the original intensity volume.
    Different colour indications depending on mode "gt" or "pred".
    Parameters
    ----------
    img : array_like
    seg : array_like
    mode : boolean (optionl)

    Returns
    -------
    array_like
    """
    if img.shape[-1] != 3:
        img = gray2rgb(img)
    if mode == 'gt':
        color_gt = tuple([255*x for x in mcolors.to_rgb('yellowgreen')])
        #img = segmentation.mark_boundaries(img, seg, color=(0.1,0.6,0.1), mode='subpixel')
        img[masks_to_edges(seg, threshold=thickness)] = color_gt        #medium sea green
    elif mode == 'pred':
        color_pred = tuple([255*x for x in mcolors.to_rgb('coral')])
        #img = segmentation.mark_boundaries(img, seg, color=(0.8, 0, 0), mode='subpixel')
        img[masks_to_edges(seg, threshold=thickness)] = color_pred      #magenta
    else:
        raise KeyError('Key that has been passed for mode is not known.')
    return img


def draw_boundaries(img, gt, pred, draw_centroid = True, thickness=1.0, z_pos=2):
    """
    Draws the boundaries of a given ground truth and prediction mask in the original intensity volume by using
    the highlight_boundaries function. Optionally centroids can be drawn in the image highlighted as crosses.
    Parameters
    ----------
    img : array_like
    gt : array_like
    pred : array_like
    draw_centroid : boolean (optional)

    Returns
    -------
    array_like
    """
    dim = img.ndim
    if dim == 3:
        img = np.moveaxis(img, z_pos, 0)
        gt = np.moveaxis(gt, z_pos, 0)
        pred = np.moveaxis(pred, z_pos, 0)
    img = highlight_boundary(highlight_boundary(img, pred, mode='pred', thickness=thickness),
                             gt, mode='gt', thickness=thickness)

    if draw_centroid and dim == 3:
        df_gt = cast_indices_to_int(get_centroids_from_mask(gt))
        df_pred = cast_indices_to_int(get_centroids_from_mask(pred))
        print(df_pred.centroid)
        for idx_pred, row_pred in df_pred.iterrows():
            row, col, slice = tuple(row_pred['centroid'])
            print(tuple(row_pred['centroid']))
            img[row-3:row+4, col, slice] = [255.0, 128.0, 0.0]
            img[row, col-3:col+4, slice] = [255.0, 128.0, 0.0]
            img[row, col, slice-3:slice+4] = [255.0, 128.0, 0.0]
        for idx_gt, row_gt in df_gt.iterrows():
            row, col, slice = tuple(row_gt['centroid'])
            print(tuple(row_gt['centroid']))
            img[row-3:row+4, col, slice] = [34.0, 139.0,  34.0]
            img[row, col-3:col+4, slice] = [34.0, 139.0,  34.0]
            img[row, col, slice-3:slice+4] = [34.0, 139.0,  34.0]
    return img


def upsample_intensity(vol, factor):
    """
    Upsamples a given 3D intensity volume by a given factor.
    Parameters
    ----------
    vol : array_like
    factor : int

    Returns
    -------
    array_like
    """
    assert isinstance(factor, int), print("Upsampling factor must be of type int.")
    factors = [(factor, 1), (factor, 1), (factor, 1)]
    for k in range(3):
        vol = resample_poly(vol, factors[k][0], factors[k][1], axis=k)
    return vol


def upsample_mask(mask, factor, in_z = True):
    """
    Upsamples a given segmentation mask by a given factor.
    Parameters
    ----------
    mask : array_like
    factor : int

    Returns
    -------
    array_like
    """
    assert isinstance(factor, int), print("Upsampling factor must be of type int.")
    if not in_z:
        mask_upsample = mask.repeat(factor, axis=1).repeat(factor, axis=2)
    else:
        mask_upsample = mask.repeat(factor, axis=0).repeat(factor, axis=1).repeat(factor, axis=2)
    return mask_upsample

def create_tiff_stack(name, path):
    with tifffile.TiffWriter(name) as stack:
        for filename in glob.glob(path + '/*.tif'):
            stack.save(
                tifffile.imread(filename),
                photometric='minisblack',
                contiguous=True
            )


def dilate_and_erode(img, kernel):
    img = np.uint8(img)
    dilation = cv2.dilate(img, kernel, iterations=1)
    return cv2.erode(dilation, kernel, iterations=1)


def rgb_to_rgba(rgb):
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]
    a = 255 - min(r, g, b) / 255
    return [(255 + (r - 255) / a) / 255, (255 + (g - 255) / a) / 255, (255 + (b - 255) / a) / 255, a]
