import numpy as np
import glob
from skimage import measure
import pandas as pd
import tifffile
import math


def iou(pred, gt):
    """
    Calculates Intersection over Union (IoU) metric

    Parameters
    ----------
    pred : binary numpy ndarray
    gt : binary numpy ndarray

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
    bbox : numpy array
        six dimensional vector with entries [row_start, col_start, slice_start, row_end, col_end, slice_end]
    margin : float
        margin value that gets added on each of the six sides of the bounding cube
    shape : triplet of int
        shape of the full volume

    Returns
    -------
    bbox : numpy array
        six dimensional vector with new boundary values
    """
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
    """
    This function finds all the cell segments around a given cell with its own bounding box.
    First the search space will be enlarged by adding a margin to the given bounding box. With that we will
    consider more candidate cells in the surrounding neighbourhood. All the contained cells within the enlarged volume a
    and their labels will be returned to the user.
    Parameters
    ----------
    gt : numpy ndarray
        multi-instance segmentation mask of ground truth segments
    pred : numpy ndarray
        multi-instance segmentation mask of prediction segments
    gt_bbox : pandas DataFrame
    df_pred : pandas DataFrame
    margin : int

    Returns
    -------
    df_pred_cand : pandas DataFrame
        contains all the segment instances that are found within the search volume
    pred_labels: numpy ndarray
        contains all the unique labels of cell instances
    """
    gt_bbox = add_margin_to_bbox(gt_bbox, margin, pred.shape)
    pred_subvol = crop_volume(pred, gt_bbox)
    values = np.unique(pred_subvol)
    pred_labels = np.delete(values, np.where(values == 0))
    df_pred_cand = df_pred.loc[df_pred['label'].isin(pred_labels)]
    # df_pred_cand = df_pred.loc[[i for i in pred_labels]]
    return df_pred_cand, pred_labels


def find_centroid_matches(centroid, df_segments, thresh):
    """
    Parameters
    ----------
    centroid : numpy ndarray
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
        df_closest = df.closest.append(df_segments.iloc[closest_idx])
    df_match = df_segments.loc[label_match]
    return df_match, label_match, df_closest, distances


def evaluate_overlap(gt, pred, gt_label, pred_label, metric='IoU'):
    """
    computes the overlap between two given labeled segments from two different volumes.
    Parameters
    ----------
    gt : numpy ndarray
        multi-instance segmentation mask of ground truth segments
    pred : numpy ndarray
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
    vol : numpy ndarray
        3 dimensional matrix of arbitrary type
    bbox : numpy ndarray
        bounding box with entries [row_start, col_start, slice_start, row_end, col_end, slice_end]

    Returns
    -------
    vol: numpy ndarray
        cropped volume
    """
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
    bbox_1 : numpy ndarray
        expected to be of type [row_start, col_start, slice_start, row_end, col_end, slice_end]
    bbox_2 : numpy ndarray
        expected to be of type [row_start, col_start, slice_start, row_end, col_end, slice_end]
    return
    -------
    convex_bbox: numpy ndarray
        bounding box of same type as input
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
    """
    Parameters
    ----------
    seg : numpy ndarray
        segmented volume with label and background values
    label : int
        label of interest

    Returns
    -------
    seg_bin : numpy ndarray
        binary volume matrix
    """
    seg_bin = np.zeros(seg.shape)
    seg_bin[seg == label] = 1
    return seg_bin


def find_segment_differences(pred, gt, centroid_thresh, iou_thresh):
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
    pred : numpy ndarray
        multi-instance segmentation mask of prediction segments
    pred : numpy ndarray
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
        df_pred_cand, pred_labels = find_candidates(gt, pred, row_gt['bbox'], df_pred, margin=20)
        # choose close cells based on centroid distance thresholding
        df_match, label_match, df_closest, distances = find_centroid_matches(row_gt['centroid'], df_pred_cand,
                                                                             centroid_thresh)
        iou_match = []
        for idx_pred, row_pred in df_match.iterrows():
            # enlarge region of interest such that all the segments of interest are fully contained within the volume
            bbox = get_convex_bbox(row_gt['bbox'], row_pred['bbox'])
            gt_roi = crop_volume(gt, bbox)
            pred_roi = crop_volume(pred, bbox)
            iou = evaluate_overlap(gt_roi, pred_roi, idx_gt + 1, idx_pred + 1)

            if iou >= iou_thresh:
                # store all the instances that fulfill threshold condition as list of tuples
                iou_match.append((iou, idx_pred))
                match_found = True

        if match_found:
            # find idx of segment with biggest iou score
            idx_max = max(iou_match, key=lambda i: i[0])[1]
            df_gt['match'] = True
            df_pred['match'] = True
            # fill up each new dataframe with the matching instance rows
            df_gt_ext = df_gt_ext.append(df_gt.loc[idx_gt], ignore_index=False)
            df_pred_ext = df_pred_ext.append(df_pred.loc[idx_max], ignore_index=False)
        if not match_found:
            # fill up each dataframe with a new line, where the dummy row in df_pred_ext corresponds to a
            # FALSE NEGATIVE of the prediction algorithm.
            df_gt_ext = df_gt_ext.append(df_gt.loc[idx_gt], ignore_index=False)
            df_pred_ext = df_pred_ext.append({'area': 0.0, 'centroid': np.array([np.nan, np.nan, np.nan]),
                                              'match': False}, ignore_index=True)

    # find all the FALSE POSITIVES in prediction mask and store them in subframe
    subframe = df_pred[df_pred['match'] == False]
    if subframe.shape[0] > 0:
        n_rows = subframe.shape[0]
        # fill up each dataframe with a new line, where the dummy row in df_gt_ext corresponds to a
        # FALSE POSITIVE of the prediction algorithm.
        df_pred_ext = df_pred_ext.append(subframe, ignore_index=False)
        df_gt_ext = df_gt_ext.append(pd.DataFrame.from_dict({'area': [0.0] * n_rows,
                                                             'centroid': [[np.nan, np.nan, np.nan]] * n_rows,
                                                             'match': [False] * n_rows}), ignore_index=True)
    return df_gt_ext, df_pred_ext


def calc_area_diff(df_gt, df_pred):
    """
    This

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


def create_tiff_stack(name, path):
    with tifffile.TiffWriter(name) as stack:
        for filename in glob.glob(path + '/*.tif'):
            stack.save(
                tifffile.imread(filename),
                photometric='minisblack',
                contiguous=True
            )
