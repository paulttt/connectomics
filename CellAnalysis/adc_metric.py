from CellAnalysis.utils import *
from scipy.stats import sem
import numpy as np


# Implemented on the idea of Zudi Lin from Pfister Lab, Harvard University.
def distance_matrix(gt, pred, real_distance=True, size=None):
    """
    Parameters
    ----------
    gt : array-like
        Centroid vector of ground truth data of shape (N,3)
    pred : array-like
        Centroid vector of prediction data of shape (M,3)
    real_distance: boolean (optional)
        Calculates the real euclidean distance using the voxel_size variable
    size: tuple (optional)
        length along each axis of each voxel

    Returns
    -------
    array-like
        Matrix of shape (N,M) with euclidean distances between centroids.
    """
    if len(gt.shape) == 3:
        diffs = gt[:, None, :] - pred[None, :, :]
        if size is None:
            size = (1, 1, 1)
    elif len(gt.shape) == 2:
        diffs = gt[:, None] - pred[None, :]
        if size is None:
            size = (1, 1)
    else:
        raise NotImplementedError

    if real_distance:
        return calc_real_dist(diffs, size)
    else:
        return np.linalg.norm(diffs, axis=-1)


def calc_real_dist(dist, size=None):
    """
    Calculates the real euclidean norms for a given matrix and given voxel dimensions.

    Parameters
    ----------
    dist : array_like
    size : tuple

    Returns
    -------
    array_like
        real euclidean distance matrix
    """
    for i, axis_size in enumerate(size):
        dist[..., i] *= axis_size
    return np.linalg.norm(dist, axis=-1)


def average_distance_between_centroids(gt, pred, dist_thresh=0.5, all_stats=False, real_dist=True, size=None):
    """
    Average Distance between Centroids (ADC) Metric
    Can be informative for cell alignment and can be used for registration purposes.
    Metric is based on two parts:
        part a - Average Distance to Ground Truth segments (ADGC): Term that penalizes false positive predictions.
        part b - Average Distance to Predicted Centroids (ADPC): Term that penalizes false negative predictions.
    ADC = (ADGC + ADPC) / 2

    Parameters
    ----------
    gt : array-like
        ground truth segmentation mask
    pred : array-like
        predicted segmentation mask by the model
    dist_thresh : float (optional)
        threshold for deciding if cells are the same w.r.t. centroid offset
    all_stats : boolean (optional)
        Calculates and returns all the stats
    real_dist: boolean (optional)
        Calculates the real euclidean distance using the voxel_size variable
    size: tuple (optional)
        length along each axis of each voxel

    Returns
    -------
    min_adc:    float
        Average Distance between Centroids (ADC)
    min_adpc:    float
        Average Distance to Prediction Centroids (ADPC)
    min_adgc:    float
        Average Distance to Ground Truth Centroids (ADGC)

    mean_adc:   float (optional)
    f_score:    float (optional)
    precision:  float (optional)
    recall:     float (optional)
    tp:         int (optional)
    fp:         int (optional)
    fn:         int (optional)
    """
    if size is None:
        if len(gt.shape) == 3:
            size = (1, 1, 1)
        elif len(gt.shape) == 2:
            size = (1, 1)
        else:
            raise NotImplementedError

    gt_np = get_centroid_array(get_centroids_from_mask(gt))
    pred_np = get_centroid_array(get_centroids_from_mask(pred))

    distances = distance_matrix(gt_np, pred_np, real_dist, size)

    min_gt = np.min(distances, axis=1)
    min_pred = np.min(distances, axis=0)

    min_adpc = np.mean(min_gt)
    min_adgc = np.mean(min_pred)
    adpc_sem = sem(min_gt)
    adgc_sem = sem(min_pred)
    min_adc = (min_adgc + min_adpc) / 2
    adc_sem = (adpc_sem + adgc_sem) / 2
    if all_stats:

        mean_gt = np.mean(distances, axis=1)
        mean_pred = np.mean(distances, axis=0)

        mean_adpc = mean_gt.sum() / mean_gt.shape[0]
        mean_adgc = mean_pred.sum() / mean_pred.shape[0]

        mean_adc = (mean_adgc + mean_adpc) / 2

        truth_table = np.squeeze(np.stack([distances <= dist_thresh], axis=0))

        tp = np.sum(np.any(truth_table, axis=1))
        fp = np.sum((~truth_table).all(axis=0))
        fn = np.sum((~truth_table).all(axis=1))

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f_score = 2.0 * precision + recall / (precision + recall)
        
        return min_adc, min_adpc, min_adgc, mean_adc, f_score, precision, recall, tp, fp, fn
    else:
        return min_adc, min_adpc, min_adgc, adc_sem, adpc_sem, adgc_sem
