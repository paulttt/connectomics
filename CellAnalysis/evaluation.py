import numpy as np
from CellAnalysis.utils import *

# Implemented on the idea of Zudi Lin of from Pfister Lab, Harvard University.
def distance_matrix(gt, pred):
    """
    Parameters
    ----------
    gt : array-like
        Centroid vector of ground truth data of shape (N,3)
    pred : array-like
        Centroid vector of prediction data of shape (M,3)
    Returns
    -------
    array-like
        Matrix of shape (N,M) with euclidean distances between centroids.
    """
    print('Shape of ground truth centroid vector: \t \t {}'.format(gt.shape))
    print('Shape of model prediction centroid vector: \t {}'.format(pred.shape))
    print('Calculating distance matrix...')
    distance_matrix = np.linalg.norm(gt[:, None, :] - pred[None, :, :], axis=-1)
    print('Shape of calculated distance matrix: \t \t {}'.format(distance_matrix.shape))
    return distance_matrix

def average_distance_between_centroids(gt, pred, dist_thresh=0.5, all_stats=False):
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

    Returns
    -------
    min_adc:    float
        Average Distance between Centroids (ADC)

    mean_adc:   float (optional)
    f_score:    float (optional)
    precision:  float (optional)
    recall:     float (optional)
    tp:         int (optional)
    fp:         int (optional)
    fn:         int (optional)
    """
    gt_np = get_centroid_array(get_centroids_from_mask(gt))
    pred_np = get_centroid_array(get_centroids_from_mask(pred))

    distances = distance_matrix(gt_np, pred_np)

    min_gt = np.min(distances, axis=1)
    min_pred = np.min(distances, axis=0)

    min_adpc = min_gt.sum() / min_gt.shape[0]
    min_adgc = min_pred.sum() / min_pred.shape[0]

    min_adc = (min_adgc + min_adpc) / 2
    
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
        
        return min_adc, mean_adc, f_score, precision, recall, tp, fp, fn
    else:
        return min_adc