import numpy as np
import pandas as pd
from skimage import measure

def get_centroids_from_mask(seg):
    df = pd.DataFrame.from_dict(measure.regionprops_table(seg, properties=['centroid']))
    df['centroid'] = df.apply(lambda x: [x['centroid-0'], x['centroid-1'], x['centroid-2']], axis=1)
    return df

def get_centroid_array(df):
    a = df['centroid'].to_numpy()
    a = np.stack(a, axis=0)
    return a

def distance_matrix(gt, pred):
    print('Shape of ground truth centroid vector: \t \t {}'.format(gt.shape))
    print('Shape of model prediction centroid vector: \t {}'.format(pred.shape))
    print('Calculating distance matrix...')
    distance_matrix = np.linalg.norm(gt[:, None, :] - pred[None, :, :], axis=-1)
    print('Shape of calculated distance matrix: \t \t {}'.format(distance_matrix.shape))
    return distance_matrix

def average_distance_between_centroids(gt, pred, dist_thresh=0.5, all_stats=False):
    """
    Function that computes all the metrics 
    Parameters
    ----------
    gt : array-like
        ground truth segmentation mask
    pred : array-like
        predicted segmentation mask by the model
    dist_thresh : float
        threshold for deciding if cells are the same w.r.t. centroid offset
    all_stats : boolean
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