from CellAnalysis.utils import *
import time


# Implemented on the idea of Zudi Lin from Pfister Lab, Harvard University.
def distance_matrix(gt, pred, real_distance=True, voxel_size=(1, 1, 1)):
    """
    Parameters
    ----------
    gt : array-like
        Centroid vector of ground truth data of shape (N,3)
    pred : array-like
        Centroid vector of prediction data of shape (M,3)
    real_distance: boolean (optional)
        Calculates the real euclidean distance using the voxel_size variable
    voxel_size: tuple (optional)
        length along each axis of each voxel

    Returns
    -------
    array-like
        Matrix of shape (N,M) with euclidean distances between centroids.
    """
    diffs = gt[:, None, :] - pred[None, :, :]
    if real_distance:
        return calc_real_dist(diffs, voxel_size)
    else:
        return np.linalg.norm(gt[:, None, :] - pred[None, :, :], axis=-1)


def calc_real_dist(dist, voxel_size=(1, 1, 1)):
    """
    Calculates the real euclidean norms for a given matrix and given voxel dimensions.
    Parameters
    ----------
    dist : array_like
    voxel_size : tuple

    Returns
    -------
    array_like
        real euclidean distance matrix
    """
    assert(dist.shape[-1] == 3), print('last dimension of input matrix must be of shape 3.')
    for i, axis_size in enumerate(voxel_size):
        dist[..., i] *= axis_size
    return np.linalg.norm(dist, axis=-1)


def average_distance_between_centroids(gt, pred, dist_thresh=0.5, all_stats=False, real_dist=True, voxel_size=(1, 1, 1)):
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
    voxel_size: tuple (optional)
        length along each axis of each voxel

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
    get_centroids = time.time()
    gt_np = get_centroid_array(get_centroids_from_mask(gt))
    pred_np = get_centroid_array(get_centroids_from_mask(pred))
    time_centroid = time.time()-get_centroids
    print("Runtime for extracting centroid vectors from label masks: \t\t{:.8f}".format(time_centroid))

    calc_distance = time.time()
    distances = distance_matrix(gt_np, pred_np, real_dist, voxel_size)

    min_gt = np.min(distances, axis=1)
    min_pred = np.min(distances, axis=0)

    min_adpc = min_gt.sum() / min_gt.shape[0]
    min_adgc = min_pred.sum() / min_pred.shape[0]

    min_adc = (min_adgc + min_adpc) / 2
    time_metric = time.time() - calc_distance
    print("Runtime for calculating distance matrix and ADC metrics: \t\t{:.8f}".format(time_metric))
    print("Runtimes varies by a factor of: \t\t\t\t\t{:.3f}\n".format(time_centroid/time_metric))
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
        return min_adc, min_adpc, min_adgc