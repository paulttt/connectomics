import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage import io
from CellAnalysis.utils import precision, recall, f1, accuracy
#from utils import precision, recall, f1, accuracy


def relabel_mask(label_mask):
    unique_ids = np.unique(label_mask)
    unique_ids = unique_ids[unique_ids > 0]
    for new_id, old_id in enumerate(unique_ids, start=1):
        label_mask[label_mask == old_id] = new_id
    return label_mask


def iou_matrix(gt, pred):
    gt = gt.ravel()
    pred = pred.ravel()

    intersect = np.zeros((gt.max() + 1, pred.max() + 1), dtype=np.float64)
    for i in range(len(gt)):
        intersect[gt[i], pred[i]] += 1

    n_gt = np.sum(intersect, axis=1, keepdims=True)
    n_pred = np.sum(intersect, axis=0, keepdims=True)
    union = n_gt + n_pred - intersect
    iou = np.divide(intersect, union, out=np.zeros_like(intersect), where=np.abs(union)>1e-10)
    return iou

def match_stats_at_iou(gt, pred, iou_thresh=0.5):
    gt = relabel_mask(gt)
    pred = relabel_mask(pred)
    ious = iou_matrix(gt, pred)[1:, 1:]
    n_gt_labels, n_pred_labels = ious.shape
    max_matches = min(n_gt_labels, n_pred_labels)
    cost_matrix = -(ious >= iou_thresh).astype(float) - ious / (2 * max_matches)
    gt_idx, pred_idx = linear_sum_assignment(cost_matrix)
    matches_at_iou = ious[gt_idx, pred_idx]
    match_beyond_thresh = matches_at_iou >= iou_thresh
    tp = np.count_nonzero(match_beyond_thresh)
    fp = n_pred_labels - tp
    fn = n_gt_labels - tp
    summed_ious_for_tp = np.sum(matches_at_iou[match_beyond_thresh])
    panoptic_divisor = tp+fp/2+fn/2
    panoptic_quality = np.divide(summed_ious_for_tp,
                                 panoptic_divisor,
                                 out=np.zeros_like(summed_ious_for_tp),
                                 where=np.abs(panoptic_divisor)>1e-10)
    sq = np.divide(summed_ious_for_tp,
                   tp,
                   out=np.zeros_like(summed_ious_for_tp),
                   where=np.abs(tp)>1e-10)

    if np.abs(panoptic_divisor) > 0.4:
        rq = tp / panoptic_divisor
    else:
        rq = 0

    stats = dict(
        tp                      = tp,
        fp                      = fp,
        fn                      = fn,
        precision               = precision(tp, fp),
        recall                  = recall(tp, fn),
        accuracy                = accuracy(tp, fp, fn),
        f1                      = f1(tp, fp, fn),
        panoptic_quality        = panoptic_quality,
        segmentation_quality    = sq,
        recognition_quality     = rq
    )

    return stats


def match_stats(gt, pred):
    iou_threshs = np.arange(0.5, 0.96, 0.05)
    stat_list = []
    for thresh in iou_threshs:
        stat_list.append(match_stats_at_iou(gt, pred, iou_thresh=thresh))
    return {metric_key: [stat_dict[metric_key] for stat_dict in stat_list] for metric_key in stat_list[0]}

if __name__ == "__main__":
    gt = io.imread('gt.tif').astype(np.uint16)
    pred = io.imread('pred.tif').astype(np.uint16)
    print(match_stats(gt, pred))
