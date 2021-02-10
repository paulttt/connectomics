import os
import sys
path = os.path.dirname(os.path.abspath(""))+"/"
sys.path.append(path)
sys.path.insert(1, path+'cellpose/')
print(path)
from skimage import io
from CellAnalysis.evaluation import *
from CellAnalysis import visualize
from CellAnalysis.utils import *
import matplotlib.pyplot as plt
import tifffile


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


# file root
file_root = path + 'data/2photon/testing/'
kernel = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]], dtype=np.uint8)

volume_JBW = np.moveaxis(upsample_mask(io.imread(file_root + 'box6_vol.tif'), 2).astype(np.uint8), 1, -1)
seg_JBW = np.moveaxis(np.uint8(upsample_mask(io.imread(file_root + 'box6_seg.tif'), 2)), 1, -1)
seg_JBW_cp = np.moveaxis(upsample_mask(io.imread(file_root + 'box6_cp.tif'), 2).astype(np.uint8), 1, -1)

volume_PT = np.moveaxis(upsample_mask(io.imread(file_root + 'box2_vol.tif'), 2).astype(np.uint8), 1, -1)
seg_PT = np.moveaxis(upsample_mask(io.imread(file_root + 'box2_seg.tif'), 2).astype(int), 1, -1)
seg_PT_cp = np.moveaxis(upsample_mask(io.imread(file_root + 'box2_cp.tif'), 2).astype(int), 1, -1)

volume_DL = np.moveaxis(upsample_mask(io.imread(file_root + 'box4_vol.tif'), 2).astype(int), 1, -1)
seg_DL = np.moveaxis(upsample_mask(io.imread(file_root + 'box4_seg.tif'), 2).astype(int), 1, -1)
seg_DL_cp = np.moveaxis(upsample_mask(io.imread(file_root + 'box4_cp.tif'), 2).astype(int), 1, -1)

margin = 10
iou_thresh = 0.5
centroid_thresh = 10
df_JBW, df_JBW_cp = find_segment_differences(seg_JBW_cp, seg_JBW,
                                             margin=margin,
                                             iou_thresh=iou_thresh,
                                             centroid_thresh=centroid_thresh,
                                             do_centroid_thresh=False)
df_JBW_cen, df_JBW_cp_cen = find_segment_differences(seg_JBW_cp, seg_JBW,
                                             margin=margin,
                                             iou_thresh=iou_thresh,
                                             centroid_thresh=centroid_thresh,
                                             do_centroid_thresh=True)


df_PT, df_PT_cp = find_segment_differences(seg_PT_cp, seg_PT,
                                             margin=margin,
                                             iou_thresh=iou_thresh,
                                             centroid_thresh=centroid_thresh,
                                             do_centroid_thresh=False)
df_DL, df_DL_cp = find_segment_differences(seg_DL_cp, seg_DL,
                                             margin=margin,
                                             iou_thresh=iou_thresh,
                                             centroid_thresh=centroid_thresh,
                                             do_centroid_thresh=False)

img1 = np.moveaxis(draw_boundaries(volume_JBW, seg_JBW, seg_JBW_cp, draw_centroid=False, thickness=1.0, z_pos=2), 0, 2)
img2 = np.moveaxis(draw_boundaries(volume_PT, seg_PT, seg_PT_cp, draw_centroid=False, thickness=1.0, z_pos=2), 0, 2)
img3 = np.moveaxis(draw_boundaries(volume_DL, seg_DL, seg_DL_cp, draw_centroid=False, thickness=1.0, z_pos=2), 0, 2)
#added_image = cv2.addWeighted(img1, 0.4, create_centroid_mask(df_JBW, shape=img1.shape), 0.1, 0)

gt_JBW, pred_JBW, merged_JBW_gt, merged_JBW_pred = sync_instance_masks(seg_JBW, seg_JBW_cp, df_JBW, df_JBW_cp)
gt_JBW_cen, pred_JBW_cen, merged_JBW_gt_cen, merged_JBW_pred_cen = sync_instance_masks(seg_JBW, seg_JBW_cp,
                                                                                       df_JBW_cen, df_JBW_cp_cen)
gt_PT, pred_PT, merged_PT_gt, merged_PT_pred = sync_instance_masks(seg_PT, seg_PT_cp, df_PT, df_PT_cp)
gt_DL, pred_DL, merged_DL_gt, merged_DL_pred = sync_instance_masks(seg_DL, seg_DL_cp, df_DL, df_DL_cp)
np.set_printoptions(threshold=sys.maxsize)
cells_rm_JBW = (df_JBW_cp.confusion.values == 'FP').sum()   #FP
cells_add_JBW = (df_JBW_cp.confusion.values == 'FN').sum()  #FN
matches_JBW = (df_JBW_cp.confusion.values == 'TP').sum()    #TP
cells_rm_PT = (df_PT_cp.confusion.values == 'FP').sum()     #FP
cells_add_PT = (df_PT_cp.confusion.values == 'FN').sum()    #FN
matches_PT = (df_PT_cp.confusion.values == 'TP').sum()      #TP
cells_rm_DL = (df_DL_cp.confusion.values == 'FP').sum()     #FP
cells_add_DL = (df_DL_cp.confusion.values == 'FN').sum()    #FN
matches_DL = (df_DL_cp.confusion.values == 'TP').sum()      #TP

map_JBW_interval = 0.258
map_JBW_fifty = 0.874
map_JBW_seventyfive = 0.028
map_PT_interval = 0.007
map_PT_fifty = 0.008
map_PT_seventyfive = 0.008
map_Dl_interval = 0.141
map_DL_fifty = 0.710
map_DL_seventyfive = 0.003


#area_diff_JBW = calc_area_diff(df_JBW, df_JBW_cp)
#area_diff_PT = calc_area_diff(df_PT, df_PT_cp)
#area_diff_DL = calc_area_diff(df_DL, df_DL_cp)

voxel_size = (0.3, 0.3, 0.3)
adc_JBW, adpc_JBW, adgc_JBW = average_distance_between_centroids(gt_JBW.astype(np.uint8), pred_JBW.astype(np.uint8),
                                                                 voxel_size=voxel_size)
adc_PT, adpc_PT, adgc_PT = average_distance_between_centroids(gt_PT.astype(np.uint8), pred_PT.astype(np.uint8),
                                                              voxel_size=voxel_size)
adc_DL, adpc_DL, adgc_DL = average_distance_between_centroids(gt_DL.astype(np.uint8), pred_DL.astype(np.uint8),
                                                              voxel_size=voxel_size)

rows = ('Cells Added (FN)', 'Cells Removed (FP)', 'Matches (TP)', 'ADC', 'ADPC (FN Error)', 'ADGC (FP Error)',
        'mAP @IoU=0.5:0.95', 'AP @IoU=0.5', 'AP @IoU=0.75')
columns = ('Box Jonathan', 'Box Paul', 'Box Damien')

data = [[cells_add_JBW, cells_add_PT, cells_add_DL],
        [cells_rm_JBW, cells_rm_PT, cells_rm_DL],
        [matches_JBW, matches_PT, matches_DL],
        [adc_JBW, adc_PT, adc_DL],
        [adpc_JBW, adpc_PT, adpc_DL],
        [adgc_JBW, adgc_PT, adgc_DL],
        [map_JBW_interval, map_PT_interval, map_Dl_interval],
        [map_JBW_fifty, map_PT_fifty, map_DL_fifty],
        [map_JBW_seventyfive, map_PT_seventyfive, map_DL_seventyfive]]

#print('Cells that have been removed:\t {}'.format(cells_removed))
#print('Cells that have been added:\t {}'.format(cells_added))
#print('Cells that have been matched:\t {}'.format(matches))
fig = plt.figure(figsize=(18, 15))

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

ax1 = plt.subplot2grid((3, 4), (0, 0))
ax2 = plt.subplot2grid((3, 4), (0, 1))
ax3 = plt.subplot2grid((3, 4), (0, 2))
ax4 = plt.subplot2grid((3, 4), (1, 0))
ax5 = plt.subplot2grid((3, 4), (1, 1))
ax6 = plt.subplot2grid((3, 4), (1, 2))
ax7 = plt.subplot2grid((3, 4), (2, 0))
ax8 = plt.subplot2grid((3, 4), (2, 1))
ax9 = plt.subplot2grid((3, 4), (2, 2))
ax10 = plt.subplot2grid((3, 4), (0, 3))
ax11 = plt.subplot2grid((3, 4), (1, 3))
ax12 = plt.subplot2grid((3, 4), (2, 3))
#ax13 = plt.subplot2grid((3, 3), (3, 0), colspan=3)

fig.suptitle('Analysis of cellpose segmentation with respect to proofread boxes', fontsize='x-large')
plt.style.use('fivethirtyeight')
'''
colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
cell_text = []
for row in range(3):
    cell_text.append(['%d' % x for x in data[row]])
for row in range(3, 6):
    cell_text.append(['%.2f µm' % x for x in data[row]])
for row in range(6, len(rows)):
    cell_text.append(['%.3f ' % x for x in data[row]])
# Reverse colors and text labels to display the last value at the top.
colors = colors[::-1]
ax10.axis('off')
the_table = ax10.table(cellText=cell_text,
                      rowLabels=rows,
                      rowColours=colors,
                      colLabels=columns,
                      loc='upper center',
                      bbox=[0.0, 0.5, 1, 0.6])
'''
viewer1 = visualize.SliceViewer(ax1, img1)
viewer2 = visualize.SliceViewer(ax2, img2)
viewer3 = visualize.SliceViewer(ax3, img3)
viewer4 = visualize.SliceViewer(ax4, merged_JBW_gt)
viewer5 = visualize.SliceViewer(ax5, merged_PT_gt)
viewer6 = visualize.SliceViewer(ax6, merged_DL_gt)
viewer7 = visualize.SliceViewer(ax7, merged_JBW_pred)
viewer8 = visualize.SliceViewer(ax8, merged_PT_pred)
viewer9 = visualize.SliceViewer(ax9, merged_DL_pred)
viewer10 = visualize.SliceViewer(ax10, img1)
viewer11 = visualize.SliceViewer(ax11, merged_JBW_gt_cen)
viewer12 = visualize.SliceViewer(ax12, merged_JBW_pred_cen)
'''
ax1.set_title('Box Jonathan - boundaries highlighted', size=10)
ax2.set_title('Box Paul - boundaries highlighted', size=10)
ax3.set_title('Box Damien - boundaries highlighted', size=10)
ax4.set_title('Box Jonathan - color coded merged masks', size=10)
ax5.set_title('Box Paul - color coded merged masks', size=10)
ax6.set_title('Box Damien - color coded merged masks', size=10)
'''
ax1.set_axis_off()
ax2.set_axis_off()
ax3.set_axis_off()
ax4.set_axis_off()
ax5.set_axis_off()
ax6.set_axis_off()
ax7.set_axis_off()
ax8.set_axis_off()
ax9.set_axis_off()
ax10.set_axis_off()
ax11.set_axis_off()
ax12.set_axis_off()

fig.canvas.mpl_connect('scroll_event', viewer1.onscroll)
fig.canvas.mpl_connect('scroll_event', viewer2.onscroll)
fig.canvas.mpl_connect('scroll_event', viewer3.onscroll)
fig.canvas.mpl_connect('scroll_event', viewer4.onscroll)
fig.canvas.mpl_connect('scroll_event', viewer5.onscroll)
fig.canvas.mpl_connect('scroll_event', viewer6.onscroll)
fig.canvas.mpl_connect('scroll_event', viewer7.onscroll)
fig.canvas.mpl_connect('scroll_event', viewer8.onscroll)
fig.canvas.mpl_connect('scroll_event', viewer9.onscroll)
fig.canvas.mpl_connect('scroll_event', viewer10.onscroll)
fig.canvas.mpl_connect('scroll_event', viewer11.onscroll)
fig.canvas.mpl_connect('scroll_event', viewer12.onscroll)
fig.canvas.mpl_connect('key_press_event', viewer1.click_view)
fig.canvas.mpl_connect('key_press_event', viewer2.click_view)
fig.canvas.mpl_connect('key_press_event', viewer3.click_view)
fig.canvas.mpl_connect('key_press_event', viewer4.click_view)
fig.canvas.mpl_connect('key_press_event', viewer5.click_view)
fig.canvas.mpl_connect('key_press_event', viewer6.click_view)
fig.canvas.mpl_connect('key_press_event', viewer7.click_view)
fig.canvas.mpl_connect('key_press_event', viewer8.click_view)
fig.canvas.mpl_connect('key_press_event', viewer9.click_view)
fig.canvas.mpl_connect('key_press_event', viewer10.click_view)
fig.canvas.mpl_connect('key_press_event', viewer11.click_view)
fig.canvas.mpl_connect('key_press_event', viewer12.click_view)
'''
cmap1 = {1: mcolors.to_rgba('darkgreen', 1.0), 2: mcolors.to_rgba('darkgreen', 0.6),
         3: mcolors.to_rgba('darkgreen', 0.25), 4: mcolors.to_rgba('red', 1.0),
        5: mcolors.to_rgba('red', 0.5), 6: mcolors.to_rgba('gold', 1.0), 7: mcolors.to_rgba('gold', 0.5)}
labels1 = {1: 'TP cell & TP pixel', 2: 'TP cell & FN pixel', 3: 'TP cell & FP pixel', 4: 'FP cell & FP pixel ',
          5: 'FP cell & TP pixel', 6: 'FN cell & FN pixel', 7: 'FN cell & TP pixel'}
cmap2 = {1: mcolors.to_rgba('yellowgreen'), 2: mcolors.to_rgba('coral')}
labels2 = {1: 'Ground Truth Boundaries', 2: 'Prediction Boundaries'}

patches1 = [mpatches.Patch(color=cmap1[i], label=labels1[i]) for i in cmap1]
patches2 = [mpatches.Patch(color=cmap2[i], label=labels2[i]) for i in cmap2]

legend1 = plt.legend(handles=patches1, loc=3, bbox_to_anchor=(-0.07, 1.5, 0.3, 0.3), borderaxespad=0.)
plt.legend(handles=patches2, loc=3, bbox_to_anchor=(-0.1, 3.25, 0.3, 0.3), borderaxespad=0.)
plt.gca().add_artist(legend1)
textstr = '\n'.join((
    r'- Use scroll wheel to change slices in z direction',
    r'- Press #-key to change view plane'))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# place a text box in upper left in axes coords
plt.text(0.39, 3.7, textstr, fontsize=10, horizontalalignment='left', verticalalignment='top', bbox=props)
'''
fig.canvas.blit(fig.bbox)
plt.show()
