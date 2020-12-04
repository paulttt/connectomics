import os
import sys
path = os.path.dirname(os.path.abspath(""))+"/"
sys.path.append(path)
print(path)
from skimage import io
from CellAnalysis.evaluation import *
from CellAnalysis import visualize

# file root
file_root = path + 'x-ray-boxes/'

volume_JBW = io.imread(file_root + 'box6_vol.tif')
seg_JBW = io.imread(file_root + 'box6_seg.tif')
seg_JBW_stardist = io.imread(file_root + 'box6_stardist.tif')

volume_PT = io.imread(file_root + 'box2_vol.tif')
seg_PT = io.imread(file_root + 'box2_seg.tif')
seg_PT_stardist = io.imread(file_root + 'box2_stardist.tif')

volume_DL = io.imread(file_root + 'box4_vol.tif')
seg_DL = io.imread(file_root + 'box4_seg.tif')
seg_DL_stardist = io.imread(file_root + 'box4_stardist.tif')

centroid_thresh = 30
iou_thresh = 0.3
df_PT, df_PT_sd = find_segment_differences(seg_PT_stardist, seg_PT, centroid_thresh=centroid_thresh,
                                           iou_thresh=iou_thresh)
df_JBW, df_JBW_sd = find_segment_differences(seg_JBW_stardist, seg_JBW, centroid_thresh=centroid_thresh,
                                             iou_thresh=iou_thresh)
df_DL, df_DL_sd = find_segment_differences(seg_DL_stardist, seg_DL, centroid_thresh=centroid_thresh,
                                           iou_thresh=iou_thresh)

img1 = draw_boundaries(volume_JBW, seg_JBW, seg_JBW_stardist, draw_centroid=True)
img2 = draw_boundaries(volume_PT, seg_PT, seg_PT_stardist, draw_centroid=True)
img3 = draw_boundaries(volume_DL, seg_DL, seg_DL_stardist, draw_centroid=True)

_, _, merged_JBW = sync_instance_masks(seg_JBW, seg_JBW_stardist, df_JBW, df_JBW_sd)
_, _, merged_PT = sync_instance_masks(seg_PT, seg_PT_stardist, df_PT, df_PT_sd)
_, _, merged_DL = sync_instance_masks(seg_DL, seg_DL_stardist, df_DL, df_DL_sd)
visualize.SliceViewer.plot(img1, img2, img3, merged_JBW, merged_PT, merged_DL)