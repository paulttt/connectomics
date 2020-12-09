import os
import sys
path = os.path.dirname(os.path.abspath(""))+"/"
sys.path.append(path)
print(path)
from skimage import io
from CellAnalysis.evaluation import *
from CellAnalysis import visualize
from CellAnalysis.utils import *


# file root
file_root = path + 'data/x-ray-boxes/'

volume_JBW = upsample_intensity(io.imread(file_root + 'box6_vol.tif'), 4).astype(int)
seg_JBW = upsample_mask(io.imread(file_root + 'box6_seg.tif'), 4).astype(int)
seg_JBW_cp = upsample_mask(io.imread(file_root + 'box6_cp.tif'), 4).astype(int)

volume_PT = upsample_intensity(io.imread(file_root + 'box2_vol.tif'), 4).astype(int)
seg_PT = upsample_mask(io.imread(file_root + 'box2_seg.tif'), 4).astype(int)
seg_PT_cp = upsample_mask(io.imread(file_root + 'box2_cp.tif'), 4).astype(int)

volume_DL = upsample_intensity(io.imread(file_root + 'box4_vol.tif'), 4).astype(int)
seg_DL = upsample_mask(io.imread(file_root + 'box4_seg.tif'), 4).astype(int)
seg_DL_cp = upsample_mask(io.imread(file_root + 'box4_cp.tif'), 4).astype(int)

print(volume_PT)

centroid_thresh = 30
iou_thresh = 0.5
df_PT, df_PT_cp = find_segment_differences(seg_PT_cp, seg_PT, centroid_thresh=centroid_thresh,
                                           iou_thresh=iou_thresh)
df_JBW, df_JBW_cp = find_segment_differences(seg_JBW_cp, seg_JBW, centroid_thresh=centroid_thresh,
                                             iou_thresh=iou_thresh)
df_DL, df_DL_cp = find_segment_differences(seg_DL_cp, seg_DL, centroid_thresh=centroid_thresh,
                                           iou_thresh=iou_thresh)

img1 = draw_boundaries(volume_JBW, seg_JBW, seg_JBW_cp, draw_centroid=True)
img2 = draw_boundaries(volume_PT, seg_PT, seg_PT_cp, draw_centroid=True)
img3 = draw_boundaries(volume_DL, seg_DL, seg_DL_cp, draw_centroid=True)

_, _, merged_JBW = sync_instance_masks(seg_JBW, seg_JBW_cp, df_JBW, df_JBW_cp)
_, _, merged_PT = sync_instance_masks(seg_PT, seg_PT_cp, df_PT, df_PT_cp)
_, _, merged_DL = sync_instance_masks(seg_DL, seg_DL_cp, df_DL, df_DL_cp)


visualize.SliceViewer.plot(img1, img2, img3, merged_JBW, merged_PT, merged_DL)