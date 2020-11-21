import sys
import pdb
from skimage import io
import matplotlib.pyplot as plt
from CellAnalysis.utils import find_segment_differences

if __name__ == "__main__":


    # file root
    file_root = 'x-ray-boxes/'

    # volume_JBW = io.imread('box6_vol.tif')
    seg_JBW = io.imread(file_root + 'box6_seg.tif')
    seg_JBW_stardist = io.imread(file_root + 'box6_stardist.tif')

    # volume_PT = io.imread('box2_vol.tif')
    seg_PT = io.imread(file_root + 'box2_seg.tif')
    seg_PT_stardist = io.imread(file_root + 'box2_stardist.tif')

    # volume_MM = io.imread('box3_vol.tif')
    seg_MM = io.imread(file_root + 'box3_seg.tif')
    seg_MM_stardist = io.imread(file_root + 'box3_stardist.tif')

    # volume_DL = io.imread('box4_vol.tif')
    seg_DL = io.imread(file_root + 'box4_seg.tif')
    seg_DL_stardist = io.imread(file_root + 'box4_stardist.tif')

    # volume_RM = io.imread('box5_vol.tif')
    seg_RM = io.imread(file_root + 'box5_seg.tif')
    seg_RM_stardist = io.imread(file_root + 'box5_stardist.tif')

    centroid_thresh = 10
    iou_thresh = 0.8
    df_PT, df_PT_sd = find_segment_differences(seg_PT_stardist, seg_PT, centroid_thresh=centroid_thresh,
                                                 iou_thresh=iou_thresh)
    df_JBW, df_JBW_sd = find_segment_differences(seg_JBW_stardist, seg_JBW, centroid_thresh=centroid_thresh,
                                                 iou_thresh=iou_thresh)
    df_MM, df_MM_sd = find_segment_differences(seg_MM_stardist, seg_MM, centroid_thresh=centroid_thresh,
                                                 iou_thresh=iou_thresh)
    df_DL, df_DL_sd = find_segment_differences(seg_DL_stardist, seg_DL, centroid_thresh=centroid_thresh,
                                                 iou_thresh=iou_thresh)
    df_RM, df_RM_sd = find_segment_differences(seg_RM_stardist, seg_RM, centroid_thresh=centroid_thresh,
                                                 iou_thresh=iou_thresh)

    area_JBW = df_JBW['area']
    area_JBW_sd = df_JBW_sd['area']

    area_PT = df_PT['area']
    area_PT_sd = df_PT_sd['area']

    area_MM = df_MM['area']
    area_MM_sd = df_MM_sd['area']

    area_DL = df_DL['area']
    area_DL_sd = df_DL_sd['area']

    area_RM = df_RM['area']
    area_RM_sd = df_RM_sd['area']

    print(area_JBW.unique())

    print(area_JBW.shape)
    print(area_JBW_sd.shape)

    print(area_PT.shape)
    print(area_PT_sd.shape)

    print(area_MM.shape)
    print(area_MM_sd.shape)

    print(area_DL.shape)
    print(area_DL_sd.shape)

    print(area_RM.shape)
    print(area_RM_sd.shape)

    area_diff_JBW = area_JBW - area_JBW_sd
    area_diff_PT = area_PT - area_PT_sd
    area_diff_MM = area_MM - area_MM_sd
    area_diff_DL = area_DL - area_DL_sd
    area_diff_RM = area_RM - area_RM_sd

    fig = plt.figure()
    # fig.title("histogramms: area difference compared to stardist segmentation")
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax1.hist(area_diff_JBW.values)  # , bins = JBW_bins)
    ax1.set_title("JBW - stardist")
    # ax1.set_xlabel('intensity bins')

    ax2.hist(area_diff_PT.values)  # , bins = PT_bins)
    ax2.set_title("PT - stardist")

    ax3.hist(area_diff_MM.values)  # , bins = PT_bins)
    ax3.set_title("MM - stardist")

    ax4.hist(area_diff_DL.values)  # , bins = PT_bins)
    ax4.set_title("DL - stardist")

    ax5.hist(area_diff_RM.values)  # , bins = PT_bins)
    ax5.set_title("RM - stardist")
    # ax2.set_xlabel('intensity bins')
    '''
    plt.subplots_adjust(left=0.125,
                        bottom=0.1, 
                        right=1, 
                        top=1, 
                        wspace=2, 
                        hspace=0.2)
    '''
    plt.tight_layout()

    plt.show()