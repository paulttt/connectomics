import sys
import pdb
from skimage import io
import matplotlib.pyplot as plt
from CellAnalysis.utils import *
import pandas as pd
from imageio import volwrite, volread
from CellAnalysis.evaluation import *

if __name__ == "__main__":


    # file root
    file_root = 'x-ray-boxes/'

    volume_JBW = io.imread(file_root + 'box6_vol.tif')
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

    #df_JBW_seg = get_centroids_from_mask(seg_JBW_stardist)
    #df_JBW = get_centroids_from_mask(seg_JBW)
    #dist_matrix = distance_matrix(get_centroid_array(df_JBW), get_centroid_array(df_JBW_seg))
    #fig, ax = plt.figure()
    ADC = average_distance_between_centroids(seg_JBW, seg_JBW)
    print(ADC)

    '''
    centroid_thresh = 30
    iou_thresh = 0.3
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
    '''
    '''
    fig, ax = plt.subplots()
    img = highlight_boundary(volume_JBW, seg_JBW, mode='gt')
    img = highlight_boundary(img, seg_JBW_stardist, mode='pred')
    im = ax.imshow(img[5])
    plt.show()
    #_, _, merged_JBW = sync_instance_masks(seg_JBW, seg_JBW_stardist, df_JBW, df_JBW_sd)
    #visualize.SliceViewer.plot(merged_JBW)
    '''
    '''
    volwrite('visualize_segs_JBW.tiff', merged_JBW)
    _, _, merged_PT = sync_instance_masks(seg_PT, seg_PT_stardist, df_PT, df_PT_sd)
    volwrite('visualize_segs_PT.tiff', merged_PT)
    _, _, merged_MM = sync_instance_masks(seg_MM, seg_MM_stardist, df_MM, df_MM_sd)
    volwrite('visualize_segs_MM.tiff', merged_MM)
    _, _, merged_DL = sync_instance_masks(seg_DL, seg_DL_stardist, df_DL, df_DL_sd)
    volwrite('visualize_segs_DL.tiff', merged_DL)
    _, _, merged_RM = sync_instance_masks(seg_RM, seg_RM_stardist, df_RM, df_RM_sd)
    volwrite('visualize_segs_RM.tiff', merged_RM)
    '''
    #multi_slice_viewer(merged_JBW)
    '''
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
    #print(calc_centroid_diff(df_PT, df_PT_sd))
    fig1 = plt.figure()
    # fig.title("histogramms: area difference compared to stardist segmentation")
    ax1 = fig1.add_subplot(2, 3, 1)
    ax2 = fig1.add_subplot(2, 3, 2)
    ax3 = fig1.add_subplot(2, 3, 3)
    ax4 = fig1.add_subplot(2, 3, 4)
    ax5 = fig1.add_subplot(2, 3, 5)
    ax1.hist(area_diff_JBW.values)  # , bins = JBW_bins)
    ax1.set_title("JBW - Cellpose")
    # ax1.set_xlabel('intensity bins')

    ax2.hist(area_diff_PT.values)  # , bins = PT_bins)
    ax2.set_title("PT - Cellpose")

    ax3.hist(area_diff_MM.values)  # , bins = PT_bins)
    ax3.set_title("MM - Cellpose")

    ax4.hist(area_diff_DL.values)  # , bins = PT_bins)
    ax4.set_title("DL - Cellpose")

    ax5.hist(area_diff_RM.values)  # , bins = PT_bins)
    ax5.set_title("RM - Cellpose")
    # ax2.set_xlabel('intensity bins')
    
    plt.subplots_adjust(left=0.125,
                        bottom=0.1, 
                        right=1, 
                        top=1, 
                        wspace=2, 
                        hspace=0.2)
    
    fig2 = plt.figure()
    # fig.title("histogramms: area difference compared to Cellpose segmentation")
    ax6 = fig2.add_subplot(2, 3, 1)
    ax7 = fig2.add_subplot(2, 3, 2)
    ax8 = fig2.add_subplot(2, 3, 3)
    ax9 = fig2.add_subplot(2, 3, 4)
    ax10 = fig2.add_subplot(2, 3, 5)

    
    plt.tight_layout()

    plt.show()
    '''