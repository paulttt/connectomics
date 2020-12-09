import sys
import pdb
from skimage import io
import matplotlib.pyplot as plt
from CellAnalysis.utils import *
import pandas as pd
from imageio import volwrite, volread
from CellAnalysis.evaluation import *
from CellAnalysis import visualize

if __name__ == "__main__":

    data = dataloader("data/pred_pyc.h5")
    data = data['main']
    mask = dataloader("data/box6_label.h5")
    mask = mask['main']


    # file root
    file_root = 'data/x-ray-boxes/'

    volume_JBW = io.imread(file_root + 'box6_vol.tif')
    seg_JBW = io.imread(file_root + 'box6_seg.tif')
    seg_JBW_cp = io.imread(file_root + 'box6_cp.tif')
    print("Calculate stats for pytorch_connectomics data...")
    adc_pyc, adpc_pyc, adgc_pyc = average_distance_between_centroids(mask, data, all_stats=False, voxel_size=(0.3, 0.3, 0.3))
    print("Average Distance between Centroid score for pytorch_connectomics: \t{:.4f} µm".format(adc_pyc))
    print("Error induced due to FN (ADPC) for pytorch_connectomics: \t\t{:.4f} µm".format(adpc_pyc))
    print("Error induced due to FP (ADGC) for pytorch_connectomics: \t\t{:.4f} µm".format(adgc_pyc))
    print("\nCalculate stats for Cellpose data...")
    adc_cp, adpc_cp, adgc_cp = average_distance_between_centroids(seg_JBW, seg_JBW_cp, all_stats=False, voxel_size=(0.3, 0.3, 0.3))
    print("Average Distance between Centroid score for Cellpose: \t\t\t{:.4f} µm".format(adc_cp))
    print("Error induced due to FN (ADPC) for Cellpose: \t\t\t\t{:.4f} µm".format(adpc_cp))
    print("Error induced due to FP (ADGC) for Cellpose: \t\t\t\t{:.4f} µm".format(adgc_cp))
    """
    volume_PT = io.imread(file_root + 'box2_vol.tif')
    seg_PT = io.imread(file_root + 'box2_seg.tif')
    seg_PT_cp = io.imread(file_root + 'box2_cp.tif')

    # volume_MM = io.imread(file_root + 'box3_vol.tif')
    seg_MM = io.imread(file_root + 'box3_seg.tif')
    seg_MM_cp = io.imread(file_root + 'box3_cp.tif')

    volume_DL = io.imread(file_root + 'box4_vol.tif')
    seg_DL = io.imread(file_root + 'box4_seg.tif')
    seg_DL_cp = io.imread(file_root + 'box4_cp.tif')

    # volume_RM = io.imread(file_root + 'box5_vol.tif')
    seg_RM = io.imread(file_root + 'box5_seg.tif')
    seg_RM_cp = io.imread(file_root + 'box5_cp.tif')
    """
    '''
    fig1 = plt.figure()
    # fig.title("histogramms: area difference compared to cp segmentation")
    ax1 = fig1.add_subplot(2, 3, 1)
    ax2 = fig1.add_subplot(2, 3, 2)
    ax3 = fig1.add_subplot(2, 3, 3)
    ax4 = fig1.add_subplot(2, 3, 4)
    ax5 = fig1.add_subplot(2, 3, 5)
    '''
    '''
    #df_JBW_seg = get_centroids_from_mask(seg_JBW_cp)
    #df_JBW = get_centroids_from_mask(seg_JBW)
    #dist_matrix = distance_matrix(get_centroid_array(df_JBW), get_centroid_array(df_JBW_seg))
    #fig, ax = plt.figure()
    ADC = average_distance_between_centroids(seg_JBW, seg_JBW)
    print(ADC)
    '''
    """
    centroid_thresh = 30
    iou_thresh = 0.3
    df_PT, df_PT_cp = find_segment_differences(seg_PT_cp, seg_PT, centroid_thresh=centroid_thresh,
                                                 iou_thresh=iou_thresh)
    df_JBW, df_JBW_cp = find_segment_differences(seg_JBW_cp, seg_JBW, centroid_thresh=centroid_thresh,
                                                 iou_thresh=iou_thresh)
    df_MM, df_MM_cp = find_segment_differences(seg_MM_cp, seg_MM, centroid_thresh=centroid_thresh,
                                                 iou_thresh=iou_thresh)
    df_DL, df_DL_cp = find_segment_differences(seg_DL_cp, seg_DL, centroid_thresh=centroid_thresh,
                                                 iou_thresh=iou_thresh)
    df_RM, df_RM_cp = find_segment_differences(seg_RM_cp, seg_RM, centroid_thresh=centroid_thresh,
                                                 iou_thresh=iou_thresh)


    #fig, ax = plt.subplots()
    img1 = draw_boundaries(volume_JBW, seg_JBW, seg_JBW_cp, draw_centroid=True)
    img2 = draw_boundaries(volume_PT, seg_PT, seg_PT_cp, draw_centroid=True)
    img3 = draw_boundaries(volume_DL, seg_DL, seg_DL_cp, draw_centroid=True)
    #img = highlight_boundary(volume_JBW, seg_JBW, mode='gt')
    #img = highlight_boundary(img, seg_JBW_cp, mode='pred')
    #im = ax.imshow(img[5])
    #plt.show()
    _, _, merged_JBW = sync_instance_masks(seg_JBW, seg_JBW_cp, df_JBW, df_JBW_cp)
    _, _, merged_PT = sync_instance_masks(seg_PT, seg_PT_cp, df_PT, df_PT_cp)
    _, _, merged_DL = sync_instance_masks(seg_DL, seg_DL_cp, df_DL, df_DL_cp)
    visualize.SliceViewer.plot(img1, img2, img3, merged_JBW, merged_PT, merged_DL)
    """
    '''
    volwrite('visualize_segs_JBW.tiff', merged_JBW)
    _, _, merged_PT = sync_instance_masks(seg_PT, seg_PT_cp, df_PT, df_PT_cp)
    volwrite('visualize_segs_PT.tiff', merged_PT)
    _, _, merged_MM = sync_instance_masks(seg_MM, seg_MM_cp, df_MM, df_MM_cp)
    volwrite('visualize_segs_MM.tiff', merged_MM)
    _, _, merged_DL = sync_instance_masks(seg_DL, seg_DL_cp, df_DL, df_DL_cp)
    volwrite('visualize_segs_DL.tiff', merged_DL)
    _, _, merged_RM = sync_instance_masks(seg_RM, seg_RM_cp, df_RM, df_RM_cp)
    volwrite('visualize_segs_RM.tiff', merged_RM)
    '''
    #multi_slice_viewer(merged_JBW)
    '''
    area_JBW = df_JBW['area']
    area_JBW_cp = df_JBW_cp['area']

    area_PT = df_PT['area']
    area_PT_cp = df_PT_cp['area']

    area_MM = df_MM['area']
    area_MM_cp = df_MM_cp['area']

    area_DL = df_DL['area']
    area_DL_cp = df_DL_cp['area']

    area_RM = df_RM['area']
    area_RM_cp = df_RM_cp['area']



    print(area_JBW.shape)
    print(area_JBW_cp.shape)

    print(area_PT.shape)
    print(area_PT_cp.shape)

    print(area_MM.shape)
    print(area_MM_cp.shape)

    print(area_DL.shape)
    print(area_DL_cp.shape)

    print(area_RM.shape)
    print(area_RM_cp.shape)

    area_diff_JBW = area_JBW - area_JBW_cp
    area_diff_PT = area_PT - area_PT_cp
    area_diff_MM = area_MM - area_MM_cp
    area_diff_DL = area_DL - area_DL_cp
    area_diff_RM = area_RM - area_RM_cp
    #print(calc_centroid_diff(df_PT, df_PT_cp))
    fig1 = plt.figure()
    # fig.title("histogramms: area difference compared to cellpose segmentation")
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