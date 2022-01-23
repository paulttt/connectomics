from CellAnalysis.adc_metric import average_distance_between_centroids as adc_score
from CellAnalysis.utils import iou, load_sorted
from CellAnalysis.iou_matching import match_stats
from mAP_3Dvolume.vol3d_eval_custom import VOL3Deval
from mAP_3Dvolume.vol3d_util_custom import seg_iou2d_sorted, seg_iou3d_sorted, unique_chunk
from scipy.stats import sem
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import os
import warnings


class Eval:
    def __init__(self, gt, pred, resolution=(1, 1, 1), model_name='prediction model', name_data='test data'):
        """
        This class evaluates the quality of a predicted segmentation mask by computing IoU-related scores like Average
        Precision (AP), mean average precision (mAP), and mask offset metrics like Average Distance between
        Centroids (ADC). The class can handle both 2D and 3D datasets. The data is expected to be of type numpy arrays.
        It is also possible to input lists of numpy arrays. Pay attention that the order of entries in the input lists
        must match:
        E.g.  gt = [gt_mask_img_1,   gt_mask_img_2,   ..., gt_mask_img_n]
        and pred = [pred_mask_img_1, pred_mask_img_2, ..., pred_mask_img_n].

        Parameters
        ----------
        gt : numpy array or list of numpy arrays
            Ground Truth segmentation mask(s) with int labels from 1 to n and 0 as background. Either 2D or 3D.
        pred : numpy array or list of numpy arrays
            Prediction segmentation mask(s) with int labels from 1 to n and 0 as background. Either 2D or 3D.
        resolution : tuple (optional)
            specifies the spatial resolution of the data (voxel/pixel-size).
        model_name : string (optional)
            Name of the model to evaluate.
        name_data : string (optional)
            Name/Type of the data to evaluate.
        """
        self.size = resolution

        if isinstance(gt, np.ndarray) and isinstance(pred, np.ndarray):
            assert gt.shape == pred.shape, 'Ground Truth and Prediction masks must be of same shape. Shapes {} ' \
                                           'and {} were given'.format(gt.shape, pred.shape)

            if len(gt.shape) == 2 and len(resolution) != 2:
                self.size = (1, 1)

        elif isinstance(gt, list) and isinstance(pred, list):
            assert len(gt) == len(pred), 'Mismatch in number of GT and Pred instances. Found {} gt instances and {} ' \
                                         'prediction instances.'.format(len(gt), len(pred))

            for pred_instance, gt_instance in zip(pred, gt):
                assert pred_instance.shape == gt_instance.shape, 'Each instance pair must be of same shape. Shapes {} '\
                                                                 'and {} were given.'.format(pred_instance.shape,
                                                                                            gt_instance.shape)
        else:
            raise TypeError('Mask inputs must be of type numpy array '
                            'or lists of numpy arrays (e.g.: [gt_1, gt_2, ..., gt_n]).')

        self.gt = gt
        self.pred = pred
        self.map = {}
        self.map_full = {}
        self.match_stats = {}
        self.adc = {}
        self.adc_full = {}
        self.name = model_name
        self.data = name_data

    def accumulate(self):
        """
        Computes and accumulates the relevant metrics from the segmented masks. Stores dictionaries in the adc and map
        variables with the relevant computed metrics.
        """
        adc_keys = ['adc', 'adpc', 'adgc', 'adc_sem', 'adpc_sem', 'adgc_sem']
        if isinstance(self.pred, np.ndarray):
            self.map = self.get_map_scores(self.pred, self.gt)
            self.adc = dict(zip(adc_keys, adc_score(self.gt, self.pred, size=self.size)))
            self.match_stats = match_stats(self.gt, self.pred)
        elif isinstance(self.pred, list):
            map_list = []
            adc_list = []
            match_stat_list = []
            index_list = []
            # iterating over all images in the provided list
            for idx, (pred_instance, gt_instance) in enumerate(zip(self.pred, self.gt)):
                if iou(pred_instance, gt_instance) > 0.0:
                    map_list.append(self.get_map_scores(pred_instance, gt_instance))
                    adc_list.append(adc_score(gt_instance, pred_instance, size=self.size))
                    match_dict = match_stats(gt_instance, pred_instance)
                    match_dict['PQ @ 0.5'] = match_dict['panoptic_quality'][0]
                    match_dict['PQ @ 0.75'] = match_dict['panoptic_quality'][5]
                    match_dict['PQ @ 0.9'] = match_dict['panoptic_quality'][-2]
                    match_dict['PQ @ 0.5:0.95'] = np.mean(match_dict['panoptic_quality'])

                    match_dict['SQ @ 0.5'] = match_dict['segmentation_quality'][0]
                    match_dict['SQ @ 0.75'] = match_dict['segmentation_quality'][5]
                    match_dict['SQ @ 0.9'] = match_dict['segmentation_quality'][-2]
                    match_dict['SQ @ 0.5:0.95'] = np.mean(match_dict['segmentation_quality'])

                    match_dict['RQ @ 0.5'] = match_dict['recognition_quality'][0]
                    match_dict['RQ @ 0.75'] = match_dict['recognition_quality'][5]
                    match_dict['RQ @ 0.9'] = match_dict['recognition_quality'][-2]
                    match_dict['RQ @ 0.5:0.95'] = np.mean(match_dict['recognition_quality'])

                    match_stat_list.append(match_dict)
                    index_list.append(idx)
                else:
                    match_dict = match_stats(gt_instance, pred_instance)
                    map_list.append(dict({'mAP @ 0.5:0.95': 0.0,
                                          'AP @ 0.5': 0.0,
                                          'AP @ 0.75': 0.0,
                                          'AP @ 0.9': 0.0,
                                          'Average Precision': np.zeros((10,1))
                                          }))
                    match_dict['PQ @ 0.5'] = match_dict['panoptic_quality'][0]
                    match_dict['PQ @ 0.75'] = match_dict['panoptic_quality'][5]
                    match_dict['PQ @ 0.9'] = match_dict['panoptic_quality'][-2]
                    match_dict['PQ @ 0.5:0.95'] = np.mean(match_dict['panoptic_quality'])
                    match_stat_list.append(match_dict)
                    index_list.append(idx)
                    print("Instance no. {} in your data from model {} shows no IoU matches with the provided "
                          "ground truth. No ADC scores will be calculated for this sample.".format(idx, self.name))

            def _average_arrays_in_df(df, column_name):
                num_instance = df.shape[0]
                temp = np.array(df[column_name].iloc[0])
                for row in range(1, num_instance):
                    temp += np.array(df[column_name].iloc[row])
                return temp / num_instance
            # average stats for all boxes
            df_stats = pd.DataFrame(match_stat_list, index_list)
            # average AP scores for all test instances
            df_map = pd.DataFrame(map_list, index_list)
            '''
            num_boxes = df_map.shape[0]
            temp_ap = np.array(df_map['Average Precision'].iloc[0])
            temp_pr = np.array(df_map['Precision'].iloc[0])
            temp_rc = np.array(df_map['Recall'].iloc[0])
            temp_sc = np.array(df_map['Scores'].iloc[0])
            for row in range(1, num_boxes):
                temp_ap += np.array(df_map['Average Precision'].iloc[row])
                temp_pr += np.array(df_map['Precision'].iloc[row])
                temp_rc += np.array(df_map['Recall'].iloc[row])
                temp_sc += np.array(df_map['Scores'].iloc[row])
            avg_ap = temp_ap / num_boxes
            avg_pr = temp_pr / num_boxes
            avg_rc = temp_rc / num_boxes
            avg_sc = temp_sc / num_boxes
            '''
            self.map_full = df_map.to_dict()
            self.map = df_map.mean(numeric_only=True).to_dict()

            for metric in ['Average Precision', 'Precision', 'Recall', 'Scores']:
                self.map[metric] = _average_arrays_in_df(df_map, column_name=metric)
            for metric in df_stats.columns:
                self.match_stats[metric] = _average_arrays_in_df(df_stats, column_name=metric)


            '''
            self.map['Average Precision'] = avg_ap
            self.map['Precision'] = avg_pr
            self.map['Recall'] = avg_rc
            self.map['Scores'] = avg_sc
            '''
            ap_boxes = np.array([map_dict['Average Precision'][:, 0] for map_dict in map_list])
            if ap_boxes.shape[0] < 2:
                temp = np.empty((ap_boxes.shape[1],))
                temp.fill(np.nan)
                self.map['ap_sem'] = temp
                self.map['map_sem'] = np.nan
                self.match_stats['pq_sem'] = temp
                self.match_stats['mpq_sem'] = np.nan
                self.match_stats['sq_sem'] = temp
                self.match_stats['msq_sem'] = np.nan
                self.match_stats['rq_sem'] = temp
                self.match_stats['mrq_sem'] = np.nan
            else:
                # calculate the Standard Error of the Mean for the different AP scores across different test instances
                self.map['ap_sem'] = sem(
                    np.array([map_dict['Average Precision'][:, 0] for map_dict in map_list]), axis=0)
                self.map['map_sem'] = sem(
                    np.array([np.mean(map_dict['Average Precision'][:, 0]) for map_dict in map_list]), axis=0)
                # panoptic quality
                self.match_stats['pq_sem'] = sem(
                    np.array([stat_dict['panoptic_quality'] for stat_dict in match_stat_list]), axis=0)
                self.match_stats['mpq_sem'] = sem(
                    np.array([np.mean(stat_dict['panoptic_quality']) for stat_dict in match_stat_list]), axis=0)
                # segmentation quality
                self.match_stats['sq_sem'] = sem(
                    np.array([stat_dict['segmentation_quality'] for stat_dict in match_stat_list]), axis=0)
                self.match_stats['msq_sem'] = sem(
                    np.array([np.mean(stat_dict['segmentation_quality']) for stat_dict in match_stat_list]), axis=0)
                # recognition quality
                self.match_stats['rq_sem'] = sem(
                    np.array([stat_dict['recognition_quality'] for stat_dict in match_stat_list]), axis=0)
                self.match_stats['mrq_sem'] = sem(
                    np.array([np.mean(stat_dict['recognition_quality']) for stat_dict in match_stat_list]), axis=0)

            self.adc = pd.DataFrame(adc_list, columns=adc_keys).mean().to_dict()
            self.adc_full = pd.DataFrame(adc_list, columns=adc_keys).to_dict()
        else:
            raise TypeError('Mask inputs must be of type numpy array '
                            'or lists of numpy arrays (e.g.: [gt_1, gt_2, ..., gt_n]).')

    def get_all_stats(self, as_pandas_df=False):
        if not self.adc or not self.map:
            print('Please run accumulate() method first.')
        if as_pandas_df:
            raise NotImplementedError
            # return pd.DataFrame({**self.map, **self.adc})
        else:
            return {**self.map, **self.adc}

    def plot_ap(self, ax, color='b', label='model', linestyle='-', marker='o', error_band=False):
        """
        Plots a graph of the average precision values for IoU thresholds from 0.5 to 0.95.

        Parameters
        ----------
        ax : matplotlib.axes.Axes object
            figure element where the plot should be visualized in.
        color : string (optional)
        label : string (optional)
        linestyle : string (optional)
        marker : string (optional)
        error_band : boolean (optional)
            plots the standard error of the mean around the AP curve. Only possible if several test instances inputted.

        Returns
        -------
        matplotlib.axes.Axes object
        """
        if not self.map:
            print('Please run accumulate() method first.')
        ap = self.map['Average Precision'][:, 0]
        x = np.arange(0.5, 1.0, 0.05)
        ax.plot(x, ap, linestyle=linestyle, marker=marker, color=color, label=label)
        if isinstance(self.pred, np.ndarray) and error_band:
            print('Plotting the error band not possible with only one test instance.')
        elif error_band:
            sem_error = self.map['ap_sem']
            ax.fill_between(x, ap-sem_error, ap+sem_error, color=color, alpha=0.2)
        return ax

    def box_plots(self, ax=None, metric='adc', title=None, y_label=None, set_legend=True, fontsize=12,
                  legend_font_size=8, legend_loc='best'):
        if ax is None:
            fig, ax = plt.subplots()
        if title is not None:
            ax.set_title(title, fontsize=fontsize)
        if y_label is not None:
            ax.set_ylabel(y_label)

        palette = sns.color_palette('colorblind')
        data = []
        custom_lines = []
        if metric == 'adc':
            metric_names = ['adc', 'adpc', 'adgc']
        elif metric == 'ap':
            metric_names = ['mAP @ 0.5:0.95', 'AP @ 0.5', 'AP @ 0.75', 'AP @ 0.9']
            if isinstance(self.pred, np.ndarray):
                raise UserWarning('No error variance can be shown, because only test instance was given.')
        elif metric == 'pq':
            metric_names = ['PQ @ 0.5:0.95', 'PQ @ 0.5', 'PQ @ 0.75', 'PQ @ 0.9']
            if isinstance(self.pred, np.ndarray):
                raise UserWarning('No error variance can be shown, because only test instance was given.')
        else:
            raise KeyError('Metric is not supported. Either choose {} or {} instead.'.format('adc', 'ap'))
        for i, metric_key in enumerate(metric_names):
            if metric == 'adc':
                data.append(list(self.adc_full[metric_key].values()))
            elif metric == 'ap':
                data.append(list(self.map_full[metric_key].values()))
            elif metric == 'pq':
                data.append(list(self.match_stats[metric_key].values()))
            else:
                print('Choose from existing metric.')
            custom_lines.append(Line2D([0], [0], color=palette[i], lw=4))

        sns.set(context='notebook')
        sns.boxplot(ax=ax, data=data, width=.18, palette=palette)
        if set_legend:
            ax.legend(custom_lines, metric_names, fontsize=legend_font_size, loc=legend_loc)
        ax.set_xticks([])
        return ax

    def summarize(self, title=None, save_to_file=None, figsize=None, error_bands=False, metric='ap'):
        if title is None:
            title = 'Summarized Stats for Segmentation'
        if figsize is None:
            figsize = (15, 8.5)
        sns.set()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(title, fontsize=24)
        font_size = 12
        self.plot_ap(ax1, error_band=error_bands)
        if metric == 'ap':
            ax1.set_title('Average Precision across IoU thresholds')
            title = 'Average Precision'
        elif metric == 'pq':
            ax1.set_title('Panoptic Quality across IoU thresholds')
            title = 'Panoptic Quality'
        ax2 = self.box_plots(ax2, metric=metric, title=title, legend_font_size=12, fontsize=14)
        ax2.set_ylim([0.0, 1.1])
        ax3 = self.box_plots(ax3, metric='adc', title='Average Distance between Centroids',
                             legend_font_size=12, fontsize=14)
        ax3.set_ylabel('Euclidean Distance [µm]', fontsize=font_size)
        mpl.rcParams['xtick.labelsize'] = 12
        mpl.rcParams['ytick.labelsize'] = 12

        if save_to_file is not None and save_to_file is not False:
            plt.savefig(save_to_file + '.png')

    def print_adc_scores(self):
        """
        prints the average distances between centroids (adc) metrics in the console.
        """
        print(''.ljust(100, '-'))
        print('Compute distance metrics for {} for {}...\n'.format(self.name, self.data))
        print('Average Distance between Centroids for {}:'.format(self.name).ljust(87, ' ')
              + '{:.3f} ± {:.3f}'.format(self.adc['adc'], self.adc['adc_sem']))
        print('Average Distance between Prediction Centroids for {}:'.format(self.name).ljust(87, ' ')
              + '{:.3f} ± {:.3f}'.format(self.adc['adpc'], self.adc['adpc_sem']))
        print('Average Distance between Ground Truth Centroids for {}:'.format(self.name).ljust(87, ' ')
              + '{:.3f} ± {:.3f}'.format(self.adc['adgc'], self.adc['adgc_sem']))
        print(''.ljust(100, '-'))

    def print_ap_scores(self):
        """
        prints the most relevant scores from the average precision (AP) metric in the console.
        """
        print(''.ljust(100, '-'))
        print('Compute AP scores for {} for {}...\n'.format(self.name, self.data))
        if np.isnan(self.map['map_sem']):
            print('Average Precision @ IoU=0.5 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f}'.format(self.map['AP @ 0.5']))
            print('Average Precision @ IoU=0.75 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f}'.format(self.map['AP @ 0.75']))
            print('Average Precision @ IoU=0.9 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f}'.format(self.map['AP @ 0.9']))
            print('Average Precision @ IoU=0.5:0.95 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f}'.format(self.map['mAP @ 0.5:0.95']))
        else:
            print('Average Precision @ IoU=0.5 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f} ± {:.3f}'.format(self.map['AP @ 0.5'], self.map['ap_sem'][0]))
            print('Average Precision @ IoU=0.75 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f} ± {:.3f}'.format(self.map['AP @ 0.75'], self.map['ap_sem'][5]))
            print('Average Precision @ IoU=0.9 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f} ± {:.3f}'.format(self.map['AP @ 0.9'], self.map['ap_sem'][8]))
            print('Average Precision @ IoU=0.5:0.95 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f} ± {:.3f}'.format(self.map['mAP @ 0.5:0.95'], self.map['map_sem']))
        print(''.ljust(100, '-'))

    def print_pq_scores(self):
        """
        prints the most relevant scores from the panoptic quality (PQ) metric in the console.
        """
        print(''.ljust(100, '-'))
        print('Compute PQ scores for {} for {}...\n'.format(self.name, self.data))
        if np.isnan(self.match_stats['mpq_sem']):
            print('Panoptic Quality @ IoU=0.5 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f}'.format(self.match_stats['PQ @ 0.5']))
            print('Segmentation Quality @ IoU=0.5 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f}'.format(self.match_stats['SQ @ 0.5']))
            print('Recognition Quality @ IoU=0.5 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f}'.format(self.match_stats['RQ @ 0.5']))

            print('Panoptic Quality @ IoU=0.75 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f}'.format(self.match_stats['PQ @ 0.75']))
            print('Segmentation Quality @ IoU=0.75 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f}'.format(self.match_stats['SQ @ 0.75']))
            print('Recognition Quality @ IoU=0.75 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f}'.format(self.match_stats['RQ @ 0.75']))

            print('Panoptic Quality @ IoU=0.9 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f}'.format(self.match_stats['PQ @ 0.9']))
            print('Segmentation Quality @ IoU=0.9 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f}'.format(self.match_stats['SQ @ 0.9']))
            print('Recognition Quality @ IoU=0.9 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f}'.format(self.match_stats['RQ @ 0.9']))

            print('Panoptic Quality @ IoU=0.5:0.95 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f}'.format(self.match_stats['PQ @ 0.5:0.95']))
            print('Segmentation Quality @ IoU=0.0.5:0.95 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f}'.format(self.match_stats['SQ @ 0.5:0.95']))
            print('Recognition Quality @ IoU=0.5:0.95 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f}'.format(self.match_stats['RQ @ 0.5:0.95']))
        else:
            print('Panoptic Quality @ IoU=0.5 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f} ± {:.3f}'.format(self.match_stats['PQ @ 0.5'], self.match_stats['pq_sem'][0]))
            print('Segmentation Quality @ IoU=0.5 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f} ± {:.3f}'.format(self.match_stats['SQ @ 0.5'], self.match_stats['sq_sem'][0]))
            print('Recognition Quality @ IoU=0.5 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f} ± {:.3f}'.format(self.match_stats['RQ @ 0.5'], self.match_stats['rq_sem'][0]))

            print('Panoptic Quality @ IoU=0.75 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f} ± {:.3f}'.format(self.match_stats['PQ @ 0.75'], self.match_stats['pq_sem'][5]))
            print('Segmentation Quality @ IoU=0.75 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f} ± {:.3f}'.format(self.match_stats['SQ @ 0.75'], self.match_stats['sq_sem'][5]))
            print('Recognition Quality @ IoU=0.75 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f} ± {:.3f}'.format(self.match_stats['RQ @ 0.75'], self.match_stats['rq_sem'][5]))

            print('Panoptic Quality @ IoU=0.9 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f} ± {:.3f}'.format(self.match_stats['PQ @ 0.9'], self.match_stats['pq_sem'][8]))
            print('Segmentation Quality @ IoU=0.9 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f} ± {:.3f}'.format(self.match_stats['SQ @ 0.9'], self.match_stats['sq_sem'][8]))
            print('Recognition Quality @ IoU=0.9 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f} ± {:.3f}'.format(self.match_stats['RQ @ 0.9'], self.match_stats['rq_sem'][8]))

            print('Panoptic Quality @ IoU=0.5:0.95 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f} ± {:.3f}'.format(self.match_stats['PQ @ 0.5:0.95'], self.match_stats['mpq_sem']))
            print('Segmentation Quality @ IoU=0.5:0.95 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f} ± {:.3f}'.format(self.match_stats['SQ @ 0.5:0.95'], self.match_stats['msq_sem']))
            print('Recognition Quality @ IoU=0.5:0.95 for {}:'.format(self.name).ljust(87, ' ')
                  + '{:.3f} ± {:.3f}'.format(self.match_stats['RQ @ 0.5:0.95'], self.match_stats['mrq_sem']))
        print(''.ljust(100, '-'))

    def get_map_scores(self, pred, gt):
        """
        Calculates the AP and mAP values from the mAP_3Dvolume package.

        Parameters
        ----------
        pred : array-like
        gt : array-like

        Returns
        -------
        dict
            dictionary with several stats like mAP AP@0.5, AP@0.75, Precision, Recall, etc.
        """
        def _get_scores(pred_seg, gt_seg):
            sz_gt = np.array(gt_seg.shape)
            sz_pred = pred_seg.shape
            if np.abs((sz_gt - sz_pred)).max() > 0:
                print('Warning: size mismatch. gt: ', sz_gt, ', pred: ', sz_pred)
            sz = np.minimum(sz_gt, sz_pred)
            pred_seg = pred_seg[:sz[0], :sz[1]]
            threshold_crumb = 16
            chunk_size = 100
            slices = [0, gt_seg.shape[0]]

            ui, uc = unique_chunk(pred_seg, slices, chunk_size=chunk_size)
            # ui is equivalent to unique IDs of detected cells
            # uc is equivalent to size/num_voxels of detected cells
            uc = uc[ui > 0]
            ui = ui[ui > 0]
            pred_score = np.ones([len(ui), 2], int)
            pred_score[:, 0] = ui
            pred_score[:, 1] = uc

            thres = np.fromstring('5e3, 1.5e4', sep=",")
            area_range = np.zeros((len(thres) + 2, 2), int)
            area_range[0, 1] = 1e10
            area_range[-1, 1] = 1e10
            area_range[2:, 0] = thres
            area_range[1:-1, 1] = thres

            return pred_score, area_range, slices, chunk_size, threshold_crumb

        def _get_stats(pred_seg, gt_seg):
            pred_score, area_range, slices, chunk_size, threshold_crumb = _get_scores(pred_seg, gt_seg)

            if len(pred_seg.shape) == 3:
                result_p, result_fn, pred_score_sorted = seg_iou3d_sorted(pred_seg, gt_seg, pred_score, slices,
                                                                          area_range, chunk_size, threshold_crumb)
            elif len(pred_seg.shape) == 2:
                result_p, result_fn, pred_score_sorted = seg_iou2d_sorted(pred_seg, gt_seg, pred_score, area_range)
            else:
                raise TypeError('Mask inputs must be of 2- or 3-dimensional shape.')

            v3deval = VOL3Deval(result_p, result_fn, pred_score_sorted, output_name='map_output')
            stats = v3deval.get_stats()
            return stats
        self.not_implemented()  # dummy function that prevents interpreting the method as static -> uses self keyword.
        return _get_stats(pred, gt)

    def not_implemented(self):
        pass


class Benchmarker:
    def __init__(self, model_list):
        """
        Class that compactly visualizes the scores of different evaluated prediction model outputs.

        Parameters
        ----------
        model_list : list of Eval-objects
            expects a list of Eval objects (CellAnalysis.eval.Eval) where it fetches the relevant data to display.
        """
        assert all(isinstance(model, Eval) for model in model_list), 'list must be of class CellAnalysis.eval.Eval'
        self.models = model_list
        self.markers = ['^', 'o', '>', '<', 'x', 'v', '*', 'D', 's', 'd', '1', '2', '3', '4', 'h', 'H']
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'limegreen', 'orange', 'darkblue', 'goldenrod',
                       'peru', 'tomato', 'deeppink', 'skyblue', 'black']

    def plot_ap_curves(self,
                       ax=None,
                       set_ax_layout=True,
                       error_band=False,
                       fontsize=12,
                       ticksize=16,
                       legend_fontsize=16):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 7))

        for model, color, marker in zip(self.models, self.colors, self.markers):
            model.plot_ap(ax, color=color, label=model.name, marker=marker, error_band=error_band)

        if set_ax_layout:
            ax.tick_params(axis='both', which='major', labelsize=ticksize)
            ax.set_ylabel('Average Precision (AP)', fontsize=fontsize)
            ax.set_xlabel('IoU Threshold', fontsize=fontsize)
            ax.legend(fontsize=legend_fontsize)
            ax.set_ylim(0.0, 1.0)

    def show_adc_scores(self):
        for model in self.models:
            model.print_adc_scores()

    def plot_error_bars(self, ax=None, metric='adc', title=None, y_label=None, set_legend=True, fontsize=12,
                        legend_fontsize=8, legend_loc='best', ticksize=16):
        """
        Make Boxplot with Errorbars for multiple evaluated models of base class Eval. ADC, ADPC and ADGC metrics
        are supported for error visualization purposes.
        Parameters
        ----------
        ax : matplotlib.pyplot.ax
        metric : str
        title : str
        y_label : str
        set_legend : boolean
        fontsize : int
        legend_fontsize : int

        Returns
        -------

        matplotlib.pyplot.ax
        """
        if ax is None:
            fig, ax = plt.subplots()
        if title is not None:
            ax.set_title(title, fontsize=fontsize)
        if y_label is not None:
            ax.set_ylabel(y_label)

        model_names = []
        data = []
        for model in self.models:
            model_names.append(model.name)
            adc_list = [metrics[0] for metrics in list(model.adc_full.items())]
            map_list = [metrics[0] for metrics in list(model.map.items())]
            metric_list = adc_list + map_list
            assert metric in metric_list, 'Wrong value key. Choose value from {}'.format(metric_list)
            if metric in adc_list:
                data.append(list(model.adc_full[metric].values()))
            else:
                data.append(list(model.map_full[metric].values()))
        sns.set(context='notebook')
        custom_lines = []
        for i in range(len(self.models)):
            custom_lines.append(Line2D([0], [0], color=self.colors[i], lw=4))
        if len(data[0]) < 2:
            num_models = len(data)
            x = np.arange(1, num_models+1)
            y = np.squeeze(np.array([data[i] for i in range(len(data))]))
            df = pd.DataFrame({'x': x, 'y': y, 'model_name': model_names})
            sns.scatterplot(ax=ax, x='x', y='y', data=df, hue='model_name', palette=self.colors[:num_models],
                            s=300, legend=False)
        else:
            sns.boxplot(ax=ax, data=data, width=.18, palette=self.colors)
        if set_legend:
            ax.legend(custom_lines, model_names, fontsize=legend_fontsize, loc=legend_loc)
        ax.set_xticks([])
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.tick_params(axis='both', which='major', labelsize=ticksize)
        return ax

    def print_results(self, metric='adc'):
        for model in self.models:
            if metric == 'adc':
                model.print_adc_scores()
            elif metric == 'ap':
                model.print_ap_scores()
            elif metric == 'pq':
                model.print_pq_scores()
            else:
                break

    def summarize(self,
                  title=None,
                  save_to_file=None,
                  figsize=None,
                  error_bands=False,
                  file_type='png',
                  fontsize=15,
                  legend_fontsize=15,
                  ticksize=16,
                  curve_legend=True):

        if curve_legend:
            set_error_bar_legend = False
        else:
            set_error_bar_legend = True
        if title is None:
            title = 'Summarized Stats for Segmentation'
        if figsize is None:
            figsize = (20, 8.5)
        sns.set()
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, fontsize=24)
        font_size = fontsize
        gs = GridSpec(2, 5, figure=fig)
        ax1 = fig.add_subplot(gs[:, :2])
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[0, 3])
        ax4 = fig.add_subplot(gs[0, 4])
        ax5 = fig.add_subplot(gs[1, 2])
        ax6 = fig.add_subplot(gs[1, 3])
        ax7 = fig.add_subplot(gs[1, 4])

        self.plot_ap_curves(ax1, fontsize=fontsize, error_band=error_bands,
                            ticksize=ticksize, legend_fontsize=legend_fontsize)
        ax1.set_title('Average Precision across IoU thresholds', fontsize=fontsize)

        ax2 = self.plot_error_bars(ax2, metric='mAP @ 0.5:0.95', title='mean Average Precision (mAP) @ 0.5:0.95',
                                        fontsize=font_size, set_legend=set_error_bar_legend, legend_loc='best',
                                        legend_fontsize=legend_fontsize, ticksize=ticksize)
        ax3 = self.plot_error_bars(ax3, metric='AP @ 0.5', title='Average Precision (AP) @ 0.5',
                                        fontsize=font_size, set_legend=set_error_bar_legend, legend_loc='best',
                                        legend_fontsize=legend_fontsize, ticksize=ticksize)
        ax4 = self.plot_error_bars(ax4, metric='AP @ 0.75', title='Average Precision (AP) @ 0.75',
                                        fontsize=font_size, set_legend=set_error_bar_legend, legend_loc='best',
                                        legend_fontsize=legend_fontsize, ticksize=ticksize)
        ax2.set_ylim([0.0, 1.05])
        ax3.set_ylim([0.0, 1.05])
        ax4.set_ylim([0.0, 1.05])

        data = []
        for model in self.models:
            for adc_metric in ['adc', 'adpc', 'adgc']:
                array = np.fromiter(model.adc_full[adc_metric].values(), dtype=float).ravel()
                data.append(array)

        adc_values = np.concatenate(data)
        min = 0.0
        max = np.max(adc_values) + 0.05

        ax5 = self.plot_error_bars(ax5, metric='adc', title='ADC', fontsize=font_size, set_legend=set_error_bar_legend,
                                        legend_loc='upper left', legend_fontsize=legend_fontsize, ticksize=ticksize)
        ax6 = self.plot_error_bars(ax6, metric='adpc', title='ADPC (FN Error)', fontsize=font_size,
                                        set_legend=set_error_bar_legend, legend_loc='upper left',
                                        legend_fontsize=legend_fontsize, ticksize=ticksize)
        ax7 = self.plot_error_bars(ax7, metric='adgc', title='ADGC (FP Error)', fontsize=font_size,
                                        set_legend=set_error_bar_legend, legend_loc='upper left',
                                        legend_fontsize=legend_fontsize, ticksize=ticksize)
        ax5.set_ylim([min, max])
        ax6.set_ylim([min, max])
        ax7.set_ylim([min, max])

        ax5.set_ylabel('Euclidean Distance [µm]', fontsize=font_size)
        mpl.rcParams['xtick.labelsize'] = ticksize
        mpl.rcParams['ytick.labelsize'] = ticksize
        fig.tight_layout()

        if save_to_file is not None and save_to_file is not False:
            plt.savefig(save_to_file + '.' + file_type)


def load_data(path, benchmark=False):
    models = []
    for i, (root, d_names, f_names) in enumerate(os.walk(path)):
        if i == 0:
            data_name = root.rsplit('/', 1)[1]
            print('Load data of type {} ...'.format(data_name))
        elif i < 3:
            data_type = root.rsplit('/', 1)[1]
            if data_type == 'gt':
                gt = load_sorted(root + '/')
                print('Load {} Ground Truth test instance(s) ...'.format(len(gt)))

            elif data_type == 'prediction':
                if benchmark:
                    model_names = d_names
                    print('Load prediction mask instance(s) from models: {} ...'.format(model_names))
                else:
                    model_names = data_type
                    models = load_sorted(root + '/')
            else:
                if benchmark:
                    raise NameError(
                        'Folder Structure seems to be not concise. ´prediction´ and/or ´gt´ folder not found '
                        'on first level of folder-hierarchy.')
                else:
                    model_names = data_type
                    models = load_sorted(root + '/')
        else:
            models.append(load_sorted(root + '/'))
    print('Dataloading finished succesfully!')
    return gt, models, model_names, data_name


def benchmark(path, resolution=(0.51, 0.51, 0.51)):
    gt, models, model_names, data_name = load_data(path, benchmark=True)
    evaluated = []
    for pred, name in sorted(zip(models, model_names), key=lambda t: t[1]):
        evaluator = Eval(gt, pred, resolution=resolution,
                         model_name=name, name_data=data_name)
        evaluator.accumulate()
        evaluated.append(evaluator)
    return Benchmarker(evaluated)


def evaluate(path, resolution=(1, 1, 1)):
    gt, pred, model_names, data_name = load_data(path, benchmark=False)
    eval = Eval(gt, pred, resolution=resolution, model_name=model_names, name_data=data_name)
    eval.accumulate()
    return eval
