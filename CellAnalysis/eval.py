from CellAnalysis.adc_metric import average_distance_between_centroids as adc_score
from mAP_3Dvolume.vol3d_eval_custom import VOL3Deval
from mAP_3Dvolume.vol3d_util_custom import seg_iou2d_sorted, seg_iou3d_sorted, unique_chunk
from scipy.stats import sem
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


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
            assert gt.shape == pred.shape, 'Ground Truth and Prediction masks must be of same shape.'

            if len(gt.shape) == 2 and len(resolution) != 2:
                self.size = (1, 1)

        elif isinstance(gt, list) and isinstance(pred, list):
            assert len(gt) == len(pred), 'Mismatch in number of GT and Pred instances.'

            for pred_instance, gt_instance in zip(pred, gt):
                assert pred_instance.shape == gt_instance.shape, 'Each instance pair must be of same shape.'
        else:
            raise TypeError('Mask inputs must be of type numpy array '
                            'or lists of numpy arrays (e.g.: [gt_1, gt_2, ..., gt_n]).')

        self.gt = gt
        self.pred = pred
        self.map = {}
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
        elif isinstance(self.pred, list):
            map_list = []
            adc_list = []
            index_list = []
            # iterating over all images in the provided list
            for idx, (pred_instance, gt_instance) in enumerate(zip(self.pred, self.gt)):
                map_list.append(self.get_map_scores(pred_instance, gt_instance))
                adc_list.append(adc_score(gt_instance, pred_instance, size=self.size))
                index_list.append(idx)
            # average AP scores for all test instances
            self.map = pd.DataFrame(map_list, index_list).mean().to_dict()
            # calculate the Standard Error of the Mean for the different AP scores across different test instances
            self.map['ap_sem'] = sem(np.array([map_dict['Average Precision'][:, 0] for map_dict in map_list]), axis=0)
            self.adc = pd.DataFrame(adc_list, index=index_list, columns=adc_keys).mean().to_dict()
            self.adc_full = pd.DataFrame(adc_list, index=index_list, columns=adc_keys).to_dict()
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
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'black', 'orange', 'darkblue', 'goldenrod',
                       'peru', 'tomato', 'deeppink', 'skyblue', 'limegreen']

    def plot_ap_curves(self, ax=None, set_ax_layout=True, error_band=False):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 7))

        for model, color, marker in zip(self.models, self.colors, self.markers):
            model.plot_ap(ax, color=color, label=model.name, marker=marker, error_band=error_band)

        if set_ax_layout:
            mpl.rcParams['xtick.labelsize'] = 16
            mpl.rcParams['ytick.labelsize'] = 16
            ax.set_ylabel('Average Precision (AP)', fontsize=15)
            ax.set_xlabel('IoU Threshold', fontsize=15)
            ax.legend(fontsize=12)
            ax.set_ylim(0.0, 1.0)

    def show_adc_scores(self):
        for model in self.models:
            model.print_adc_scores()

    def plot_error_bars(self, ax=None, metric='adc', title=None, y_label=None):
        if ax is None:
            fig, ax = plt.subplots()
        if title is not None:
            ax.set_title(title, fontsize=14)
        if y_label is not None:
            ax.set_ylabel(y_label)

        model_names = []
        data = []
        for model in self.models:
            model_names.append(model.name)
            metric_list = [metrics[0] for metrics in list(model.adc_full.items())]
            assert metric in metric_list, 'Wrong value key. Choose value from {}'.format(metric_list)
            data.append(list(model.adc_full[metric].values()))

        sns.set(context='notebook', style='whitegrid')
        custom_lines = []
        for i in range(len(self.models)):
            custom_lines.append(Line2D([0], [0], color=self.colors[i], lw=4))

        sns.boxplot(ax=ax, data=data, width=.18, palette=self.colors)
        ax.legend(custom_lines, model_names)
        ax.set_xticks([])

        return ax
