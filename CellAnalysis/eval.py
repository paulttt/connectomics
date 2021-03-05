from CellAnalysis.adc_metric import average_distance_between_centroids as adc_score
from mAP_3Dvolume.vol3d_eval import VOL3Deval
from mAP_3Dvolume.vol3d_util import seg_iou2d_sorted, seg_iou3d_sorted
from scipy.stats import sem
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


class Eval:
    def __init__(self, gt, pred, resolution=(1, 1, 1), model_name='prediction model', name_data='test data'):
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
            raise TypeError('Mask inputs must be of type numpy.ndarray '
                            'or lists of numpy.ndarrays (e.g.: [gt_1, gt_2, ..., gt_n]).')

        self.gt = gt
        self.pred = pred
        self.map = {}
        self.adc = {}
        self.name = model_name
        self.data = name_data

    def plot_ap(self, ax, color='b', label='model', linestyle='-', marker='o', error_band=False):
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
        print(''.ljust(100, '-'))
        print('Compute distance metrics for {} for {}...\n'.format(self.name, self.data))
        print('Average Distance between Centroids for {}:'.format(self.name).ljust(87, ' ')
              + '{:.3f} ± {:.3f}'.format(self.adc['adc'], self.adc['adc_sem']))
        print('Average Distance between Prediction Centroids for {}:'.format(self.name).ljust(87, ' ')
              + '{:.3f} ± {:.3f}'.format(self.adc['adpc'], self.adc['adpc_sem']))
        print('Average Distance between Ground Truth Centroids for {}:'.format(self.name).ljust(87, ' ')
              + '{:.3f} ± {:.3f}'.format(self.adc['adgc'], self.adc['adgc_sem']))
        print(''.ljust(100, '-'))

    def accumulate(self):
        adc_keys = ['adc', 'adpc', 'adgc', 'adc_sem', 'adpc_sem', 'adgc_sem']
        if isinstance(self.pred, np.ndarray):
            self.map = self.get_map_scores(self.pred, self.gt)
            self.adc = dict(zip(adc_keys, adc_score(self.gt, self.pred, size=self.size)))
        elif isinstance(self.pred, list):
            map_list = []
            adc_list = []
            index_list = []
            for idx, (pred_instance, gt_instance) in enumerate(zip(self.pred, self.gt)):
                map_list.append(self.get_map_scores(pred_instance, gt_instance))
                adc_list.append(adc_score(gt_instance, pred_instance, size=self.size))
                index_list.append(idx)
            self.map = pd.DataFrame(map_list, index_list).mean().to_dict()
            self.map['ap_sem'] = sem(np.array([map_dict['Average Precision'][:, 0] for map_dict in map_list]), axis=0)
            self.adc = pd.DataFrame(adc_list, index=index_list, columns=adc_keys).mean().to_dict()
        else:
            raise TypeError('Mask inputs must be of type numpy.ndarray '
                            'or lists of numpy.ndarrays (e.g.: [gt_1, gt_2, ..., gt_n]).')

    def get_all_stats(self, as_pandas_df=False):
        if not self.adc or not self.map:
            print('Please run accumulate() method first.')
        if as_pandas_df:
            return pd.DataFrame({**self.map, **self.adc})
        else:
            return {**self.map, **self.adc}

    def get_map_scores(self, pred, gt):
        def _get_scores(pred_seg, gt_seg):
            sz_gt = np.array(gt_seg.shape)
            sz_pred = pred_seg.shape
            if np.abs((sz_gt - sz_pred)).max() > 0:
                print('Warning: size mismatch. gt: ', sz_gt, ', pred: ', sz_pred)
            sz = np.minimum(sz_gt, sz_pred)
            pred_seg = pred_seg[:sz[0], :sz[1]]

            ui, uc = np.unique(pred_seg, return_counts=True)
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

            return pred_score, area_range

        def _get_stats(pred_seg, gt_seg):
            pred_score, area_range = _get_scores(pred_seg, gt_seg)

            if len(pred_seg.shape) == 3:
                result_p, result_fn, pred_score_sorted = seg_iou3d_sorted(pred_seg, gt_seg, pred_score, area_range)
            elif len(pred_seg.shape) == 2:
                result_p, result_fn, pred_score_sorted = seg_iou2d_sorted(pred_seg, gt_seg, pred_score, area_range)
            else:
                raise TypeError('Mask inputs must be of 2- or 3-dimensional shape.')

            v3deval = VOL3Deval(result_p, result_fn, pred_score_sorted, output_name='map_output')
            stats = v3deval.get_stats()
            return stats

        return _get_stats(pred, gt)


class Benchmarker:
    def __init__(self, model_list):
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
