import warnings
from typing import Union, List

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import plotnine as p9

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score,
                             roc_curve,
                             log_loss,
                             classification_report)

from anomalyforecast.evaluation.episode_metrics import EpisodeEvaluation
from anomalyforecast.evaluation.episode_builder import EpisodeBuilder

StrOrList = Union[str, List[str]]

warnings.filterwarnings('ignore')


class AnomalyForecastMetrics:
    """
    Evaluating an activity monitoring model across multiple episodes
    """

    def __init__(self,
                 alarm_duration: int,
                 min_n_instances: int,
                 max_event_duration: int,
                 id_col: StrOrList = ['unique_id', 'episode'],
                 actual_col: str = 'event',
                 preds_col: str = 'pred',
                 probs_col: str = 'prob'):
        """
        :param alarm_duration: (int) Number of periods an alarm is active
        """

        self.id_col = id_col
        self.actual_col = actual_col
        self.preds_col = preds_col
        self.probs_col = probs_col

        self.eval_ep = EpisodeEvaluation(alarm_duration=alarm_duration)
        self.episodes = EpisodeBuilder(min_n_instances=min_n_instances,
                                       max_event_duration=max_event_duration)

        self.episode_metrics = None
        self.roc_df = None
        self.standard_roc_df = None

        self.metrics = {}

    def calc_episode_metrics(self, df: pd.DataFrame):
        """
        Evaluating a classifier in a dictionary of episodes

        :return: self, with the following metrics computed for each episode:
        - anticipation time;
        - discounted false positives;
        - false alarm rate
        """

        ep_results_d = {}
        for ep_id, ep in df.groupby(self.id_col):
            data = {'y_hat': ep[self.preds_col].values,
                    'y_prob': ep[self.probs_col].values,
                    'y': ep[self.actual_col].values}

            ep_metrics = self.eval_ep.calc_metrics(**data)

            ep_results_d[ep_id] = ep_metrics

        self.episode_metrics = pd.DataFrame(ep_results_d).T

        return self.episode_metrics

    def calc_metrics(self, df: pd.DataFrame):
        eps = self.episodes.episode_split(df)

        ep_metrics = self.calc_episode_metrics(eps)

        recall_ratio, recall_score = self.event_recall()
        rp_binary, rp_soft = self.reduced_precision()

        avg_anticipation = ep_metrics['anticipation_time'].mean()
        false_alarm_ratio = ep_metrics['false_alarms_ratio'].mean()
        false_alarm_time = ep_metrics['false_alarms_time'].mean()
        avg_stb = ep_metrics['stability'].mean()

        metrics = {'recall_events': recall_ratio,  # ratio of events predicted
                   'recall_score': recall_score,
                   'red_precision': rp_binary,
                   'red_prec_score': rp_soft,
                   'anticipation': avg_anticipation,  # same as recall_score?
                   'false_alarm_ratio': false_alarm_ratio,
                   'false_alarm_time': false_alarm_time,
                   'stability': avg_stb}

        return metrics

    def roc_analysis(self, df: pd.DataFrame):

        roc_data = []

        thr_space = np.arange(0.0, 1.0, 0.005).tolist()

        for thr in thr_space:
            df_ = df.copy()
            df_['pred'] = (df_['prob'] > thr).astype(int)

            thr_scr = self.calc_metrics(df_)

            roc_data.append(thr_scr)

        self.roc_df = pd.DataFrame(roc_data)
        self.roc_df['threshold'] = thr_space

        auc_rec = np.trapezoid(y=self.roc_df['recall_events'], x=1 - self.roc_df['false_alarm_ratio'])
        auc_recs = np.trapezoid(y=self.roc_df['recall_score'], x=1 - self.roc_df['false_alarm_ratio'])
        auc_rec_fat = np.trapezoid(y=self.roc_df['recall_events'], x=1 - self.roc_df['false_alarm_time'])
        auc_recs_fat = np.trapezoid(y=self.roc_df['recall_score'], x=1 - self.roc_df['false_alarm_time'])

        auc = {
            'auc_recall_far': auc_rec,
            'auc_anticipation_far': auc_recs,
            'auc_recall_fat': auc_rec_fat,
            'auc_anticipation_fat': auc_recs_fat,
        }

        return self.roc_df, auc

    def reduced_precision(self):
        """
        Reduced Precision

        Ratio between the (total of events predicted) and
        (the sum between the total of events predicted plus the total false alarm events (DFP))

        :return: Reduced precision metric
        """

        detections_bin = self.episode_metrics['detected_event'].sum()
        detections_soft = self.episode_metrics['anticipation_time'].sum()

        tot_dfp = self.episode_metrics['discounted_fp'].sum()

        rp_soft = detections_soft / (detections_soft + tot_dfp)
        rp_binary = detections_bin / (detections_bin + tot_dfp)

        return rp_binary, rp_soft

    def event_recall(self):
        detections = self.episode_metrics['detected_event']

        recall_ratio = detections.mean()

        tot_events = detections[~detections.isna()].__len__()

        recall_score = self.episode_metrics['anticipation_time'].sum() / tot_events

        return recall_ratio, recall_score

    def calc_pointwise_metrics(self, df: pd.DataFrame):
        y_true = df[self.actual_col]
        y_pred = df[self.preds_col]
        y_prob = df[self.probs_col]

        auc = roc_auc_score(y_true=y_true, y_score=y_prob)
        ll = log_loss(y_true=y_true, y_pred=y_prob)

        self.standard_roc_df = roc_curve(y_true=y_true, y_score=y_prob)

        cr = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)

        metrics_ = {
            'AUC': auc,
            'Logloss': ll,
            **cr['weighted avg']
        }

        return metrics_

    @staticmethod
    def plot_roc_curve(roc_df: pd.DataFrame,
                       x_col: str = 'false_alarm_ratio',
                       y_col: str = 'recall_events'):

        plot = \
            p9.ggplot(roc_df) + \
            p9.aes(x=x_col,
                   y=y_col,
                   group='Model',
                   color='Model') + \
            p9.geom_line(size=1) + \
            p9.geom_point() + \
            p9.theme_538(base_family='Palatino', base_size=12) + \
            p9.theme(plot_margin=.025,
                     panel_background=p9.element_rect(fill='white'),
                     plot_background=p9.element_rect(fill='white'),
                     legend_box_background=p9.element_rect(fill='white'),
                     strip_background=p9.element_rect(fill='white'),
                     legend_background=p9.element_rect(fill='white'),
                     axis_text_x=p9.element_text(size=9, angle=0),
                     axis_text_y=p9.element_text(size=9),
                     legend_title=p9.element_blank()) + \
            p9.labs(x='False alarm ratio',
                    y='Event recall')

        return plot
