import numpy as np
import pandas as pd


class EpisodeEvaluation:
    """
    EpisodeEvaluation

    Metrics for measuring performance in a given episode

    Metrics include:
    1 - Anticipation Time;
    2 - Discounted False Positives
    3 - False Alarms per unit of time
    """

    def __init__(self, alarm_duration: int):
        self.alarm_duration = alarm_duration

    def calc_metrics(self, y_hat: np.ndarray, y_prob: np.ndarray, y: np.ndarray):
        data = {'y_hat': y_hat, 'y': y}

        far, tot_fa = self.false_alarms_ratio(**data)

        ep_metrics = {
            'detected_event': self.event_was_predicted(**data),
            'anticipation_time': self.anticipation_time(**data),
            'discounted_fp': self.discounted_false_positives(**data),
            'false_alarms_ratio': far,
            'false_alarms_total': tot_fa,
            'false_alarms_time': self.time_under_fp(**data),
            'stability': self.prob_stability(y_prob),
        }

        return ep_metrics

    @staticmethod
    def prob_stability(y_hat: np.ndarray):
        # probs.pct_change().var()
        # probs.pct_change().abs().mean()
        # probs.diff().std()
        # print(y_hat)

        pc = (100 * pd.Series(y_hat)).diff().abs().var()

        # pc = pd.Series(y_hat).pct_change()
        # pc[np.isinf(pc)] = np.nan
        # pc = pc.fillna(0)

        # avg_diff = 1 - pd.Series(y_hat).diff().abs().mean()
        stb = (100 - pc) / 100

        return stb

    @classmethod
    def event_was_predicted(cls, y_hat: np.ndarray, y: np.ndarray):
        at = cls.anticipation_time(y_hat=y_hat, y=y)

        if np.isnan(at):
            er = np.nan
        else:
            if at > 0:
                er = 1
            else:
                er = 0

        return er

    @staticmethod
    def anticipation_time(y_hat: np.ndarray, y: np.ndarray):
        """
        :param y_hat: (1d arr) vector of binary predictions
        :param y: (1d arr) true binary values

        :return: Anticipation score which denotes how soon the model detects the event.
        - 1 means that the event was predicted in the earliest possible moment
        - 0 means that the event was not detected
        - nan means no event happened at all
        """

        event_happens = any(y > 0)

        if not event_happens:
            return np.nan

        event_idx = np.where(y > 0)[0]

        preds_on_event_idx = y_hat[event_idx]
        tp_alarms = preds_on_event_idx > 0

        if any(tp_alarms):
            alarm_idx = np.min(np.where(tp_alarms)[0])
            no_anticipation_periods = len(event_idx) - alarm_idx
        else:
            no_anticipation_periods = 0

        anticipation_score = no_anticipation_periods / len(event_idx)

        return anticipation_score

    def discounted_false_positives(self, y_hat: np.ndarray, y: np.ndarray):
        """
        Discounted False Positive - A metric for false positive but which takes into account
        sequential FP. After a false positive prediction (which counts as 1 discounted FP),
        the predictions are 'turned off' for a sleep_period number of periods.


        :param y_hat: (1d arr) vector of binary predictions
        :param y: (1d arr) true binary values

        :return: DFP count - the lower the better
        """
        neg_activity = y < 1

        if any(neg_activity):
            y_neg_act = y[y < 1]
            yh_neg_act = y_hat[y < 1]
        else:
            return 0

        false_alarm_count, i = 0, 0
        while i < len(y_neg_act):
            is_false_alarm = yh_neg_act[i] != y_neg_act[i]
            if is_false_alarm:
                false_alarm_count += 1
                i += self.alarm_duration
            else:
                i += 1

        return false_alarm_count

    def time_under_fp(self, y_hat: np.ndarray, y: np.ndarray):
        """
        :param y_hat: (1d arr) vector of binary predictions
        :param y: (1d arr) true binary values

        :return:
        """
        neg_activity = y < 1

        if any(neg_activity):
            y_neg_act = y[y < 1]
            yh_neg_act = y_hat[y < 1]
        else:
            return 0

        time_count, i = 0, 0
        while i < len(y_neg_act):
            is_false_alarm = yh_neg_act[i] != y_neg_act[i]
            if is_false_alarm:
                time_count += np.min((len(y_neg_act) - i, self.alarm_duration))
                i += self.alarm_duration
            else:
                i += 1

        time_count_n = time_count / len(y_neg_act)

        return time_count_n

    @staticmethod
    def false_alarms_ratio(y_hat, y):
        """
        :param y_hat: (1d arr) vector of binary predictions
        :param y: (1d arr) true binary values

        :return: False alarm ratio score
        """

        neg_activity = y < 1
        if not any(neg_activity):
            return 0

        yh_neg_act = y_hat[neg_activity]

        tot_fa = np.sum(yh_neg_act)

        false_alarms_p_unit = tot_fa / len(yh_neg_act)

        return false_alarms_p_unit, tot_fa
