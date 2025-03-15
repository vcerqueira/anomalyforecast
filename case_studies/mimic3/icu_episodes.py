from pathlib import Path

import numpy as np
import pandas as pd

from anomalyforecast.feature_engineering.supervision import AnomalyForecastingTask

DATA_DIR = Path(__file__).parent.parent.parent / 'assets' / 'datasets'

HypotensionEpisodeParams = {
    'input_size': 60,
    'warning_size': 30,
    'target_size': 30,
}

HypotensionEpisodeParams['total_size'] = sum([*HypotensionEpisodeParams.values()])
HypotensionEpisodeParams['alarm_duration'] = HypotensionEpisodeParams['warning_size']
HypotensionEpisodeParams['max_event_duration'] = HypotensionEpisodeParams['target_size']
HypotensionEpisodeParams['min_n_instances'] = 2 * HypotensionEpisodeParams['total_size']


class MIMI3DataLoader:

    @staticmethod
    def load_data():
        data = pd.read_csv(f'{DATA_DIR}/mimic3.csv', parse_dates=['ds'])

        return data

    @staticmethod
    def tr_ts_split(df: pd.DataFrame, test_size: float):
        """
        sampling 30% of patients to test
        :param test_size:
        :param df:
        :return:
        """
        unq_patients = df['patient'].unique()
        test_patients = np.random.choice(unq_patients,
                                         size=int(len(unq_patients) * test_size),
                                         replace=False).tolist()

        train = df.query('patient !=@test_patients')
        test = df.query('patient ==@test_patients')

        train = train.drop(columns='patient').rename(columns={'episode': 'unique_id', 'bpm': 'y'})
        test = test.drop(columns='patient').rename(columns={'episode': 'unique_id', 'bpm': 'y'})

        return train, test


class MIMI3HypotensionSupervision(AnomalyForecastingTask):
    RAMP_THRESHOLD = 0.15

    LGB_PARAMS = {
        'boosting_type': 'gbdt',
        'lambda_l1': 1,
        'lambda_l2': 100,
        'learning_rate': 0.2,
        'linear_tree': True,
        'max_depth': -1,
        'min_child_samples': 15,
        'n_jobs': 1,
        'num_boost_round': 50,
        'num_leaves': 30
    }

    @classmethod
    def event_definition(cls, target_series: pd.Series) -> int:
        below_thr = target_series < 65

        event_occurs = int(below_thr.mean() > 0.9)

        return event_occurs
