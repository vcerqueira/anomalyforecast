from pathlib import Path

import numpy as np
import pandas as pd

from anomalyforecast.feature_engineering.supervision import AnomalyForecastingTask

DATA_DIR = Path(__file__).parent.parent / 'assets' / 'datasets'

WindPowerRampsParams = {
    'input_size': 24,
    'warning_size': 3,
    'target_size': 6,
}

WindPowerRampsParams['total_size'] = sum([*WindPowerRampsParams.values()])
WindPowerRampsParams['alarm_duration'] = WindPowerRampsParams['warning_size']
WindPowerRampsParams['max_event_duration'] = WindPowerRampsParams['target_size']
WindPowerRampsParams['min_n_instances'] = 2 * WindPowerRampsParams['total_size']


class WindPowerRampsDataLoader:

    @staticmethod
    def load_data():
        data = pd.read_csv(f'{DATA_DIR}/wind_power.csv', parse_dates=['Datetime'])
        data['wind_power'][data['wind_power'] > data['capacity']] = np.nan

        data['norm_wp'] = data['wind_power'] / data['capacity']

        data = data.groupby('region').resample('h', on='Datetime').mean()
        data = data[['norm_wp']].reset_index()
        data.columns = ['unique_id', 'ds', 'y']

        data = data.sort_values(['unique_id', 'ds'])

        return data

    @staticmethod
    def temporal_tr_ts_split(df: pd.DataFrame):
        test_condition = df['ds'].dt.year > 2020
        test_idx = np.where(test_condition)
        train_idx = np.where(~test_condition)

        test = df.iloc[test_idx]
        train = df.iloc[train_idx]

        return train, test

    @staticmethod
    def plot1():
        # todo
        # https://github.com/vcerqueira/tsa4climate/blob/main/content/part_1/code/1_reading_data.py
        pass


class WindPowerRampSupervision(AnomalyForecastingTask):
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
    def event_definition(cls, target_series: pd.Series, event_type='upward_ramp') -> int:
        # abs_diff_t = target_series.diff().abs()
        diff_t = target_series.diff()

        if event_type == 'upward_ramp':
            event_occurs = int((diff_t > cls.RAMP_THRESHOLD).any())
        elif event_type == 'downward_ramp':
            event_occurs = int((diff_t < -cls.RAMP_THRESHOLD).any())
        else:
            event_occurs = int((diff_t.abs() > cls.RAMP_THRESHOLD).any())

        return event_occurs
