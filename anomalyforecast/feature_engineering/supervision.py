from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class AnomalyForecastingTask(ABC):

    def __init__(self,
                 input_size: int,
                 warning_size: int,
                 target_size: int):

        self.check_input_parameters()

        self.input_size = input_size
        self.warning_size = warning_size
        self.target_size = target_size

    def create_windows(self, df: pd.DataFrame):
        """

        :param df: DF with a nixtla-like structure
        :return:
        """
        rec_uid = []
        df_uid = df.set_index(['unique_id', 'ds']).groupby('unique_id')
        for uid, df_ in df_uid:
            rec_df_uid = self._create_windows(df_['y'])
            rec_uid.append(rec_df_uid)

        rec_df = pd.concat(rec_uid)

        return rec_df

    def _create_windows(self, series: pd.Series) -> pd.DataFrame:
        if series.name is None:
            name = 'Series'
        else:
            name = series.name

        input_shifts = list(range(self.input_size - 1, -1, -1))
        output_shifts = list(range(-1, -(self.warning_size + self.target_size + 1), -1))
        shifts_levels = input_shifts + output_shifts

        shifted_series = [series.shift(i) for i in shifts_levels]

        rec_df = pd.concat(shifted_series, axis=1).dropna()
        column_names = []
        for i in shifts_levels:
            if i >= 0:
                col = f'{name}(L-{i})'
            else:
                if i < -self.warning_size:
                    col = f'{name}(T-{np.abs(i) - self.warning_size})'
                else:
                    col = f'{name}(W-{np.abs(i)})'

            column_names.append(col)

        rec_df.columns = column_names
        rec_df.index.name = f'{rec_df.index.name} (at L-0)'

        return rec_df

    @classmethod
    def get_input_output(cls, reconstructed_df: pd.DataFrame):
        is_lags = reconstructed_df.columns.str.contains('L-')
        is_target = reconstructed_df.columns.str.contains('T-')

        X = reconstructed_df.iloc[:, is_lags]
        Y = reconstructed_df.iloc[:, is_target]

        y = Y.apply(lambda x: cls.event_definition(x), axis=1)

        return X, y

    def check_input_parameters(self):
        pass

    @classmethod
    @abstractmethod
    def event_definition(cls, target_series: pd.Series) -> int:
        """
        Checking if an event of interest occurs in the target sub-sequence.

        :param target_series: (pd.Series) A set of contiguous obs.

        :return: boolean that represents whether there's an impending hypoxia case
        """

        pass

