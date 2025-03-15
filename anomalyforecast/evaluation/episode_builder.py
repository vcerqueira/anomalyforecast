import numpy as np
import pandas as pd


class EpisodeBuilder:
    """
    Episodes

    Splitting a time series into episodes.
    An episode is a contiguous subset of a given time series.
    It spans from the end of an event (or the initial data collection) to
    the end of the next event (or the final observation)

    """

    def __init__(self,
                 min_n_instances: int,
                 max_event_duration: int,
                 id_col: str = 'unique_id',
                 event_col: str = 'event'):
        """

        :param min_n_instances: minimum number of instance necessary to consider the episode
        :param max_event_duration: maximum number of instance before the event is missed by a model
        :param id_col: str, like 'unique_id'
        :param event_col: str, like 'event' or "y_true"
        """

        self.min_n_instances = min_n_instances
        self.max_event_duration = max_event_duration
        self.id_col = id_col
        self.event_col = event_col

    def episode_split(self, df: pd.DataFrame):

        valid_episodes = []
        for uid, uid_df in df.groupby('unique_id'):
            # rleid on episode target
            event_rleid = self.rleid(uid_df['event'])

            episode_splits = self.split_dataframe(uid_df, event_rleid)

            # if an event occurs in the first rleid, skip to the next non-event segment
            if episode_splits[0]['event'].iloc[0] > 0:
                episode_splits = episode_splits[1:]

            n_segments = len(episode_splits)
            episode_counter = 0
            for i, ep in enumerate(episode_splits):
                # start only on negative activity (non-anomalies)
                if ep['event'].iloc[0] > 0:
                    continue

                if i == n_segments - 1:
                    ep['episode'] = f'EP_{i}'
                    valid_episodes.append(ep)
                else:
                    event_predecessor = ep.copy()
                    event_data = episode_splits[i + 1].head(self.max_event_duration)

                    ep_ext = pd.concat([event_predecessor, event_data])
                    ep_ext['episode'] = f'EP_{i}'

                    if ep_ext.shape[0] < self.min_n_instances:
                        continue

                    valid_episodes.append(ep_ext)

                episode_counter += 1

        episode_df = pd.concat(valid_episodes).reset_index(drop=True)

        return episode_df

    @staticmethod
    def split_dataframe(df: pd.DataFrame, factor: np.ndarray):
        grouped_df = df.groupby(factor)
        grouped_df = list(grouped_df)

        out = [x[1] for x in grouped_df]

        return out

    @staticmethod
    def rleid(x):
        if not isinstance(x, pd.Series):
            raise ValueError('x must be a pd.Series')

        xb = (x.shift(0) != x.shift(1))
        x_rleid = xb.cumsum().astype(int)

        return x_rleid
