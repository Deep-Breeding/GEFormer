import pandas as pd
import torch
from torch.utils import data

from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    features_by_offsets = {
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]


def time_features(dates, freq):
    dates['month'] = dates.date.apply(lambda row: row.month, 1)
    dates['day'] = dates.date.apply(lambda row: row.day, 1)
    dates['weekday'] = dates.date.apply(lambda row: row.weekday(), 1)

    freq_map = {
        'd': ['month', 'day', 'weekday']
    }
    return dates[freq_map[freq.lower()]].values


class myDataset(data.Dataset):
    def __init__(self, id, phe, dictseq, env, scheme):  # id, phe, seq, env
        self.id = id
        self.phe = phe
        self.dictseq = list(dictseq.values())
        # if scheme != M1  dictenv else data_E
        self.scheme = scheme
        if self.scheme != 'M1':
            self.env = env
        else:
            self.__handle_envData__(env)

    def __getitem__(self, index):
        if self.scheme != 'M1':
            env_list = self.env.get(self.id[index])
            self.__handle_envData__(env_list)
        return self.id[index], self.phe[index], self.dictseq[index], self.data_x, self.data_stamp

    def __handle_envData__(self, env):
        # self.data_stamp = env['date']
        self.data_stamp = env.iloc[:, 0]
        self.data_x = env.iloc[:, 1:]

        self.data_stamp = pd.to_datetime(self.data_stamp)
        self.data_stamp = pd.DataFrame(self.data_stamp)
        self.data_stamp = time_features(self.data_stamp, freq='d')
        self.data_x = torch.tensor(self.data_x.to_numpy())
        self.data_stamp = torch.tensor(self.data_stamp)

    def __len__(self):
        return len(self.id)  # ??
