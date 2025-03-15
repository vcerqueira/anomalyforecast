import os
import re

import numpy as np
import pandas as pd

from case_studies.mimic3.data_downloader import MIMIC3DataGetter

COLUMN_MAP = {
    'Time and date': 'ds',
    'HR': 'hr',
    'RESP': 'resp',
    'SpO2': 'spo2',
    'NBPSys': 'bps',
    'NBP Sys': 'bps',
    # 'ABP Sys': 'bps',
    'NBPDias': 'bpd',
    'NBP Dias': 'bpd',
    # 'ABP Dias': 'bpd',
    'NBPMean': 'bpm',
    'NBP Mean': 'bpm',
    # 'ABP Mean': 'bpm',
}

TARGET_COLUMNS = ['hr', 'resp', 'spo2', 'bps', 'bpd', 'bpm']

files = os.listdir(MIMIC3DataGetter.DATA_DIR)

dataset_list = []
for i, file in enumerate(files):
    print(i)
    # file = files[0]
    try:
        df = pd.read_csv(MIMIC3DataGetter.DATA_DIR / file, skiprows=[1], na_values='-')
    except (UnicodeDecodeError, pd.errors.EmptyDataError):
        continue

    df.columns = [re.sub("'", "", x) for x in df.columns]
    try:
        df["Time and date"] = pd.to_datetime(df["Time and date"].apply(lambda x: re.sub("'\[|]'", "", x)),
                                             format='mixed')
    except pd._libs.tslibs.parsing.DateParseError:
        continue

    df = df.rename(columns=COLUMN_MAP)
    df = df.set_index('ds')

    df = df.resample('1min').median()
    try:
        df = df[TARGET_COLUMNS]
    except KeyError:
        continue

    if df.shape[0] < 60 * 6:  # 6 hours
        continue

    if (df['spo2'] > 100).any():
        df['spo2'].loc[df['spo2'] > 100] = np.nan

    if (df['spo2'] < 50).any():
        df['spo2'].loc[df['spo2'] < 50] = np.nan

    for col in ['hr', 'bps', 'bpd', 'bpm']:
        if (df[col] > 200).any():
            df[col].loc[df[col] > 200] = np.nan

        if (df[col] < 10).any():
            df[col].loc[df[col] < 10] = np.nan

    if (df.isna().mean() > 0.5).any():
        continue

    patient = file.split('_')[0]
    ep = int(file.split('_')[1].split('.')[0])

    df['patient'] = patient
    df['episode'] = ep

    df = df.reset_index()

    print('in')

    dataset_list.append(df)

ds = pd.concat(dataset_list)
ds = ds.sort_values(['patient', 'episode', 'ds']).reset_index(drop=True)

ds.to_csv('assets/datasets/mimic3.csv', index=False)
