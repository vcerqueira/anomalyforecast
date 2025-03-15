import os
from pathlib import Path
from collections import defaultdict

import requests


class MIMIC3DataGetter:
    # need software @ https://archive.physionet.org/physiotools/wfdb-darwin-quick-start.shtml
    # https://archive.physionet.org/physiotools/wag/
    DATA_DIR = Path('/Users/vcerq/Developer/datasets/mimic3')

    URL = {
        'source': 'https://physionet.org/content/mimic3wdb-matched/1.0/',
        'patients': 'https://physionet.org/files/mimic3wdb/1.0/matched/RECORDS',
        'records_num': 'https://physionet.org/files/mimic3wdb/1.0/matched/RECORDS-numerics',
        'records_wf': 'https://physionet.org/files/mimic3wdb/1.0/matched/RECORDS-waveforms',
    }

    @classmethod
    def read_lines(cls, record_type: str):
        url = cls.URL[record_type]

        data = requests.get(url)

        rec_list = data.text.split('\n')
        rec_list = [x for x in rec_list if x != '']

        return rec_list

    @classmethod
    def get_numeric_records(cls, sample: bool):
        """

        :param sample: if sample, get patients from subdir p00
        """

        rec_list = MIMIC3DataGetter.read_lines('records_num')

        if sample:
            rec_list = [x for x in rec_list if x.startswith('p00')]

        return rec_list

    @staticmethod
    def get_data_rdsamp(record: str, filepath: Path):
        expr1 = 'rdsamp -r mimic3wdb/matched/'
        expr2 = ' -c -H -f 0 -v -pd > '

        url = f'{expr1}{record}{expr2}{filepath}'

        os.system(url)

    @classmethod
    def download_records(cls):
        rec_list = cls.get_numeric_records(sample=True)

        episode_counter = defaultdict(lambda: 0)
        for i, rec in enumerate(rec_list):
            print(i, rec)
            patient = rec.split('/')[1]

            filepath = cls.DATA_DIR / f'{patient}_{episode_counter[patient]}.csv'

            if os.path.exists(filepath):
                continue

            cls.get_data_rdsamp(rec, filepath)

            episode_counter[patient] += 1


MIMIC3DataGetter.download_records()
