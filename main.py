import os

from dtw.data import Data
from dtw.dtw import DTW

PATH = os.path.dirname(__file__)


if __name__ == '__main__':
    window_size = 4

    user_count = 11
    genuine_sample_count = 42
    forged_sample_count = 36
    forger_count = 4

    genuine_file_path_template = os.path.join(PATH, 'data/Genuine/{user}/{sample}_{user}.HWR')
    forged_file_path_template = os.path.join(PATH, 'data/Forged/{user}/{sample}_{forger}_{user}.HWR')

    data = Data(
        user_count=user_count,
        genuine_sample_count=genuine_sample_count,
        forged_sample_count=forged_sample_count,
        forger_count=forger_count,
        genuine_file_path_template=genuine_file_path_template,
        forged_file_path_template=forged_file_path_template
    )

    genuine_threshold = list()
    with open(PATH + 'genuine.txt', 'a') as f:
        for user in range(user_count):
            for x, y in data.get_combinations(user, forged=False):
                dtw = DTW(x, y, window_size, DTW.euclidean)
                f.write(dtw.calculate())

    forged_threshold = list()
    with open(PATH + 'forged.txt', 'a') as f:
        for user in range(user_count):
            for x, y in data.get_combinations(user, forged=True):
                dtw = DTW(x, y, window_size, DTW.euclidean)
                f.write(dtw.calculate())
