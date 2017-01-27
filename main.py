import os

from dtw.data import Data
from dtw.dtw import DTW

PATH = os.path.dirname(__file__)


if __name__ == '__main__':
    window_size = 4
    user_count = 40
    sample_count = 40
    file_path_template = os.path.join(PATH, 'data/SVC2004/Task1/U{user}S{sample}.txt')

    data = Data(user_count, sample_count, file_path_template)

    for user in range(1, user_count):
        for x, y in data.get_combinations(user):
            dtw = DTW(x, y, window_size, DTW.euclidean)
            print(dtw.calculate())
