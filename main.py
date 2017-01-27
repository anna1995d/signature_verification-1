import os
import itertools

from dtw.data import extract_user
from dtw.dtw import DTW

PATH = os.path.dirname(__file__)


if __name__ == '__main__':
    file_path_template = os.path.join(PATH, 'data/SVC2004/Task1/U{user}S{sample}.txt')

    extracted_data = [extract_user(1, i or 1, file_path_template) for i in range(40, step=10)]

    for x, y in itertools.product(extracted_data[:1], extracted_data[1:]):
        dtw = DTW(x, y, 4, DTW.euclidean)
        print(dtw.calculate())
