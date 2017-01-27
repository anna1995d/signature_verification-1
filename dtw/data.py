import itertools
import numpy as np


class Data(object):

    @staticmethod
    def extract_sample(file_path):
        with open(file_path) as f:
            data = f.read().split()

        return np.array(
            [np.take(chunk, [0, 1]).astype(np.int32) for chunk in np.split(np.array(data[1:]), (len(data) - 1) / 4)]
        )

    @staticmethod
    def extract_user(user, sample_count, file_path_template):
        return np.array(
            [Data.extract_sample(file_path_template.format(user=user, sample=i)) for i in range(1, sample_count + 1)]
        )

    def __init__(self, user_count, sample_count, file_path_template):
        self.extracted_data = [
            Data.extract_user(i, sample_count, file_path_template) for i in range(1, user_count)
        ]

    def get_combinations(self, user):
        return itertools.combinations(self.extracted_data[user], 2)
