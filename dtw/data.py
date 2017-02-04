import itertools
import numpy as np


class Data(object):

    @staticmethod
    def extract_sample(file_path):
        with open(file_path) as f:
            data = f.read().split()

        return np.array(
            [np.take(chunk, [0, 1]).astype(np.int32) for chunk in np.split(np.array(data), len(data) / 3)]
        )

    @staticmethod
    def extract_genuine(user, sample_count, file_path_template):
        result = list()

        for i in range(1, sample_count + 1):
            file_path = file_path_template.format(user=user, sample="%.2d" % i)
            result.append(Data.extract_sample(file_path=file_path))

        return np.array(result)

    @staticmethod
    def extract_forged(user, sample_count, forger_count, file_path_template):
        result = list()

        for i in range(1, sample_count + 1):
            file_path = file_path_template.format(
                user=user,
                sample="%.2d" % ((i + forger_count - 1) // forger_count),
                forger="%.3d" % (i % forger_count)
            )
            result.append(Data.extract_sample(file_path=file_path))

        return np.array(result)

    def __init__(self, user_count, genuine_sample_count, forged_sample_count, forger_count, genuine_file_path_template,
                 forged_file_path_template):
        self.genuine_extracted_data = []
        self.forged_extracted_data = []

        for i in range(user_count):
            self.genuine_extracted_data.append(
                Data.extract_genuine("%.3d" % i, genuine_sample_count, genuine_file_path_template)
            )

        for i in range(user_count):
            self.forged_extracted_data.append(
                Data.extract_forged("%.3d" % i, forged_sample_count, forger_count, forged_file_path_template)
            )

    def get_combinations(self, user, forged=False):
        if not forged:
            return itertools.combinations(self.genuine_extracted_data[user], 2)
        return itertools.product(self.genuine_extracted_data[user], self.forged_extracted_data[user])
