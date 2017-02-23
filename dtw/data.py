import itertools
import numpy as np

from sklearn.utils import shuffle


class Data(object):

    @staticmethod
    def extract_sample(file_path):
        with open(file_path) as f:
            data = f.read().split()

        return np.array(
            [np.take(chunk, [0, 1]).astype(np.int32) for chunk in np.split(np.array(data), len(data) // 3)]
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
                 forged_file_path_template, train_slice_per):

        self.genuine = [Data.extract_genuine("%.3d" % i, genuine_sample_count, genuine_file_path_template)
                        for i in range(user_count)]

        self.forge = [Data.extract_forged("%.3d" % i, forged_sample_count, forger_count, forged_file_path_template)
                      for i in range(user_count)]

        genuine = [_ for _ in itertools.chain.from_iterable(self.genuine)]
        forge = [_ for _ in itertools.chain.from_iterable(self.forge)]

        x = np.array(genuine + forge)
        y = np.concatenate((np.zeros((len(forge), 1)), np.ones((len(genuine), 1))))
        x, y = shuffle(x, y, random_state=100)

        self.train_x = x[:(len(x) * train_slice_per) // 100]
        self.train_y = y[:(len(y) * train_slice_per) // 100]
        self.train_max_len = len(max(self.train_x, key=lambda elem: len(elem)))

        self.dev_x = x[(len(x) * train_slice_per) // 100:]
        self.dev_y = y[(len(y) * train_slice_per) // 100:]

    def get_combinations(self, user, forged=False):
        if not forged:
            return itertools.combinations(self.genuine[user], 2)
        return itertools.product(self.genuine[user], self.forge[user])
