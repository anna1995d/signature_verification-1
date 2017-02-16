import itertools
import numpy as np


class Data(object):

    @staticmethod
    def extract_sample(file_path, frame_size):
        with open(file_path) as f:
            data = f.read().split()

        if len(data) % int(3 * frame_size) != 0:
            data = data[:-(len(data) % int(3 * frame_size))]

        return np.array(
            [np.take(chunk, [0, 1]).astype(np.int32) for chunk in np.split(np.array(data), len(data) // 3)]
        )

    @staticmethod
    def extract_framed_sample(file_path, frame_size):
        with open(file_path) as f:
            data = f.read().split()

        if len(data) % int(3 * frame_size) != 0:
            data = data[:-(len(data) % int(3 * frame_size))]

        return np.array(
            [np.take(chunk, [0, 1]).astype(np.int32) for chunk in np.split(np.array(data), len(data) / 3)]
        ).reshape((-1, frame_size, 2))

    @staticmethod
    def extract_genuine(user, sample_count, file_path_template, frame_size):
        result = list()
        framed_result = list()

        for i in range(1, sample_count + 1):
            file_path = file_path_template.format(user=user, sample="%.2d" % i)
            result.append(Data.extract_sample(file_path=file_path, frame_size=frame_size))
            framed_result.append(Data.extract_framed_sample(file_path=file_path, frame_size=frame_size))

        return np.array(result), np.concatenate(framed_result)

    @staticmethod
    def extract_forged(user, sample_count, forger_count, file_path_template, frame_size):
        result = list()
        framed_result = list()

        for i in range(1, sample_count + 1):
            file_path = file_path_template.format(
                user=user,
                sample="%.2d" % ((i + forger_count - 1) // forger_count),
                forger="%.3d" % (i % forger_count)
            )
            result.append(Data.extract_sample(file_path=file_path, frame_size=frame_size))
            framed_result.append(Data.extract_framed_sample(file_path=file_path, frame_size=frame_size))

        return np.array(result), np.concatenate(framed_result)

    def __init__(self, user_count, genuine_sample_count, forged_sample_count, forger_count, genuine_file_path_template,
                 forged_file_path_template, frame_size):

        self.frame_size = frame_size

        self.genuine_extracted_data = list()
        self.forged_extracted_data = list()

        self.framed_genuine_extracted_data = list()
        self.framed_forged_extracted_data = list()

        for i in range(user_count):
            result, framed_result = Data.extract_genuine(
                "%.3d" % i, genuine_sample_count, genuine_file_path_template, frame_size
            )
            self.genuine_extracted_data.append(result)
            self.framed_genuine_extracted_data.append(framed_result)

            result, framed_result = Data.extract_forged(
                "%.3d" % i, forged_sample_count, forger_count, forged_file_path_template, frame_size
            )
            self.forged_extracted_data.append(result)
            self.framed_forged_extracted_data.append(framed_result)

    def get_combinations(self, user, forged=False):
        if not forged:
            return itertools.combinations(self.genuine_extracted_data[user], 2)
        return itertools.product(self.genuine_extracted_data[user], self.forged_extracted_data[user])
