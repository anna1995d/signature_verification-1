import numpy as np


def extract_sample(file_path):
    with open(file_path) as f:
        data = f.read().split()

    return [np.take(chunk, [0, 1]).astype(np.int32) for chunk in np.split(np.array(data[1:]), 4)]


def extract_user(user, sample_count, file_path_template):
    return np.array(
        [extract_sample(file_path_template.format(user=user, sample=i)) for i in range(1, sample_count + 1)]
    )
