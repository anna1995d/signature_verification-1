import itertools

import numpy as np


class Data(object):
    @staticmethod
    def extract_sample(path):
        with open(path, 'r') as f:
            d = np.reshape(f.read().split(), newshape=(-1, 3))

        rd = np.take(d, indices=[0, 1], axis=1).astype(np.int32)

        return np.subtract(rd[1:], rd[:-1])

    @staticmethod
    def extract_genuine(usr, smp_cnt, path_temp):
        res = list()

        for i in range(smp_cnt):
            path = path_temp.format(user=usr, sample='{:02d}'.format(i + 1))
            res.append(Data.extract_sample(path=path))

        return res

    @staticmethod
    def extract_forged(usr, smp_cnt, frg_cnt, path_temp):
        res = list()

        for i in range(1, smp_cnt + 1):
            path = path_temp.format(
                user=usr,
                sample='{:02d}'.format((i + frg_cnt - 1) // frg_cnt),
                forger='{:03d}'.format(i % frg_cnt)
            )
            res.append(Data.extract_sample(path=path))

        return res

    def __init__(self, usr_cnt, gen_smp_cnt, frg_smp_cnt, frg_cnt, gen_path_temp, frg_path_temp):

        gen = [Data.extract_genuine('{:03d}'.format(i), gen_smp_cnt, gen_path_temp) for i in range(usr_cnt)]
        frg = [Data.extract_forged('{:03d}'.format(i), frg_smp_cnt, frg_cnt, frg_path_temp) for i in range(usr_cnt)]

        self.genuine = [*itertools.chain.from_iterable(gen)]
        self.forge = [*itertools.chain.from_iterable(frg)]
        self.max_len = max(map(lambda x: x.shape[0], itertools.chain(self.genuine, self.forge)))

    # TODO: Fix it ...
    def get_combinations(self, user, forged=False):
        if not forged:
            return itertools.combinations(self.genuine[user], 2)
        return itertools.product(self.genuine[user], self.forge[user])
