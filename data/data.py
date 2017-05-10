import itertools

import numpy as np


class Data(object):
    @staticmethod
    def calculate_derivatives(d, smpl):
        if smpl:
            return np.concatenate(((d[1] - d[0]).reshape((1, -1)), d[1:] - d[:-1]))
        else:
            return np.concatenate((
                (d[1] - d[0]).reshape((1, -1)) + 2 * (d[2] - d[0]).reshape((1, -1)),
                (d[2] - d[0]).reshape((1, -1)) + 2 * (d[3] - d[0]).reshape((1, -1)),
                (d[3:-1] - d[1:-3]) + 2 * (d[3:-1] - d[1:-3]),
                (d[-1] - d[-3]).reshape((1, -1)) + 2 * (d[-1] - d[-4]).reshape((1, -1)),
                (d[-1] - d[-2]).reshape((1, -1)) + 2 * (d[-1] - d[-3]).reshape((1, -1)),
            )) / 10

    @staticmethod
    def extract(d):
        drv = Data.calculate_derivatives(d, smpl=True)
        t_n = np.arctan(drv[:, 1], drv[:, 0]).reshape((-1, 1))
        t_n[np.argwhere(np.isnan(t_n))] = 0
        v_n = np.sqrt(drv[:, 0] ** 2 + drv[:, 1] ** 2).reshape((-1, 1))
        dt_n = Data.calculate_derivatives(t_n, smpl=True)
        r_n = np.log(np.abs(v_n / (dt_n + np.finfo(np.float64).eps)) + np.finfo(np.float64).eps)

        return np.concatenate((
            d,
            Data.calculate_derivatives(d, smpl=False),
            t_n,
            v_n,
            r_n,
            drv,
        ), axis=1)

    @staticmethod
    def normalize(d, nrm):
        d -= np.mean(d, axis=0) if 'm' in nrm else 0
        d -= np.std(d, axis=0, ddof=1) if 's' in nrm else 0
        d -= d[0]
        return d

    @staticmethod
    def extract_features(d, nrm):
        return Data.extract(Data.normalize(d, nrm))

    @staticmethod
    def extract_sample(smp_stp, rl_win_sz, rl_win_stp, ftr_cnt, nrm, path):
        with open(path, 'r') as f:
            rd = np.reshape(f.read().split()[1:], newshape=(-1, ftr_cnt))[::smp_stp, :2].astype(np.float64)

        d = np.concatenate(
            [np.roll(rd, -ln, axis=0) for ln in range(rl_win_sz)], axis=1
        )[:(1 - rl_win_sz) or None:rl_win_stp]

        return Data.extract_features(d, nrm)

    @staticmethod
    def extract_genuine(smp_stp, rl_win_sz, rl_win_stp, nrm, usr, smp_cnt, ftr_cnt, path_temp):
        res = list()

        for smp in range(smp_cnt):
            path = path_temp.format(user=usr, sample=smp + 1)
            res.append(Data.extract_sample(
                smp_stp=smp_stp, rl_win_sz=rl_win_sz, rl_win_stp=rl_win_stp, ftr_cnt=ftr_cnt, nrm=nrm, path=path
            ))
        np.random.shuffle(res)

        return res

    @staticmethod
    def extract_forged(smp_stp, rl_win_sz, rl_win_stp, nrm, usr, smp_cnt, ftr_cnt, path_temp):
        res = list()

        for smp in range(smp_cnt):
            path = path_temp.format(user=usr, sample=smp + 21)
            res.append(Data.extract_sample(
                smp_stp=smp_stp, rl_win_sz=rl_win_sz, rl_win_stp=rl_win_stp, ftr_cnt=ftr_cnt, nrm=nrm, path=path
            ))
        np.random.shuffle(res)

        return res

    def __init__(self, smp_stp, rl_win_sz, rl_win_stp, ftr_cnt, nrm, usr_cnt, gen_smp_cnt, frg_smp_cnt, gen_path_temp,
                 frg_path_temp):

        self.gen = [Data.extract_genuine(
            smp_stp=smp_stp,
            rl_win_sz=rl_win_sz,
            rl_win_stp=rl_win_stp,
            nrm=nrm,
            usr=usr,
            smp_cnt=gen_smp_cnt,
            ftr_cnt=ftr_cnt,
            path_temp=gen_path_temp
        ) for usr in range(1, usr_cnt + 1)]

        self.frg = [Data.extract_forged(
            smp_stp=smp_stp,
            rl_win_sz=rl_win_sz,
            rl_win_stp=rl_win_stp,
            nrm=nrm,
            usr=usr,
            smp_cnt=frg_smp_cnt,
            ftr_cnt=ftr_cnt,
            path_temp=frg_path_temp
        ) for usr in range(1, usr_cnt + 1)]

    def get_genuine_combinations(self, usr_num, smp_cnt):
        return np.array(list(itertools.product(self.gen[usr_num][:smp_cnt], self.gen[usr_num][:smp_cnt]))).T
