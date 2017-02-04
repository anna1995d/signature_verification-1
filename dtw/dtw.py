import numpy as np


class DTW(object):

    @staticmethod
    def euclidean(first_vec, second_vec):
        return np.linalg.norm(first_vec - second_vec)

    def __init__(self, first_vec, second_vec, window_size, dist_func=None):
        self.first_vec = first_vec
        self.second_vec = second_vec
        self.window_size = max(window_size, abs(first_vec.shape[0] - second_vec.shape[0]) + 1)
        self.dist_func = dist_func or DTW.euclidean

    def calculate(self):
        ans = np.full(shape=(self.first_vec.shape[0], self.second_vec.shape[0]), fill_value=np.inf)
        dis = np.full(shape=(self.first_vec.shape[0], self.second_vec.shape[0]), fill_value=-1)

        ans[0][0] = dis[0][0] = self.dist_func(self.first_vec[0], self.second_vec[0])
        for i in range(1, min(self.window_size, self.first_vec.shape[0])):
            if dis[i][0] == -1:
                dis[i][0] = self.dist_func(self.first_vec[i], self.second_vec[0])
            ans[i][0] = ans[i - 1][0] + dis[i][0]

        for i in range(1, min(self.window_size, self.second_vec.shape[0])):
            if dis[0][i] == -1:
                dis[0][i] = self.dist_func(self.first_vec[0], self.second_vec[i])
            ans[0][i] = ans[0][i - 1] + dis[0][i]

        for i in range(1, self.first_vec.shape[0]):
            for j in range(max(1, i - self.window_size + 1), min(self.second_vec.shape[0], i + self.window_size)):
                if dis[i][j] == -1:
                    dis[i][j] = self.dist_func(self.first_vec[i], self.second_vec[j])
                ans[i][j] = min(ans[i - 1][j - 1], ans[i][j - 1], ans[i - 1][j]) + dis[i][j]

        return ans[self.first_vec.shape[0] - 1][self.second_vec.shape[0] - 1]
