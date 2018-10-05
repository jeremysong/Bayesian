import numpy as np


def gamma_shape_rate_from_mean_sd(mean, sd):
    shape = mean ** 2 / sd ** 2
    rate = mean / sd ** 2
    return shape, rate


def gamma_shape_rate_from_mode_sd(mode, sd):
    rate = (mode + np.sqrt(mode ** 2 + 4 * sd ** 2)) / (2 * sd ** 2)
    shape = 1 + mode * rate
    return shape, rate


if __name__ == '__main__':
    print(gamma_shape_rate_from_mean_sd(10, 100))
    print(gamma_shape_rate_from_mode_sd(10, 100))
