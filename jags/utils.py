import numpy as np


def summary(values, variable_name, index=None, p=97.5):
    ci = np.percentile(values, [100 - p, p])
    print('{:<20} mean = {:>5.3f}, {}% credible interval [{:>4.3f} {:>4.3f}]'.format(
        variable_name, np.mean(values), p, *ci))
