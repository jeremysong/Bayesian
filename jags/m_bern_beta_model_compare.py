import itertools

import numpy as np
import os
import pandas as pd
import pyjags
import matplotlib.pyplot as plt

from jags.utils import summary

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/'

z = 6  # number of heads
n = 9
y = list(itertools.repeat(1, z)) + list(itertools.repeat(0, n - z))

code = """
model {
  for (i in 1:N) {
    y[i] ~ dbern(theta)  # likelihood
  }
  theta ~ dbeta(omega[m] * (kappa - 2) + 1, (1 - omega[m]) * (kappa - 2) + 1)
  omega[1] <- .25
  omega[2] <- .75
  kappa <- 12
  m ~ dcat(mPriorProb[])
  mPriorProb[1] <- .5
  mPriorProb[2] <- .5
}
"""

model = pyjags.Model(code, data=dict(y=y, N=n), chains=3, adapt=500)
model.update(500)
samples = model.sample(3334, vars=['theta', 'm'])

samples_flatten = dict([(k, v.flatten()) for k, v in samples.items()])
print(samples_flatten)

df_samples = pd.DataFrame.from_dict(samples_flatten)

theta_m1 = df_samples[df_samples['m'] == 1]['theta'].tolist()
theta_m2 = df_samples[df_samples['m'] == 2]['theta'].tolist()

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.hist(df_samples['theta'].tolist(), 50, density=True, facecolor='g', rwidth=0.5)
# plt.hist(df_samples['m'], 5, density=True, facecolor='g', rwidth=0.5)
ax2.hist(theta_m1, 50, density=True, facecolor='r', rwidth=0.5)
ax2.hist(theta_m2, 50, density=True, facecolor='b', rwidth=0.5)
plt.show()
