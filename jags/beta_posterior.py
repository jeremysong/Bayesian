import numpy as np
import os
import pandas as pd
import pyjags

from jags.utils import summary

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/'

np.random.seed(0)

data_csv = pd.read_csv(ROOT_DIR + "z15N50.csv")
y = data_csv['y'].tolist()
n = len(y)

code = """
model {
  for (i in 1:N) {
    y[i] ~ dbern(theta)  # likelihood
  }
  theta ~ dbeta(1, 1)    # prior
}
"""


def generate_init_theta():
    resampled_y = np.random.choice(y, np.random.randint(1, high=n))
    theta_init = sum(resampled_y) / len(resampled_y)
    # keep away from 0, 1
    theta_init = 0.001 + 0.998 * theta_init
    return theta_init


num_chains = 3
init_thetas = [dict(theta=generate_init_theta()) for _ in range(num_chains)]

# Initialize models
model = pyjags.Model(code, data=dict(y=y, N=n), init=init_thetas, chains=num_chains, adapt=500)
model.update(500)
samples = model.sample(3334, vars=['theta'])

print(samples['theta'])
print(np.shape(samples['theta']))
summary(samples['theta'], 'theta')
