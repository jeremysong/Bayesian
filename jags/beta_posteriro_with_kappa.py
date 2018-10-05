import numpy as np
import os
import pandas as pd
import pyjags
import matplotlib.pyplot as plt

from jags.utils import summary

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/'

data_csv = pd.read_csv(ROOT_DIR + 'z6N8z2N7.csv')
y = data_csv['y'].tolist()
# Categorize the 's' column: Reginald -> 0, Tony -> 1
s_category = data_csv['s'].astype('category').cat
s = [v + 1 for v in s_category.codes.tolist()]

n_total = len(y)
n_subject = len(s_category.categories)

# Build model

code = """
model {
  for ( i in 1:Ntotal ) {
    y[i] ~ dbern( theta[s[i]] )
  }
  for ( s in 1:Nsubj ) {
    theta[s] ~ dbeta( omega*(kappa-2)+1 , (1-omega)*(kappa-2)+1 )
  }
  omega ~ dbeta( 1 , 1 )
  kappa <- kappaMinusTwo + 2
  kappaMinusTwo ~ dgamma( 6.25 , 0.125 ) 
}
"""

num_chains = 3
model = pyjags.Model(code, data=dict(y=y, s=s, Ntotal=n_total, Nsubj=n_subject), chains=num_chains)
model.update(500)
samples = model.sample(20000, vars=['theta'])

summary(samples['theta'][0], 'theta[1]')
summary(samples['theta'][1], 'theta[2]')

difference = samples['theta'][0] - samples['theta'][1]
print(np.shape(difference))
summary(difference, 'theta[1] - theta[2]')

plt.hist(samples['theta'][0].flatten(), 50, density=True, facecolor='g', rwidth=0.5)
plt.hist(samples['theta'][1].flatten(), 50, density=True, facecolor='b', rwidth=0.5)
plt.hist(difference.flatten(), 50, density=True, facecolor='r', rwidth=0.5)
plt.show()
