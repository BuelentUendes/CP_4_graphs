#Visualization of the beta distribution
#Code adapted from:
#https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/correctness_checks.ipynb

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, betabinom
from scipy.optimize import brentq
import itertools
import seaborn as sns

ns = [20, 80, 100, 500, 1000, 10000]
alpha = 0.10

sns.set_palette('pastel')
plt.figure()
ax = plt.gca()

for i in range(len(ns)):
  n = ns[i]
  l = np.floor((n+1)*alpha)
  a = n + 1 - l
  b = l
  variance = (a*b)/((a*b)**2)*(a+b+1)
  print(f"variance for {n}: {variance}")
  x = np.linspace(0.825,0.975,1000)
  rv = beta(a, b)
  ax.plot(x, rv.pdf(x), lw=3, label=f'n={n}')
ax.vlines(1-alpha,ymin=0,ymax=150,color='#888888',linestyles='dashed',label=r'$1-\alpha$')
sns.despine(top=True,right=True)
plt.yticks([])
plt.legend()
plt.title('Distribution of coverage (infinite validation set)')
plt.tight_layout()
plt.show()