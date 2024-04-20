# Visualization of the beta distribution
# Code adapted from:
# https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/correctness_checks.ipynb

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import beta

from helper_path import FIGURES_PATH

ns = [20, 80, 100, 500, 1000, 10000]
alpha = 0.10

sns.set_palette("pastel")
plt.figure()
ax = plt.gca()

for i in range(len(ns)):
    n = ns[i]
    l = np.floor((n + 1) * alpha)
    a = n + 1 - l
    b = l
    variance = (a * b) / ((a * b) ** 2) * (a + b + 1)
    print(f"variance for {n}: {np.round(variance, 4)}")
    x = np.linspace(0.825, 0.975, 1000)
    rv = beta(a, b)
    ax.plot(x, rv.pdf(x), lw=3, label=f"n={n}")

ax.vlines(
    1 - alpha,
    ymin=0,
    ymax=150,
    color="#888888",
    linestyles="dashed",
    label=r"$1-\alpha$",
)
sns.despine(top=True, right=True)
plt.yticks([])
plt.legend()
plt.title(
    f"Distribution of coverage (infinite validation set): {(1-alpha)}, alpha: {alpha}"
)
plt.tight_layout()
plt.savefig(
    os.path.join(FIGURES_PATH, f"Beta_distribution_alpha_{alpha}.png"),
    dpi=400,
    format="png",
)
plt.show()
plt.close()
