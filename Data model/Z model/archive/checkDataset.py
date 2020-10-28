import os
import argparse
import logging as log


import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from utils import mkdir

from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc, roc_auc_score
import lightgbm as lgb


def h5ToDf(filename):
    """
    Make pandas dataframe from {filename}.h5 file.
    """
    log.info(f"Import data from: {filename}")
    with h5py.File(filename, "r") as hf :
        d = {}
        for name in list(hf.keys()):
            d[name] = np.array(hf[name][:])
        df = pd.DataFrame(data=d)
    return(df)

data = h5ToDf("/Users/sda/hep/work/Data model/Z model/output/MuoPairHdf5/160920_2/160920_2.h5")
np.unique(data["type"], return_counts=True)

data.columns

fig, ax = plt.subplots(3,1,figsize=(5,15))
ax = ax.flatten()
ax[0].hist(data["pt"][(data["type"]==0)], bins = 100, color='g', range = (4.5,120), histtype="step", label = "Background");
ax[0].hist(data["pt"][(data["type"]==1)], bins = 100, color='tab:purple', range = (4.5,120), histtype="step", weights = 10*np.ones_like(data["pt"][(data["type"]==1)]), label = "Signal, rescaled with a factor of 10");
ax[0].set(xlabel = "pt (muon pair)", ylabel = "Frequency")
ax[0].legend(prop={'size': 11.5})
for item in ([ax[0].title, ax[0].xaxis.label, ax[0].yaxis.label] +
             ax[0].get_xticklabels() + ax[0].get_yticklabels()):
    item.set_fontsize(15)

ax[1].hist(data["eta"][(data["type"]==0)], bins = 100, color='g', range = (-4,4), histtype="step", label = "Background");
ax[1].hist(data["eta"][(data["type"]==1)], bins = 100, color='tab:purple', range = (-4,4), histtype="step", weights = 10*np.ones_like(data["pt"][(data["type"]==1)]), label = "Signal, rescaled with a factor of 10");
ax[1].set(xlabel = "eta (muon pair)", ylabel = "Frequency")
ax[1].legend(prop={'size': 11.5})
for item in ([ax[1].title, ax[1].xaxis.label, ax[1].yaxis.label] +
             ax[1].get_xticklabels() + ax[1].get_yticklabels()):
    item.set_fontsize(15)

ax[2].hist(data["correctedScaledAverageMu"][(data["type"]==0)], bins = 20, color='g', range = (40,60), histtype="step", label = "Background");
ax[2].hist(data["correctedScaledAverageMu"][(data["type"]==1)], bins = 20, color='tab:purple', range = (40,60), histtype="step", weights = 10*np.ones_like(data["pt"][(data["type"]==1)]), label = "Signal, rescaled with a factor of 10");
ax[2].set(xlabel = r"$\langle\mu\rangle$", ylabel = "Frequency")
ax[2].legend(prop={'size': 11.5})
for item in ([ax[2].title, ax[2].xaxis.label, ax[2].yaxis.label] +
             ax[2].get_xticklabels() + ax[2].get_yticklabels()):
    item.set_fontsize(15)

fig.tight_layout()
fig.savefig("Z model/output/MuoPairHdf5/160920_2/DataDists.pdf")
