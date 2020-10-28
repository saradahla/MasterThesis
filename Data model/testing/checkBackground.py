import h5py
import numpy as np
import logging as log
import argparse
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib

from utils import mkdir
from itertools import combinations
from skhep.math import vectors
import multiprocessing

from time import time
from datetime import timedelta
import pandas as pd

import lightgbm as lgb
from scipy.special import logit
import klib
import seaborn as sns

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

bb_file = h5ToDf("/Users/sda/hep/work/Zmm model/PID_ISO_models/output/PID_ISO_Dataset/110920_bb/110920_bb.h5")
Wmn_file = h5ToDf("/Users/sda/hep/work/Zmm model/PID_ISO_models/output/PID_ISO_Dataset/110920_Wmn/110920_Wmn.h5")
Zmm_file = h5ToDf("/Users/sda/hep/work/Zmm model/PID_ISO_models/output/PID_ISO_Dataset/110920_Zmm/110920_Zmm.h5")
data_file = h5ToDf("/Users/sda/hep/work/Data model/output/MuoSingleHdf5/010920_3/010920_3.h5")


Zmm_file["label"] = 0
Zmm_file.loc[Zmm_file["muo_truthOrigin"]==13,"label"] = 1


np.unique(Zmm_file["label"], return_counts = True)
Zmm_file_bkg = Zmm_file[Zmm_file["label"] == 0]
data_file_bkg = data_file[data_file["Type"] == 0]

bb_pt = (bb_file["muo_eta"]).to_numpy()
Wmn_pt = (Wmn_file["muo_eta"]).to_numpy()
Zmm_pt = (Zmm_file_bkg["muo_eta"]).to_numpy()
data_pt = (data_file_bkg["muo_eta"]).to_numpy()


# Find normalization factor to get the errorbars
n, bins, patches = plt.hist(data_pt, 80, range = (-3,3), stacked=True, density = False)
n_norm, bins, patches = plt.hist(data_pt, 80, range = (-3,3), stacked=True, density = True)
c = (n_norm/n)[0]

bin_centers = (bins[1:] + bins[:-1])/2
bin_centers.shape

fig, ax = plt.subplots(figsize=(6,5))
ax.hist([Wmn_pt, Zmm_pt, bb_pt], bins = bins, stacked = True, density = True, label = ["Wmn", "Zmm bkg", "bb"]);
ax.errorbar(bin_centers, n_norm, yerr = np.sqrt(n)*c, color = 'k', fmt = '.', label = "Data bkg distribution")
ax.set(yscale = "log", xlabel = "eta", ylabel = "Frequency")
ax.legend(prop={'size': 15})
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)
fig.tight_layout()
fig.savefig("BackgroundDistMCvsDataEta.pdf")
