import h5py
import numpy as np
import logging as log
import argparse
import os
import matplotlib.pyplot as plt
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

hf_data = h5ToDf("/Users/sda/hep/work/Data model/Z model/output/ZReweightFiles/170920/combined_160920_train.h5")

plt.hist(hf_data["muo2_PID_score_ATLAS"][hf_data["label"] == 1], bins = 100, histtype = "step");
plt.hist(hf_data["muo2_PID_score_ATLAS"][hf_data["label"] == 0], bins = 100, histtype = "step");

plt.hist(hf_data["pt"][hf_data["label"] == 0], bins = 100, histtype = "step", range = (4.5,100));
plt.hist(hf_data["pt"][hf_data["label"] == 1], bins = 100, histtype = "step", range = (4.5,100));



c,n=np.unique(hf_data["muo1_PID_score"][(hf_data["muo1_PID_score"] < -5.5) & (hf_data["muo1_PID_score"] > -6.5) ], return_counts=True)
c[697]
n.argmax()
np.unique(hf_data["muo1_PID_score"][hf_data["muo1_PID_score"] == c[697]], return_counts = True)

np.unique(hf_data[hf_data["muo1_PID_score"] == c[697]]["muo1_deltatheta_1"], return_counts = True)


hf_data[hf_data["muo1_PID_score"] == c[697]]["muo1_deltatheta_1"]

np.unique(hf_data["label"], return_counts = True)
np.unique(hf_data["muo1_CaloLRLikelihood"], return_counts = True)


plt.hist(hf_data["muo1_InnerDetectorPt"][type == 0]/1000, bins = 100,range=(0,100));
plt.hist(hf_data["muo1_InnerDetectorPt"][type == 1]/1000, bins = 100,range=(0,100));
#plt.ylim([0,800])



hf_data["muo1_ET_TileCore"]

type = hf_data["label"]
fig, ax = plt.subplots(1,3,figsize=(15,5))
ax = ax.flatten()
ax[0].plot(hf_data['muo1_PID_score'][type==0],hf_data['muo1_ISO_score'][type==0],'.', color = 'g', alpha=0.3, label = "Background");
ax[0].plot(hf_data['muo1_PID_score'][type==1],hf_data['muo1_ISO_score'][type==1],'.', color = 'tab:purple', alpha=1, label = "Signal");
ax[0].set(xlabel = "ML PID 11 vars", ylabel = "ML ISO")
ax[0].legend()
for item in ([ax[0].title, ax[0].xaxis.label, ax[0].yaxis.label] +
         ax[0].get_xticklabels() + ax[0].get_yticklabels()):
    item.set_fontsize(20)

ax[1].plot(hf_data['muo1_PID_score_ATLAS'][type==0],hf_data['muo1_ISO_score'][type==0],'.', color = 'g', alpha=0.3, label = "Background");
ax[1].plot(hf_data['muo1_PID_score_ATLAS'][type==1],hf_data['muo1_ISO_score'][type==1],'.', color = 'tab:purple', alpha=1, label = "Signal");
ax[1].set(xlabel = "ML PID 8 vars", ylabel = "ML ISO")
ax[1].legend()
for item in ([ax[1].title, ax[1].xaxis.label, ax[1].yaxis.label] +
         ax[1].get_xticklabels() + ax[1].get_yticklabels()):
    item.set_fontsize(20)

ax[2].plot(hf_data['muo1_PID_score_ATLAS6'][type==0],hf_data['muo1_ISO_score'][type==0],'.', color = 'g', alpha=0.3, label = "Background");
ax[2].plot(hf_data['muo1_PID_score_ATLAS6'][type==1],hf_data['muo1_ISO_score'][type==1],'.', color = 'tab:purple', alpha=1, label = "Signal");
ax[2].set(xlabel = "ML PID 6 vars", ylabel = "ML ISO")
ax[2].legend()
for item in ([ax[2].title, ax[2].xaxis.label, ax[2].yaxis.label] +
         ax[2].get_xticklabels() + ax[2].get_yticklabels()):
    item.set_fontsize(20)
fig.tight_layout()
fig.savefig("Z model/MLpredPID_ISO_3.png", dpi=600)
