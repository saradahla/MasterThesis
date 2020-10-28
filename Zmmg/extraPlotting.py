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
from tqdm import tqdm

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

dataZmmg = h5ToDf("/Users/sda/hep/work/Zmmg/output/MuoPairGammaDataset/Zmmgam20201023/Zmmgam20201023.h5")


sig = dataZmmg[(dataZmmg["type"] == 1) & (dataZmmg["pho_et"] > 9.5) ]
sig2 = dataZmmg[(dataZmmg["type"] == 1) & (dataZmmg["pho_et"] > 9.5) & (dataZmmg["muo2_pt"]/1000 > 10) &  (dataZmmg["invM"]/1000 > 84) & (dataZmmg["invM"]/1000 < 96)]
bkg = dataZmmg[(dataZmmg["type"] == 0)]

# (len(sig)/len(dataZmmg))*100 # percentage signal


fig, ax = plt.subplots(1,3, figsize=(15,5))
ax[0].hist(dataZmmg["invM"]/1000, bins = 100, range = (0,120), color = 'k', histtype = "step");
ax[0].set(xlabel = "invM", ylabel = "Frequency", title = "All data");
ax[1].hist(sig["invM"]/1000, bins = 100, range = (0,120), color = 'r', histtype = "step", label = "Signal");
ax[1].hist(bkg["invM"]/1000, bins = 100, range = (0,120), color = 'b', histtype = "step", label = "background");
ax[1].legend()
ax[1].set(xlabel = "invM", ylabel = "Frequency", title = "Signal and background \n (signal: T&P muons and truth photon > 9.5 GeV \n originating from muon)");
ax[2].hist(sig["invM"]/1000, density = True, bins = 100, range = (0,120), color = 'r', histtype = "step", label = "Signal");
ax[2].hist(bkg["invM"]/1000, density = True,bins = 100, range = (0,120), color = 'b', histtype = "step", label = "background");
ax[2].legend()
ax[2].set(xlabel = "invM", ylabel = "Frequency", title = "Signal and background, normalized");

fig.tight_layout()
fig.savefig("invMdists_mumugam.pdf")

## Where does the background in the range 84 to 96 GeV come from?
bkg_invM = bkg[(bkg["invM"]/1000 > 84) & (bkg["invM"]/1000 < 96)]
len(labels)
np.unique(abs(bkg_invM["pho_truthPdgId_atlas"]), return_counts = True)

fig, ax = plt.subplots(1,1,figsize=(7,5))
ax.hist(bkg_invM["pho_truthOrigin"], bins = 42);
# ax.set_yscale('log');
labels = np.unique(bkg_invM["pho_truthOrigin"])
ax.set_xticks(labels);
ax.set_xticklabels(labels, rotation=45);
ax.set(xlabel = "truthOrigin, bkg events in range [84;96] GeV", ylabel = "Frequency");

# ax[1].hist(abs(bkg_invM["pho_truthPdgId_atlas"][bkg_invM["pho_truthPdgId_atlas"] < 400]), bins = 100);
# # ax.set_yscale('log');
# labs = np.unique(abs(bkg_invM["pho_truthPdgId_atlas"])) < 400
# labelsPdg = np.unique(abs(bkg_invM["pho_truthPdgId_atlas"]))[labs]
# labelsPdg
# # ax[1].set_xticks([]);
# ax[1].set_xticks(labelsPdg);
# ax[1].set_xticklabels(labelsPdg, rotation=45);
# ax[1].set(xlabel = "abs(truthPdgId), bkg events in range [84;96] GeV", ylabel = "Frequency");
fig.tight_layout()
fig.savefig("BkgEventsOriginPdgId.pdf")
labels
labels
plt.hist(bkg_invM["invM"]/1000, bins = 100, range=(0,120), histtype="step");
plt.hist(sig["invM"]/1000, bins = 100, range=(0,120), histtype="step");
# plt.hist(sig_test["invM"]/1000, bins = 100, range=(0,120), histtype="step");
plt.hist(bkg["invM"]/1000, density=True, bins = 100, range=(0,120), histtype="step");

dataPho = h5ToDf("/Users/sda/hep/work/Zmmg/output/pho_Dataset/20201027/20201027.h5")
len(sig) + len(bkg)
len(dataPho)
sig = (dataPho["pho_truthPdgId_atlas"] == 22) & (dataPho["pho_et"] > 4.5) & (dataPho["pho_truthOrigin"] == 3)
bkg = dataPho[~sig]
sig = dataPho[sig]


fig, ax = plt.subplots(1,3,figsize=(15,5))
ax = ax.flatten()
ax[0].hist(sig["correctedScaledAverageMu"], color = 'r', range = (0,90), histtype = "step", bins = 90);
ax[0].hist(bkg["correctedScaledAverageMu"], color = 'b', range = (0,90), histtype = "step", bins = 90);
ax[0].set(xlabel = r"$\langle\mu\rangle$", ylabel = "Frequency", yscale = 'log')
ax[1].hist(sig["pho_eta"], color = 'r', range = (-3,3), histtype = "step", bins = 90);
ax[1].hist(bkg["pho_eta"], color = 'b', range = (-3,3), histtype = "step", bins = 90);
ax[1].set(xlabel = r"$\eta$", ylabel = "Frequency", yscale = 'log')
ax[2].hist(sig["pho_et"], color = 'r', range = (0,50), histtype = "step", bins = 90);
ax[2].hist(bkg["pho_et"], color = 'b', range = (0,50), histtype = "step", bins = 90);
ax[2].set(xlabel = r"E$_T$", ylabel = "Frequency", yscale = 'log');

fig.tight_layout()


sig_type = sig["pho_truthType"]
bkg_type = bkg["pho_truthType"]

sig_unconv = sig[sig["pho_ConversionType"] == 0]
sig_conv = sig[sig["pho_ConversionType"] != 0]


len(sig_unconv)
len(sig_conv)

nSig = (10 + 2329 + 397)
pUnkown = (10/nSig)*100
pIso = (2329/nSig)*100
pBkg = (397/nSig)*100
pBkg
np.unique(sig_type, return_counts = True)[1]
np.unique(bkg_type, return_counts = True)



print(f"The signal data set is {len(sig)} long")
sigType = np.unique(sig_type, return_counts = True)
print(f"It has photons with type {sigType[0]} and counts: {sigType[1]}")
print(f"Yielding percentage: {(sigType[1][0]/len(sig))*100} %, {(sigType[1][1]/len(sig))*100} %, {(sigType[1][2]/len(sig))*100} %")
sig.reset_index(drop=True, inplace=True)


list_orgs = []
list_pdg = []
for i in range(len(dataPho)):
    if dataPho["pho_truthType"][i] == 14:
        list_orgs.append(dataPho["pho_truthOrigin"][i])
        list_pdg.append(dataPho["pho_truthPdgId_atlas"][i])
        # print(dataPho["pho_truthOrigin"][i])

list = [1,2,3,4,5,6,6,7,8,9,9,10,12]

list[::5]

fig, ax = plt.subplots()
ax.hist(np.array(list_pdg))
ax.set_xticks([20,21,22,23,24]);
ax.set_xticklabels([20,22,24]);


print(f"\n\n")
print(f"The background data set is {len(bkg)} long")
bkgType = np.unique(bkg_type, return_counts = True)
print(f"It has photons with type {bkgType[0]} and counts: {bkgType[1]}")
print(f"Yielding percentage: {(bkgType[1][0]/len(bkg))*100} %, {(bkgType[1][1]/len(bkg))*100} %, {(bkgType[1][2]/len(bkg))*100} %")
print(f"{(bkgType[1][3]/len(bkg))*100} %, {(bkgType[1][4]/len(bkg))*100} %, {(bkgType[1][5]/len(bkg))*100} %")
print(f"{(bkgType[1][6]/len(bkg))*100} %, {(bkgType[1][7]/len(bkg))*100} %, {(bkgType[1][8]/len(bkg))*100} %")
