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





def Multipleh5ToDf(filenames):
    """
    Make pandas dataframe from {filename}.h5 file.
    """
    files = filenames.split(",")
    d = {}
    for filename in files:
        print(f"Import data from: {filename}")
        with h5py.File(filename, "r") as hf :
            for name in list(hf.keys()):
                if name in d:
                    d[name] = np.append(d[name],hf[name][:])
                else:
                    d[name] = np.array(hf[name][:])
    df = pd.DataFrame(data=d)
    return(df)

dataMC = Multipleh5ToDf("/Users/sda/hep/work/hdf5Prod/output/root2hdf5/Zmm151020/Zmm151020_0004.h5,/Users/sda/hep/work/hdf5Prod/output/root2hdf5/Zmm151020/Zmm151020_0005.h5,/Users/sda/hep/work/hdf5Prod/output/root2hdf5/Zmm151020/Zmm151020_0006.h5,/Users/sda/hep/work/hdf5Prod/output/root2hdf5/Zmm151020/Zmm151020_0007.h5,/Users/sda/hep/work/hdf5Prod/output/root2hdf5/Zmm151020/Zmm151020_0008.h5,/Users/sda/hep/work/hdf5Prod/output/root2hdf5/Zmm151020/Zmm151020_0009.h5,/Users/sda/hep/work/hdf5Prod/output/root2hdf5/bb151020/bb151020_0004.h5,/Users/sda/hep/work/hdf5Prod/output/root2hdf5/bb151020/bb151020_0005.h5,/Users/sda/hep/work/hdf5Prod/output/root2hdf5/bb151020/bb151020_0006.h5,/Users/sda/hep/work/hdf5Prod/output/root2hdf5/bb151020/bb151020_0007.h5,/Users/sda/hep/work/hdf5Prod/output/root2hdf5/bb151020/bb151020_0008.h5,/Users/sda/hep/work/hdf5Prod/output/root2hdf5/bb151020/bb151020_0009.h5,/Users/sda/hep/work/hdf5Prod/output/root2hdf5/Wmn151020/Wmn151020_0004.h5,/Users/sda/hep/work/hdf5Prod/output/root2hdf5/Wmn151020/Wmn151020_0005.h5,/Users/sda/hep/work/hdf5Prod/output/root2hdf5/Wmn151020/Wmn151020_0006.h5,/Users/sda/hep/work/hdf5Prod/output/root2hdf5/Wmn151020/Wmn151020_0007.h5,/Users/sda/hep/work/hdf5Prod/output/root2hdf5/Wmn151020/Wmn151020_0008.h5,/Users/sda/hep/work/hdf5Prod/output/root2hdf5/Wmn151020/Wmn151020_0009.h5")
datadata = Multipleh5ToDf("/Users/sda/hep/work/hdf5Prod/output/root2hdf5/Data310820/Data310820_0004.h5,/Users/sda/hep/work/hdf5Prod/output/root2hdf5/Data310820/Data310820_0005.h5,/Users/sda/hep/work/hdf5Prod/output/root2hdf5/Data310820/Data310820_0006.h5,/Users/sda/hep/work/hdf5Prod/output/root2hdf5/Data310820/Data310820_0007.h5,/Users/sda/hep/work/hdf5Prod/output/root2hdf5/Data310820/Data310820_0008.h5,/Users/sda/hep/work/hdf5Prod/output/root2hdf5/Data310820/Data310820_0009.h5")
dataMCpair = h5ToDf("/Users/sda/hep/work/Zmm model/Z_model/output/MuoPairHdf5/151020_ZbbW/151020_ZbbW.h5")



fig, ax = plt.subplots(figsize=(12,7))
ax.hist(dataMCpair["correctedScaledAverageMu"], bins = 100, histtype="step", density = True, label = "MC after pairing, w/o weights");
ax.hist(dataMCpair["correctedScaledAverageMu"], bins = 100, histtype="step",weights = dataMCpair["eventWeight"], density = True, label = "MC after pairing, w/ weights");
ax.hist(dataMC["correctedScaledAverageMu"], bins = 100, density = True, histtype="step", label = "MC before pairing, w/o weights");
ax.hist(dataMC["correctedScaledAverageMu"], bins = 100, density = True, histtype="step", weights = dataMC["eventWeight"], label = "MC before pairing, w/ weights");
ax.hist(datadata["correctedScaledActualMu"], bins = 100, histtype="step", density = True, label = "Data, actualMu");
ax.legend()
fig.savefig("DifferentPileup.pdf")

fig, ax = plt.subplots(figsize=(12,7))
ax.hist(dataMCpair["NvtxReco"], bins =60, histtype="step", density = True, label = "MC after pairing, w/o weights");
ax.hist(dataMCpair["NvtxReco"], bins = 60, histtype="step",weights = dataMCpair["eventWeight"], density = True, label = "MC after pairing, w/ weights");
ax.hist(dataMC["NvtxReco"], bins = 60, density = True, histtype="step", label = "MC before pairing, w/o weights");
ax.hist(dataMC["NvtxReco"], bins = 60, density = True, histtype="step", weights = dataMC["eventWeight"], label = "MC before pairing, w/ weights");
ax.hist(datadata["NvtxReco"], bins = 60, histtype="step", density = True, label = "Data, actualMu");
ax.legend()
fig.savefig("DifferentNvtxReco.pdf")


def combinedVariables(hf, event, muoTag, muoProbe, phoProbe):
    """
    Calculate variables of the electron pair and photon.

    Arguments:
        hf: File to get variables from.
        event: Event number.
        eleTag: Index of tag electron.
        eleProbe: Index of probe electron.
        phoProbe: Index of probe photon.

    Returns:
        invM: Invariant mass of combined four-vector.
        et: Transverse energy of combined four-vector.
        eta: Eta of combined four-vector.
        phi: Phi of combined four-vector.
    """
    # Calculate invM etc. of eegamma using: https://github.com/scikit-hep/scikit-hep/blob/master/skhep/math/vectors.py?fbclid=IwAR3C0qnNlxKx-RhGjwo1c1FeZEpWbYqFrNmEqMv5iE-ibyPw_xEqmDYgRpc
    # Get variables
    p1 = hf['muo_pt'][event][muoTag]
    eta1 = hf['muo_eta'][event][muoTag]
    phi1 = hf['muo_phi'][event][muoTag]
    p2 = hf[f'muo_pt'][event][muoProbe]
    eta2 = hf[f'muo_eta'][event][muoProbe]
    phi2 = hf[f'muo_phi'][event][muoProbe]
    p3 = hf[f'pho_et'][event][phoProbe]
    eta3 = hf[f'pho_eta'][event][phoProbe]
    phi3 = hf[f'pho_phi'][event][phoProbe]

    # make four vector
    vecFour1 = vectors.LorentzVector()
    vecFour2 = vectors.LorentzVector()
    vecFour3 = vectors.LorentzVector()

    vecFour1.setptetaphim(p1,eta1,phi1,0)
    vecFour2.setptetaphim(p2,eta2,phi2,0)
    vecFour3.setptetaphim(p3,eta3,phi3,0)

    # calculate invariant mass
    vecFour = vecFour1+vecFour2+vecFour3
    invM = vecFour.mass
    pt = vecFour.pt
    eta = vecFour.eta
    phi = vecFour.phi()

    return invM, pt, eta, phi

from skhep.math import vectors


def getTagsAndProbes(hf, event):
    """
    Gets tag and probe indices.

    Arguments:
        hf: File to get variables from.
        event: Event number.

    Returns:
        mTag: Array of tag muons.
        mProbe: Array of probe muons.
        pProbe: Array of probe photons.
    """
    mTag = []
    mProbe = []
    pProbe = []

    # Get ele tags and probes
    for muo in range(len(hf['muo_truthOrigin'][event])):
        ptMuo = hf['muo_pt'][event][muo]/1000
        origZ = (hf['muo_truthOrigin'][event][muo] == 13)
        pdgId13 = (np.abs(hf['muo_truthPdgId'][event][muo]) == 13)
        trigger = hf['muo_trigger'][event][muo]
        LHTight = hf['muo_LHTight'][event][muo]

        if origZ * pdgId13 * trigger: # (trigger) and (ptMuo > 26)
            # print("triggered")
            mTag.append(muo)
        elif (origZ * pdgId13) == 1:
            mProbe.append(muo)
    # ptPho = hf['pho_et'][event][muo]
    # Get pho probes
    pProbe = np.arange(0,len(hf['pho_et'][event]),1)


    return mTag, mProbe, pProbe

dataZmmg
dataZmmg = h5ToDf("/Users/sda/hep/work/hdf5Prod/output/root2hdf5/Zmmgam20201022/Zmmgam20201022_0000.h5")
dataZmmg.columns
len(dataZmmg["muo_trigger"][7])
dataZmmg["muo_trigger"][8]
dataZmmg["muo_truthOrigin"][8]
len(dataZmmg["muo_truthOrigin"][6])
dataZmmg = h5ToDf("/Users/sda/hep/work/Zmmg/output/MuoPairGammaDataset/Zmmgam20201023/Zmmgam20201023.h5")
dataZmmg = h5ToDf("/Users/sda/hep/work/Zmmg/output/MuoPairGammaDataset/Zmmgam20201023_sigPhoOriMu_bkgRest/Zmmgam20201023.h5")

sig = dataZmmg[(dataZmmg["type"] == 1)]
bkg = dataZmmg[(dataZmmg["type"] == 0) & (abs(dataZmmg["pho_truthPdgId_atlas"]) == 22)]

pgd_bkg = bkg[(abs(bkg["pho_truthPdgId_atlas"]) < 100)]
plt.hist(abs(pgd_bkg["pho_truthPdgId_atlas"]), bins = 20);



np.unique(abs(bkg["pho_truthPdgId_atlas"]), return_counts = True)

bkg_tight = bkg[bkg["pho_isPhotonEMTight"] == 1]
np.unique(bkg_tight_invM["pho_truthOrigin"], return_counts = True)
# bkg = dataZmmg[(dataZmmg["type"] == 0) & (bkg["pho_truthOrigin"] == 3)]# & (bkg["pho_isPhotonEMTight"] == 0)]
# bkg = dataZmmg[(dataZmmg["type"] == 0) & (dataZmmg["pho_truthPdgId_atlas"] != 22) & (dataZmmg["pho_truthPdgId_atlas"] != 0)& (dataZmmg["pho_truthOrigin"] != 0) ]

bkg_tight_invM = bkg_tight[bkg_tight["invM"]/1000 > 80]

plt.hist(sig["invM"]/1000, bins = 100, range=(0,120), histtype="step");
plt.hist(bkg["invM"]/1000, bins = 100, range=(0,120), histtype="step");

plt.hist(bkg["pho_isPhotonEMTight"], bins = 2, histtype="step");


invM_save = []
pho_pt = []
pho_origin = []
tag_length = []
probe_length = []

for i in tqdm(range(len(dataZmmg["muo_truthOrigin"]))):
    tags, probes, pProbes = getTagsAndProbes(dataZmmg, i)

    if len(tags) > 0:
        tag_length.append(len(tags))
        probe_length.append(len(probes))

        for tag in tags:
            for probe in probes:
                if (dataZmmg[f'muo_truthPdgId'][i][tag] * dataZmmg[f'muo_truthPdgId'][i][probe]) < 0:
                    for pProbe in pProbes:
                        if (dataZmmg[f'pho_isPhotonEMTight'][i][pProbe]):
                            invM, pt, eta, phi = combinedVariables(dataZmmg, i, tag, probe, pProbe)
                            invM_save.append(invM/1000)
                            if (invM/1000 > 85) & (invM/1000 < 95):
                                pho_pt.append(dataZmmg[f'pho_et'][i][pProbe])
                                pho_origin.append(dataZmmg[f'pho_truthOrigin'][i][pProbe])
                        # print(invM/1000)
plt.hist(pho_origin, bins = 10);
plt.hist(probe_length, bins = 10);
plt.hist(invM_save,bins=100, range = (0 ,100));
plt.axvline(85, color = 'r');
plt.axvline(95, color = 'r');
# plt.axvline(80.385, color = 'r', label = "W mass");
plt.axvline( 91.1876, color = 'm', label = "Z mass");
plt.legend()

invM_save = np.array(invM_save)
(invM_save > 85)*1
[(invM_save > 85)*1 & (invM_save < 95)*1]
np.sum([(invM_save > 85)*1 & (invM_save < 95)*1])

dataZmmg[f'pho_et'][i][pProbe]

dataZmmg["muo_truthPdgId"]
invMass(dataZmmg, event, comb)
dataZmmg["pho_truthPdgId_atlas"]
plt.hist(dataZmmg["pho_truthOrigin"])


origin_list = []
pdgId_list = []
for origin, pdgid in zip(dataZmmg["pho_truthOrigin"], dataZmmg["pho_truthPdgId_atlas"]):
    for part1, part2 in zip(origin, pdgid):
        origin_list.append(part1)
        pdgId_list.append(part2)

origin_list = np.array(origin_list)
pdgId_list = np.array(pdgId_list)

origin_list_muo = []
pdgId_list_muo = []
for origin, pdgid in zip(dataZmmg["muo_truthOrigin"], dataZmmg["muo_truthPdgId"]):
    for part1, part2 in zip(origin, pdgid):
        origin_list_muo.append(part1)
        pdgId_list_muo.append(part2)


origin_list_muo = np.array(origin_list_muo)
pdgId_list_muo = np.array(pdgId_list_muo)
pdgId_list
origin_list[(abs(pdgId_list) == 22) & (origin_list > 0)]
pdgId_list[(abs(origin_list) == 13)]




plt.hist(origin_list[(abs(pdgId_list) == 22) & (origin_list > 0)], range = (15,45), bins = 30, label = "Origin photon");
plt.hist(origin_list_muo[(abs(pdgId_list_muo) == 13) & (origin_list_muo > 0)], range = (15,45), bins = 30, label = "Origin muon");
plt.legend();
np.unique(origin_list, return_counts=True)

np.unique(origin_list_muo, return_counts=True)
dataZmmg
plt.hist(dataZmmg["averageInteractionsPerCrossing"], weights = dataZmmg["eventWeight"], bins = 60);

dfg
dfg = dataZmmg.groupby("pho_truthOrigin").count().reset_index()
