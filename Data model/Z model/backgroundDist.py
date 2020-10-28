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


def plotRatioFigure(MCFile_bkg_var, data_var, var, range, bin, i, fig = None, ax = None, log = False):
    if fig is None:
        fig, ax = plt.subplots(1,1,figsize=(10,5), sharex=True, sharey=False)
    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("bottom", size="25%", pad=0)
    ax.figure.add_axes(ax2)
    #ax2.set_yticks((0.1,0.3))
    #ax2.set_ylim([-0.09,0.4])

    # Find normalization factor to get the errorbars
    count, bins  = np.histogram(data_var, bin, normed=False, range = rang)
    count_norm, bins  = np.histogram(data_var, bin, normed=True, range = rang)
    c = (count_norm/count)[0]

    countMC, binsMC  = np.histogram(MCFile_bkg_var, bin, normed=False, range = rang)
    count_normMC, binsMC  = np.histogram(MCFile_bkg_var, bin, normed=True, range = rang)
    cMC = (count_normMC/countMC)[0]

    bin_centers = (bins[1:] + bins[:-1])/2
    #bin_centers.shape

    #print(bins == binsMC)

    bkg_MC = ax.hist([MCFile_bkg_var], bins = bins, range = rang, color = 'tab:blue', stacked = True, density = True, label = ["MC bkg"]);
    ax.errorbar(bin_centers, count_norm, yerr = np.sqrt(count)*c, color = 'k', fmt = '.', label = "Data bkg distribution")
    ax2.errorbar(bin_centers, count_norm - count_normMC, yerr = (np.sqrt(count)*c + np.sqrt(countMC)*cMC) , color='k', fmt = '.', label = "Ratio Data - MC bkg");

    if log:
        ax2.set(yscale = "log");

    #print(count_norm/count_normMC)
    #ax2.plot(sig_data[1][:-1], sig_data[0]/sig_MC[0], 'ro', mfc='none', label = "Ratio Data/MC sig");
    ax2.axhline(0, color = 'tab:blue', linestyle = "dashed")

    plt.draw()
    labels=ax.get_xticklabels()

    for j,label in enumerate(labels):
       labels[j]=label.get_text()

    ax2.set_xticklabels(labels, rotation=0)

    ax2.set(xlabel = var, ylabel = "Data - MC");
    ax.set(ylabel = "Frequency", yscale = "log");

    if i == 0:
        ax.legend(prop={'size': 15}, loc = 4)
    #else:
    #    ax.legend(prop={'size': 15})
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)
    for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
                 ax2.get_xticklabels() + ax2.get_yticklabels()):
        item.set_fontsize(15)
    ax.set_xticks([])
    #plt.draw()
    return fig, ax
def GetISOscore(gbm, data, muoNr):
    training_var = [f'muo{muoNr}_etcone20',
                    f'muo{muoNr}_ptcone20',
                    f'muo{muoNr}_pt',
                    f'muo{muoNr}_etconecoreConeEnergyCorrection',
                    f'muo{muoNr}_neflowisolcoreConeEnergyCorrection',
                    f'muo{muoNr}_ptconecoreTrackPtrCorrection',
                    f'muo{muoNr}_topoetconecoreConeEnergyCorrection']
    score = gbm.predict(data[training_var], n_jobs=1)
    return logit(score)

def GetPIDscore(gbm, data, muoNr):
    training_var = [f'muo{muoNr}_numberOfPrecisionLayers',
                    f'muo{muoNr}_numberOfPrecisionHoleLayers',
                    f'muo{muoNr}_quality',
                    f'muo{muoNr}_ET_TileCore',
                    f'muo{muoNr}_MuonSpectrometerPt',
                    f'muo{muoNr}_deltatheta_1',
                    f'muo{muoNr}_scatteringCurvatureSignificance', # PID
                    f'muo{muoNr}_scatteringNeighbourSignificance', # PID
                    f'muo{muoNr}_momentumBalanceSignificance', # PID
                    f'muo{muoNr}_EnergyLoss', # PID
                    f'muo{muoNr}_energyLossType']

    score = gbm.predict(data[training_var], n_jobs=1)
    return logit(score)

training_var = [
'correctedScaledAverageMu',
'NvtxReco',
'Z_sig',
'muo1_PID_MC',
'muo1_ISO_MC',
'muo1_d0_d0Sig',
'muo1_priTrack_d0',
'muo1_priTrack_z0',
'muo2_PID_MC',
'muo2_ISO_MC',
'muo2_d0_d0Sig',
'muo2_priTrack_d0',
'muo2_priTrack_z0']

PID_MC = "/Users/sda/hep/work/Zmm model/PID_ISO_models/output/PIDModels/010920_ZbbW/lgbmPID.txt"
ISO_MC = "/Users/sda/hep/work/Zmm model/PID_ISO_models/output/ISOModels/110820_ZbbW/lgbmISO.txt"
PIDMC = lgb.Booster(model_file = PID_MC)
ISOMC = lgb.Booster(model_file = ISO_MC)


# MCFile = h5ToDf("/Users/sda/hep/work/Zmm model/Z_model/output/ZReweightFiles/131020_ZbbW/combined_131020_train.h5")
MCFile = h5ToDf("/Users/sda/hep/work/Data model/Z model/output/ZReweightFiles/141020_MC/combined_131020_train.h5")
data_file = h5ToDf("/Users/sda/hep/work/Data model/Z model/output/ZReweightFiles/170920_2/combined_170920_train.h5")
plt.hist(MCFile["NvtxReco"], bins=20);

data_file["correctedScaledAverageMu"].hist()

data_file['muo1_PID_MC'] = GetPIDscore(PIDMC,data_file,1)
data_file['muo2_PID_MC'] = GetPIDscore(PIDMC,data_file,2)
data_file['muo1_ISO_MC'] = GetISOscore(ISOMC,data_file,1)
data_file['muo2_ISO_MC'] = GetISOscore(ISOMC,data_file,2)
MCFile['muo1_PID_MC'] = GetPIDscore(PIDMC,MCFile,1)
MCFile['muo2_PID_MC'] = GetPIDscore(PIDMC,MCFile,2)
MCFile['muo1_ISO_MC'] = GetISOscore(ISOMC,MCFile,1)
MCFile['muo2_ISO_MC'] = GetISOscore(ISOMC,MCFile,2)

MCFile_bkg = MCFile[MCFile["label"] == 0]
# MCFile_bkg = MCFile[(MCFile["muo1_truthPdgId"] * MCFile["muo2_truthPdgId"]) > 0]
data_file_bkg = data_file[data_file["label"] == 0]


ranges = [(0, 120), (0, 80), (-200,200), (-20,20), (-6,4), (-100,100), (-100,100), (-200,200), (-20,20), (-6,4), (-100,100), (-100,100), (-200,200) ]
bins_all = [120, 80, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

fig, ax = plt.subplots(7,2,figsize=(10,35))
ax = ax.flatten()
for i, (var, rang, bin) in enumerate(zip(training_var, ranges, bins_all)):
    MCFile_bkg_var = (MCFile_bkg[var]).to_numpy()
    data_var = (data_file_bkg[var]).to_numpy()


    # Find normalization factor to get the errorbars
    count, bins  = np.histogram(data_var, bin, normed=False, range = rang)
    count_norm, bins  = np.histogram(data_var, bin, normed=True, range = rang)
    #n, bins, patches = plt.hist(data_var, 80, range = (0,100), stacked=True, density = False)
    #n_norm, bins, patches = plt.hist(data_var, 80, range = (0,100), stacked=True, density = True)
    c = (count_norm/count)[0]

    bin_centers = (bins[1:] + bins[:-1])/2
    bin_centers.shape
    #plt.clf()
    #fig, ax = plt.subplots(figsize=(6,5))
    #ax.hist([Wmn_var, Zmm_var, bb_var], bins = bins, stacked = True, density = True, label = ["Wmn", "Zmm bkg", "bb"]);
    ax[i].hist([MCFile_bkg_var], bins = bins, range = rang, stacked = True, density = True, label = ["MC bkg"]);
    ax[i].errorbar(bin_centers, count_norm, yerr = np.sqrt(count)*c, color = 'k', fmt = '.', label = "Data bkg distribution")
    ax[i].set(xlabel = var, ylabel = "Frequency", yscale = "log")
    if i == 0:
        ax[i].legend(prop={'size': 15}, loc = 4)
    else:
        ax[i].legend(prop={'size': 15})
    for item in ([ax[i].title, ax[i].xaxis.label, ax[i].yaxis.label] +
                 ax[i].get_xticklabels() + ax[i].get_yticklabels()):
        item.set_fontsize(15)
fig.tight_layout()


fig.savefig("BackgroundDistMCvsData.pdf")


#### Ratio plots
training_var = [
'correctedScaledAverageMu',
'NvtxReco',
'Z_sig',
'muo1_PID_MC',
'muo1_ISO_MC',
'muo1_d0_d0Sig',
'muo1_priTrack_d0',
'muo1_priTrack_z0',
'muo2_PID_MC',
'muo2_ISO_MC',
'muo2_d0_d0Sig',
'muo2_priTrack_d0',
'muo2_priTrack_z0']

ranges = [(0, 120), (0, 80), (-400,400), (-20,20), (-6,4), (-100,100), (-100,100), (-200,200), (-20,20), (-6,4), (-100,100), (-100,100), (-200,200) ]
bins_all = [120, 80, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

logVars = ["muo1_PID_score", "muo2_PID_score", "muo1_ISO_score", "muo2_ISO_score"]

fig, ax = plt.subplots(3,5,figsize=(25,15))
ax = ax.flatten()
for i, (var, rang, bin) in enumerate(zip(training_var, ranges, bins_all)):
    if i < 3:
        j = i+1
    else:
        j = i+2
    MCFile_bkg_var = (MCFile_bkg[var]).to_numpy()
    data_var = (data_file_bkg[var]).to_numpy()

    if var in logVars:
        plotRatioFigure(MCFile_bkg_var, data_var, var, rang, bin, i, fig = fig, ax = ax[j], log = True)
    else:
        plotRatioFigure(MCFile_bkg_var, data_var, var, rang, bin, i, fig = fig, ax = ax[j])
fig.tight_layout()
fig.savefig("BackgroundDistMCvsDataRatioTest.pdf")


########## Probe distributions ##########



MCFileProbe = h5ToDf("/Users/sda/hep/work/Data model/createDataset/output/MuoSingleHdf5/080920_MC/080920_MC.h5")
dataFileProbe = h5ToDf("/Users/sda/hep/work/Data model/createDataset/output/MuoSingleHdf5/010920_3/010920_3.h5")


training_var = [
'muo_numberOfPrecisionLayers',
'muo_numberOfPrecisionHoleLayers',
'muo_quality',
'muo_ET_TileCore',
'muo_MuonSpectrometerPt',
'muo_deltatheta_1',
'muo_scatteringCurvatureSignificance', # PID
'muo_scatteringNeighbourSignificance', # PID
'muo_momentumBalanceSignificance', # PID
'muo_EnergyLoss', # PID
'muo_energyLossType',
'muo_etcone20',
'muo_ptcone20',
'muo_pt',
'muo_etconecoreConeEnergyCorrection',
'muo_neflowisolcoreConeEnergyCorrection',
'muo_ptconecoreTrackPtrCorrection',
'muo_topoetconecoreConeEnergyCorrection'
]

MCFileProbe_bkg = MCFileProbe[MCFileProbe["Type"] == 0]
dataFileProbe_bkg = dataFileProbe[dataFileProbe["Type"] == 0]



ranges = [(0, 6), (0, 5), (0,11), (0,3000), (0,120), (-1000,100), (-2,2), (-2,2), (-0.5,0.5), (2000,6000), (0,4),
            (-15000, 15000), (0,5000), (0, 120), (-1500, 10000), (0,4000), (0,40000), (500, 40000), (0,10000) ]
bins_all = [10, 10, 11, 100, 100, 100, 100, 100, 100, 100, 10,
            100, 100, 100, 100, 100, 100, 100, 100]

fig, ax = plt.subplots(6,3,figsize=(15,30))
ax = ax.flatten()
for i, (var, rang, bin) in enumerate(zip(training_var, ranges, bins_all)):
    MCFile_bkg_var = (MCFileProbe_bkg[var]).to_numpy()
    data_var = (dataFileProbe_bkg[var]).to_numpy()

    #if var in logVars:
    #    plotRatioFigure(MCFile_bkg_var, data_var, var, rang, bin, i, fig = fig, ax = ax[i], log = True)
    #else:
    if i == 4 or i == 13:
        plotRatioFigure(MCFile_bkg_var/1000, data_var/1000, var, rang, bin, i, fig = fig, ax = ax[i])
    else:
        plotRatioFigure(MCFile_bkg_var, data_var, var, rang, bin, i, fig = fig, ax = ax[i])

fig.tight_layout()
fig.savefig("ProbeDistMCvsDataRatio.pdf")


fig, ax = plt.subplots(6,3,figsize=(15,30))
ax = ax.flatten()
for i, (var, rang, bin) in enumerate(zip(training_var, ranges, bins_all)):
    MCFile_bkg_var = (MCFileProbe_bkg[var]).to_numpy()
    data_var = (dataFileProbe_bkg[var]).to_numpy()

    #if var in logVars:
    #    plotRatioFigure(MCFile_bkg_var, data_var, var, rang, bin, i, fig = fig, ax = ax[i], log = True)
    #else:
    if i == 4 or i == 13:
        plotRatioFigure(MCFile_bkg_var/1000, data_var/1000, var, rang, bin, i, fig = fig, ax = ax[i], log = True)
    else:
        plotRatioFigure(MCFile_bkg_var, data_var, var, rang, bin, i, fig = fig, ax = ax[i], log = True)

fig.tight_layout()
fig.savefig("ProbeDistMCvsDataRatioLog.pdf")
