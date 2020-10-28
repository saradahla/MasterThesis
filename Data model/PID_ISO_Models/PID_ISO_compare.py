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

def plotRatioFigure(MCFile_bkg_var, data_var, var, rang, bin, i, fig = None, ax = None, log = False):
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

    bkg_MC = ax.hist(MCFile_bkg_var, bins = bins, range = rang, color = 'tab:blue', stacked = True, density = True, label = "MC bkg");
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
        item.set_fontsize(20)
    for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
                 ax2.get_xticklabels() + ax2.get_yticklabels()):
        item.set_fontsize(20)
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

PID_Data = "/Users/sda/hep/work/Data model/PID_ISO_Models/output/PIDModels/160920_Data/lgbmPID.txt"
ISO_Data = "/Users/sda/hep/work/Data model/PID_ISO_Models/output/ISOModels/230920/lgbmISO.txt"
PID_MC = "/Users/sda/hep/work/Zmm model/PID_ISO_models/output/PIDModels/010920_ZbbW/lgbmPID.txt"
ISO_MC = "/Users/sda/hep/work/Zmm model/PID_ISO_models/output/ISOModels/110820_ZbbW/lgbmISO.txt"

MCFile = h5ToDf("/Users/sda/hep/work/Zmm model/Z_model/output/ZReweightFiles/220920_ZbbW/combined_220920_train.h5")
data_file = h5ToDf("/Users/sda/hep/work/Data model/Z model/output/ZReweightFiles/170920_2/combined_170920_train.h5")


PIDData = lgb.Booster(model_file = PID_Data)
PIDMC = lgb.Booster(model_file = PID_MC)
ISOData = lgb.Booster(model_file = ISO_Data)
ISOMC = lgb.Booster(model_file = ISO_MC)

data_file['muo1_PID_Data'] = GetPIDscore(PIDData,data_file,1)
data_file['muo2_PID_Data'] = GetPIDscore(PIDData,data_file,2)
data_file['muo1_PID_MC'] = GetPIDscore(PIDMC,data_file,1)
data_file['muo2_PID_MC'] = GetPIDscore(PIDMC,data_file,2)

data_file['muo1_ISO_Data'] = GetISOscore(ISOData,data_file,1)
data_file['muo2_ISO_Data'] = GetISOscore(ISOData,data_file,2)
data_file['muo1_ISO_MC'] = GetISOscore(ISOMC,data_file,1)
data_file['muo2_ISO_MC'] = GetISOscore(ISOMC,data_file,2)

MCFile['muo1_PID_Data'] = GetPIDscore(PIDData,MCFile,1)
MCFile['muo2_PID_Data'] = GetPIDscore(PIDData,MCFile,2)
MCFile['muo1_PID_MC'] = GetPIDscore(PIDMC,MCFile,1)
MCFile['muo2_PID_MC'] = GetPIDscore(PIDMC,MCFile,2)

MCFile['muo1_ISO_Data'] = GetISOscore(ISOData,MCFile,1)
MCFile['muo2_ISO_Data'] = GetISOscore(ISOData,MCFile,2)
MCFile['muo1_ISO_MC'] = GetISOscore(ISOMC,MCFile,1)
MCFile['muo2_ISO_MC'] = GetISOscore(ISOMC,MCFile,2)

fig, ax = plt.subplots(2,2,figsize=(10,10))
ax = ax.flatten()
plotRatioFigure(MCFile['muo1_ISO_Data'][MCFile['label'] == 0], data_file['muo1_ISO_Data'][data_file['label'] == 0], "Data trained, muon1", (-6,6), 100, 0, fig = fig, ax = ax[0], log = False)
plotRatioFigure(MCFile['muo1_ISO_MC'][MCFile['label'] == 0], data_file['muo1_ISO_MC'][data_file['label'] == 0], "MC trained, muon1", (-6,6), 100, 1, fig = fig, ax = ax[1], log = False)
plotRatioFigure(MCFile['muo2_ISO_Data'][MCFile['label'] == 0], data_file['muo2_ISO_Data'][data_file['label'] == 0], "Data trained, muon2", (-6,6), 100, 2, fig = fig, ax = ax[2], log = False)
plotRatioFigure(MCFile['muo2_ISO_MC'][MCFile['label'] == 0], data_file['muo2_ISO_MC'][data_file['label'] == 0], "MC trained, muon2", (-6,6), 100, 3, fig = fig, ax = ax[3], log = False)
fig.tight_layout()
fig.savefig("DataMCcompareISO.pdf")

fig, ax = plt.subplots(2,2,figsize=(10,10))
ax = ax.flatten()
plotRatioFigure(MCFile['muo1_PID_Data'][MCFile['label'] == 0], data_file['muo1_PID_Data'][data_file['label'] == 0], "Data trained, muon1", (-10,10), 100, 0, fig = fig, ax = ax[0], log = False)
plotRatioFigure(MCFile['muo1_PID_MC'][MCFile['label'] == 0], data_file['muo1_PID_MC'][data_file['label'] == 0], "MC trained, muon1", (-20,20), 100, 1, fig = fig, ax = ax[1], log = False)
plotRatioFigure(MCFile['muo2_PID_Data'][MCFile['label'] == 0], data_file['muo2_PID_Data'][data_file['label'] == 0], "Data trained, muon2", (-10,10), 100, 2, fig = fig, ax = ax[2], log = False)
plotRatioFigure(MCFile['muo2_PID_MC'][MCFile['label'] == 0], data_file['muo2_PID_MC'][data_file['label'] == 0], "MC trained, muon2", (-20,20), 100, 3, fig = fig, ax = ax[3], log = False)
fig.tight_layout()
fig.savefig("DataMCcomparePID.pdf")

MCFile.columns
