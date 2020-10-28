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


training_var_PIDATLAS = [#f'muo_numberOfPrecisionLayers',
                f'muo_numberOfPrecisionHoleLayers',
                #f'muo_quality',
                f'muo_ET_TileCore',
                #f'muo_MuonSpectrometerPt',
                f'muo_deltatheta_1',
                'muo_scatteringCurvatureSignificance', # PID
                'muo_scatteringNeighbourSignificance', # PID
                'muo_momentumBalanceSignificance', # PID
                'muo_EnergyLoss', # PID
                'muo_energyLossType']

training_var_PID = [f'muo_numberOfPrecisionLayers',
                f'muo_numberOfPrecisionHoleLayers',
                f'muo_quality',
                f'muo_ET_TileCore',
                f'muo_MuonSpectrometerPt',
                f'muo_deltatheta_1',
                'muo_scatteringCurvatureSignificance', # PID
                'muo_scatteringNeighbourSignificance', # PID
                'muo_momentumBalanceSignificance', # PID
                'muo_EnergyLoss', # PID
                'muo_energyLossType']

training_var_ISO = [f'muo_etcone20',
                f'muo_ptcone20',
                f'muo_pt',
                f'muo_etconecoreConeEnergyCorrection',
                f'muo_neflowisolcoreConeEnergyCorrection',
                f'muo_ptconecoreTrackPtrCorrection',
                f'muo_topoetconecoreConeEnergyCorrection']


hf_data = h5ToDf("/Users/sda/hep/work/Data model/output/MuoSingleHdf5/090920/090920.h5")
print(f"Shape of data before cutting values with pt < 4.5: {hf_data.shape}")
mask_pt = hf_data["muo_pt"]/1000 > 4.5
hf_data = hf_data[mask_pt]
print(f"Shape of data after cutting values with pt < 4.5: {hf_data.shape}")
type_data = hf_data["Type"]


hf_MC = h5ToDf("/Users/sda/hep/work/Data model/output/MuoSingleHdf5/080920_MC/080920_MC.h5")
print(f"Shape of MC before cutting values with pt < 4.5: {hf_MC.shape}")
mask_pt_MC = hf_MC["muo_pt"]/1000 > 4.5
hf_MC = hf_MC[mask_pt_MC]
print(f"Shape of MC after cutting values with pt < 4.5: {hf_MC.shape}")
type_MC = hf_MC["Type"]


def plotRatioFigure(hf_MC, hf_data, var, rangemin, rangemax, bins=100, fig = None, ax = None):
    if fig is None:
        fig, ax = plt.subplots(1,1,figsize=(10,5), sharex=True, sharey=False)
    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("bottom", size="25%", pad=0)
    ax.figure.add_axes(ax2)
    #ax.set_xticks([])
    ax2.set_yticks((0.1,0.3))
    ax2.set_ylim([-0.09,0.4])

    bkg_MC = ax.hist(hf_MC[var][type_MC == 0], bins = bins, range = (rangemin, rangemax), histtype = "step", color = 'b', label = "MC T&P Background");
    sig_MC = ax.hist(hf_MC[var][type_MC == 1], bins = bins, range = (rangemin, rangemax), histtype = "step", color = 'r', label = "MC T&P Signal");
    bkg_data = ax.hist(hf_data[var][type_data == 0], bins = bins, range = (rangemin, rangemax), histtype = "step", color = 'g', label = "Data Background");
    sig_data = ax.hist(hf_data[var][type_data == 1], bins = bins, range = (rangemin, rangemax), histtype = "step", color = 'tab:purple', label = "Data Signal");

    ax2.plot(bkg_data[1][:-1], bkg_data[0]/bkg_MC[0] , 'b.', label = "Ratio Data/MC bkg");
    ax2.plot(sig_data[1][:-1], sig_data[0]/sig_MC[0], 'ro', mfc='none', label = "Ratio Data/MC sig");
    ax2.axhline(0, color = 'k', linestyle = "dashed", label = "Zero")
    #ax2.legend()

    plt.draw()
    labels=ax.get_xticklabels()

    for i,label in enumerate(labels):
       labels[i]=label.get_text()

    ax2.set_xticklabels(labels, rotation=0)

    ax2.set(xlabel = var, ylabel = "Ratio MC/Data");
    ax.set(ylabel = "Frequency");
    ax.legend()
    #plt.draw()
    return fig, ax


def plotFigure(hf_data, var, rangemin, rangemax, title, bins=100, fig = None, ax = None):
    if fig is None:
        fig, ax = plt.subplots(1,1,figsize=(10,5), sharex=True, sharey=False)
    # divider = make_axes_locatable(ax)
    # ax2 = divider.append_axes("bottom", size="25%", pad=0)
    # ax.figure.add_axes(ax2)
    # #ax.set_xticks([])
    # ax2.set_yticks((0.1,0.3))
    # ax2.set_ylim([-0.09,0.4])

    #bkg_MC = ax.hist(hf_MC[var][type_MC == 0], bins = bins, range = (rangemin, rangemax), histtype = "step", color = 'b', label = "MC T&P Background");
    #sig_MC = ax.hist(hf_MC[var][type_MC == 1], bins = bins, range = (rangemin, rangemax), histtype = "step", color = 'r', label = "MC T&P Signal");
    ax.set_title(title)
    bkg_data = ax.hist(hf_data[var][type_data == 0], bins = bins, range = (rangemin, rangemax), histtype = "step", color = 'b', label = "Data Background");
    sig_data = ax.hist(hf_data[var][type_data == 1], bins = bins, range = (rangemin, rangemax), histtype = "step", color = 'r', label = "Data Signal");

    #ax2.plot(bkg_data[1][:-1], bkg_data[0]/bkg_MC[0] , 'b.', label = "Ratio Data/MC bkg");
    #ax2.plot(sig_data[1][:-1], sig_data[0]/sig_MC[0], 'ro', mfc='none', label = "Ratio Data/MC sig");
    #ax2.axhline(0, color = 'k', linestyle = "dashed", label = "Zero")
    #ax2.legend()

    # plt.draw()
    # labels=ax.get_xticklabels()
    #
    # for i,label in enumerate(labels):
    #    labels[i]=label.get_text()
    #
    # ax2.set_xticklabels(labels, rotation=0)
    #
    # ax2.set(xlabel = var, ylabel = "Ratio MC/Data");
    ax.set(ylabel = "Frequency", xlabel = var);
    ax.legend()
    #plt.draw()
    return fig, ax

fig, ax = plt.subplots(5,2,figsize=(15,20))
ax = ax.flatten()
fig, ax[0] = plotRatioFigure(hf_MC/1000, hf_data/1000, training_var_ISO[2], 0, 120, bins=100, fig = fig, ax = ax[0])
fig, ax[1] = plotRatioFigure(hf_MC, hf_data, training_var_ISO[0], -15000, 15000, bins=100, fig = fig, ax = ax[1])
fig, ax[2] = plotRatioFigure(hf_MC, hf_data, training_var_ISO[1], 0, 5000, bins=100, fig = fig, ax = ax[2])
fig, ax[3] = plotRatioFigure(hf_MC, hf_data, training_var_ISO[3], -1500,10000, bins=100, fig = fig, ax = ax[3])
fig, ax[4] = plotRatioFigure(hf_MC, hf_data, training_var_ISO[4], 0, 4000, bins=100, fig = fig, ax = ax[4])
fig, ax[5] = plotRatioFigure(hf_MC, hf_data, training_var_ISO[4], 100, 4000, bins=100, fig = fig, ax = ax[5])
fig, ax[6] = plotRatioFigure(hf_MC, hf_data, training_var_ISO[5], 0, 40000,bins=100, fig = fig, ax = ax[6])
fig, ax[7] = plotRatioFigure(hf_MC, hf_data, training_var_ISO[5], 500, 40000, bins=100, fig = fig, ax = ax[7])
fig, ax[8] = plotRatioFigure(hf_MC, hf_data, training_var_ISO[6], 0, 10000,bins=100, fig = fig, ax = ax[8])
fig, ax[9] = plotRatioFigure(hf_MC, hf_data, t raining_var_ISO[6], 500, 10000, bins=100, fig = fig, ax = ax[9])

fig.tight_layout()
fig.savefig("ISO_ratioPlot.pdf")
plt.draw()


fig, ax = plt.subplots(7,2,figsize=(15,30))
ax = ax.flatten()
fig, ax[0] = plotRatioFigure(hf_MC, hf_data, training_var_PID[2], 0, 11, bins=11, fig = fig, ax = ax[0])
fig, ax[1] = plotRatioFigure(hf_MC, hf_data, training_var_PID[0], 0, 6, bins=10, fig = fig, ax = ax[1])
fig, ax[2] = plotRatioFigure(hf_MC, hf_data, training_var_PID[1], 0, 5, bins=10, fig = fig, ax = ax[2])
fig, ax[3] = plotRatioFigure(hf_MC, hf_data, training_var_PID[3], 0, 3000, bins=100, fig = fig, ax = ax[3])
fig, ax[4] = plotRatioFigure(hf_MC, hf_data, training_var_PID[3], 100, 3000, bins=100, fig = fig, ax = ax[4])
fig, ax[5] = plotRatioFigure(hf_MC/1000, hf_data/1000, training_var_PID[4], 0, 100, bins=100, fig = fig, ax = ax[5])
fig, ax[6] = plotRatioFigure(hf_MC/1000, hf_data/1000, training_var_PID[4], 5, 100, bins=100, fig = fig, ax = ax[6])
fig, ax[7] = plotRatioFigure(hf_MC, hf_data, training_var_PID[5], -1000, 100, bins=100, fig = fig, ax = ax[7])
fig, ax[8] = plotRatioFigure(hf_MC, hf_data, training_var_PID[5], -0.5, 0.5, bins=100, fig = fig, ax = ax[8])
fig, ax[9] = plotRatioFigure(hf_MC, hf_data, training_var_PID[6], -2, 2, bins=100, fig = fig, ax = ax[9])
fig, ax[10] = plotRatioFigure(hf_MC, hf_data, training_var_PID[7], -2, 2, bins=100, fig = fig, ax = ax[10])
fig, ax[11] = plotRatioFigure(hf_MC, hf_data, training_var_PID[8], -0.5, 0.5, bins=100, fig = fig, ax = ax[11])
fig, ax[12] = plotRatioFigure(hf_MC, hf_data, training_var_PID[9], 2000, 6000, bins=100, fig = fig, ax = ax[12])
fig, ax[13] = plotRatioFigure(hf_MC, hf_data, training_var_PID[10], 0, 4, bins=10, fig = fig, ax = ax[13])

fig.tight_layout()
fig.savefig("PID_ratioPlot.pdf")

plt.draw()












fig, ax = plt.subplots(5,2,figsize=(15,20))
ax = ax.flatten()
fig, ax[0] = plotFigure(hf_data/1000, training_var_ISO[2], 0, 120, f"{training_var_ISO[2]}", bins=100, fig = fig, ax = ax[0])
fig, ax[1] = plotFigure(hf_data, training_var_ISO[0], -15000, 15000,  f"{training_var_ISO[0]}", bins=100, fig = fig, ax = ax[1])
fig, ax[2] = plotFigure(hf_data, training_var_ISO[1], 0, 5000, f"{training_var_ISO[1]}", bins=100,   fig = fig, ax = ax[2])
fig, ax[3] = plotFigure(hf_data, training_var_ISO[3], -1500, 10000, f"{training_var_ISO[3]}", bins=100,   fig = fig, ax = ax[3])
fig, ax[4] = plotFigure(hf_data, training_var_ISO[4], 0, 4000,  f"{training_var_ISO[4]}",  bins=100, fig = fig, ax = ax[4])
fig, ax[5] = plotFigure(hf_data, training_var_ISO[4], 100, 4000,  f"{training_var_ISO[4]} > 100", bins=100, fig = fig, ax = ax[5])
fig, ax[6] = plotFigure(hf_data, training_var_ISO[5], 0, 40000, f"{training_var_ISO[5]}", bins=100, fig = fig, ax = ax[6])
fig, ax[7] = plotFigure(hf_data, training_var_ISO[5], 500, 40000,  f"{training_var_ISO[5]} > 500", bins=100, fig = fig, ax = ax[7])
fig, ax[8] = plotFigure(hf_data, training_var_ISO[6], 0, 10000,  f"{training_var_ISO[6]}", bins=100, fig = fig, ax = ax[8])
fig, ax[9] = plotFigure(hf_data, training_var_ISO[6], 100, 10000,  f"{training_var_ISO[6]} > 100", bins=100, fig = fig, ax = ax[9])

fig.tight_layout()
fig.savefig("ISO_figurePlot.pdf")
plt.draw()


fig, ax = plt.subplots(7,2,figsize=(15,30))
ax = ax.flatten()
fig, ax[0] = plotFigure(hf_data, training_var_PID[2], 0, 11, bins=11, fig = fig, ax = ax[0])
fig, ax[1] = plotFigure(hf_data, training_var_PID[0], 0, 6, bins=10, fig = fig, ax = ax[1])
fig, ax[2] = plotFigure(hf_data, training_var_PID[1], 0, 5, bins=10, fig = fig, ax = ax[2])
fig, ax[3] = plotFigure(hf_data, training_var_PID[3], 0, 3000, bins=100, fig = fig, ax = ax[3])
fig, ax[4] = plotFigure(hf_data, training_var_PID[3], 100, 3000, bins=100, fig = fig, ax = ax[4])
fig, ax[5] = plotFigure(hf_data/1000, training_var_PID[4], 0, 100, bins=100, fig = fig, ax = ax[5])
fig, ax[6] = plotFigure(hf_data/1000, training_var_PID[4], 5, 100, bins=100, fig = fig, ax = ax[6])
fig, ax[7] = plotFigure(hf_data, training_var_PID[5], -1000, 100, bins=100, fig = fig, ax = ax[7])
fig, ax[8] = plotFigure(hf_data, training_var_PID[5], -0.5, 0.5, bins=100, fig = fig, ax = ax[8])
fig, ax[9] = plotFigure(hf_data, training_var_PID[6], -2, 2, bins=100, fig = fig, ax = ax[9])
fig, ax[10] = plotFigure(hf_data, training_var_PID[7], -2, 2, bins=100, fig = fig, ax = ax[10])
fig, ax[11] = plotFigure(hf_data, training_var_PID[8], -0.5, 0.5, bins=100, fig = fig, ax = ax[11])
fig, ax[12] = plotFigure(hf_data, training_var_PID[9], 2000, 6000, bins=100, fig = fig, ax = ax[12])
fig, ax[13] = plotFigure(hf_data, training_var_PID[10], 0, 4, bins=10, fig = fig, ax = ax[13])

fig.tight_layout()
fig.savefig("PID_varPlot.pdf")
