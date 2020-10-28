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

hf_data = h5ToDf("/Users/sda/hep/work/Data model/output/MuoSingleHdf5/090920/090920.h5")


hf_data[hf_data["Type"] == 2].index

hf_data = hf_data.drop(hf_data[hf_data["Type"] == 2].index, axis = 0)
(data['muo_PID_score'] > -20)
np.unique(hf_data["Type"], return_counts = True)

hf_data = h5ToDf("/Users/sda/hep/work/Data model/output/MuoSingleHdf5/080920_MC/080920_MC.h5")
c1 = 'b'#'g'
c2 = 'r'#'tab:purple'

fig, ax = plt.subplots(5,2,figsize=(15,20))
ax = ax.flatten()

ax[0].set_title("Distribution of muo_deltatheta_1 > -0.5")
ax[0].hist(hf_data[hf_data["Type"] == 0]["muo_deltatheta_1"], range = (-0.4,0.4), bins = 50, color = c1, label = "Background", histtype = "step")
ax[0].hist(hf_data[hf_data["Type"] == 1]["muo_deltatheta_1"], range = (-0.4,0.4), bins = 50, color = c2, label = "Signal", histtype = "step")
ax[0].set(xlabel = "muo_deltatheta_1", ylabel = "Frequency")

ax[0].text(0.98,0.78, f"muo_deltatheta_1 = -999:", color = 'k', horizontalalignment='right', transform=ax[0].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))
ax[0].text(0.98,0.73, f"{np.round(nBkg[0][0]/sum(nBkg[0])*100,2)} %, n = {int(nBkg[0][0])}", color = c1, horizontalalignment='right', transform=ax[0].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))
ax[0].text(0.98,0.68, f"{np.round(nSig[0][0]/sum(nSig[0])*100,2)} %, n = {int(nSig[0][0])}", color = c2, horizontalalignment='right', transform=ax[0].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))

ax[0].legend()


ax[1].set_title("Distribution of muo_MuonSpectrometerPt > 5 GeV")
ax[1].hist(hf_data[hf_data["Type"] == 0]["muo_MuonSpectrometerPt"]/1000, bins = 95, range = (5, 100), color = c1, label = "Background", histtype = "step")
ax[1].hist(hf_data[hf_data["Type"] == 1]["muo_MuonSpectrometerPt"]/1000, bins = 95, range = (5, 100), color = c2, label = "Signal", histtype = "step")
ax[1].set(xlabel = "muo_MuonSpectrometerPt", ylabel = "Frequency")

ax[1].text(0.98,0.78, f"pt < 5 GeV:", color = 'k', horizontalalignment='right', transform=ax[1].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))
ax[1].text(0.98,0.73, f"{np.round(sum(nBkg[0][:5])/sum(nBkg[0])*100,2)} %, n = {int(sum(nBkg[0][:5]))}", color = c1, horizontalalignment='right', transform=ax[1].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))
ax[1].text(0.98,0.68, f"{np.round(sum(nSig[0][:5])/sum(nSig[0])*100,2)} %, n = {int(sum(nSig[0][:5]))}", color = c2, horizontalalignment='right', transform=ax[1].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))


ax[1].legend()

ax[2].set_title("Distribution of muo_ptcone20 > 500")
ax[2].hist(hf_data[hf_data["Type"] == 0]["muo_ptcone20"], bins = 95, range = (500, 5000), color = c1, label = "Background", histtype = "step")
ax[2].hist(hf_data[hf_data["Type"] == 1]["muo_ptcone20"], bins = 95, range = (500, 5000), color = c2, label = "Signal", histtype = "step")
ax[2].set(xlabel = "muo_ptcone20", ylabel = "Frequency")

ax[2].text(0.98,0.78, f"muo_ptcone20 = 0", color = 'k', horizontalalignment='right', transform=ax[2].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))
ax[2].text(0.98,0.73, f"{np.round(nBkg[0][0]/sum(nBkg[0])*100,2)} %, n = {int(nBkg[0][0])}", color = c1, horizontalalignment='right', transform=ax[2].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))
ax[2].text(0.98,0.68, f"{np.round(nSig[0][0]/sum(nSig[0])*100,2)} %, n = {int(nSig[0][0])}", color = c2, horizontalalignment='right', transform=ax[2].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))


ax[2].legend()


ax[3].set_title("Distribution of muo_ET_TileCore > 100")
ax[3].hist(hf_data[hf_data["Type"] == 0]["muo_ET_TileCore"], bins = 100, range = (100, 3000), color = c1, label = "Background", histtype = "step")
ax[3].hist(hf_data[hf_data["Type"] == 1]["muo_ET_TileCore"], bins = 100, range = (100, 3000), color = c2, label = "Signal", histtype = "step")
ax[3].set(xlabel = "muo_ET_TileCore", ylabel = "Frequency")

ax[3].text(0.98,0.78, f"muo_ET_TileCore < 100", color = 'k', horizontalalignment='right', transform=ax[3].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))
ax[3].text(0.98,0.73, f"{np.round(sum(nBkg[0][:4])/sum(nBkg[0])*100,2)} %, n = {int(sum(nBkg[0][:4]))}", color = c1, horizontalalignment='right', transform=ax[3].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))
ax[3].text(0.98,0.68, f"{np.round(sum(nSig[0][:4])/sum(nSig[0])*100,2)} %, n = {int(sum(nSig[0][:4]))}", color = c2, horizontalalignment='right', transform=ax[3].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))

ax[3].legend()


ax[4].set_title("Distribution of muo_eta")
ax[4].hist(hf_data[hf_data["Type"] == 0]["muo_eta"], bins = 100, range = (-3, 3), color = c1, label = "Background", histtype = "step")
ax[4].hist(hf_data[hf_data["Type"] == 1]["muo_eta"], bins = 100, range = (-3, 3), color = c2, label = "Signal", histtype = "step")
ax[4].set(xlabel = "muo_eta", ylabel = "Frequency")
ax[4].legend()


ax[5].set_title("Distribution of invM")
ax[5].hist(hf_data[hf_data["Type"] == 0]["invM"], bins = 120, range = (0, 120), color = c1, label = "Background", histtype = "step")
ax[5].hist(hf_data[hf_data["Type"] == 1]["invM"], bins = 120, range = (0, 120), color = c2, label = "Signal", histtype = "step")
ax[5].set(xlabel = "invM", ylabel = "Frequency")
ax[5].legend()


ax[6].set_title("Distribution of muo_etcone20")
ax[6].hist(hf_data[hf_data["Type"] == 0]["muo_etcone20"], bins = 100, range = (-15000, 15000), color = c1, label = "Background", histtype = "step")
ax[6].hist(hf_data[hf_data["Type"] == 1]["muo_etcone20"], bins = 100, range = (-15000, 15000), color = c2, label = "Signal", histtype = "step")
ax[6].set(xlabel = "muo_etcone20", ylabel = "Frequency")
ax[6].legend()


ax[7].set_title("Distribution of muo_etconecoreConeEnergyCorrection")
ax[7].hist(hf_data[hf_data["Type"] == 0]["muo_etconecoreConeEnergyCorrection"], bins = 100, range = (-2000, 10000), color = c1, label = "Background", histtype = "step")
ax[7].hist(hf_data[hf_data["Type"] == 1]["muo_etconecoreConeEnergyCorrection"], bins = 100, range = (-2000, 10000), color = c2, label = "Signal", histtype = "step")
ax[7].set(xlabel = "muo_etconecoreConeEnergyCorrection", ylabel = "Frequency")
ax[7].legend()


ax[8].set_title("Distribution of muo_neflowisolcoreConeEnergyCorrection > 100")
ax[8].hist(hf_data[hf_data["Type"] == 0]["muo_neflowisolcoreConeEnergyCorrection"], bins = 100, range = (100, 4000), color = c1, label = "Background", histtype = "step")
ax[8].hist(hf_data[hf_data["Type"] == 1]["muo_neflowisolcoreConeEnergyCorrection"], bins = 100, range = (100, 4000), color = c2, label = "Signal", histtype = "step")
ax[8].set(xlabel = "muo_neflowisolcoreConeEnergyCorrection", ylabel = "Frequency")

ax[8].text(0.98,0.78, f"muo_neflowisolcoreConeEnergyCorrection < 100", color = 'k', horizontalalignment='right', transform=ax[8].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))
ax[8].text(0.98,0.73, f"{np.round(sum(nBkg[0][:2])/sum(nBkg[0])*100,2)} %, n = {int(sum(nBkg[0][:2]))}", color = c1, horizontalalignment='right', transform=ax[8].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))
ax[8].text(0.98,0.68, f"{np.round(sum(nSig[0][:2])/sum(nSig[0])*100,2)} %, n = {int(sum(nSig[0][:2]))}", color = c2, horizontalalignment='right', transform=ax[8].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))


ax[8].legend()



ax[9].set_title("Distribution of muo_ptconecoreTrackPtrCorrection > 500")
ax[9].hist(hf_data[hf_data["Type"] == 0]["muo_ptconecoreTrackPtrCorrection"], bins = 100, range = (500, 40000), color = c1, label = "Background", histtype = "step")
ax[9].hist(hf_data[hf_data["Type"] == 1]["muo_ptconecoreTrackPtrCorrection"], bins = 100, range = (500, 40000), color = c2, label = "Signal", histtype = "step")
ax[9].set(xlabel = "muo_ptconecoreTrackPtrCorrection", ylabel = "Frequency")

ax[9].text(0.98,0.78, f"muo_ptconecoreTrackPtrCorrection < 500", color = 'k', horizontalalignment='right', transform=ax[9].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))
ax[9].text(0.98,0.73, f"{np.round(nBkg[0][0]/sum(nBkg[0])*100,2)} %, n = {int(nBkg[0][0])}", color = c1, horizontalalignment='right', transform=ax[9].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))
ax[9].text(0.98,0.68, f"{np.round(nSig[0][0]/sum(nSig[0])*100,2)} %, n = {int(nSig[0][0])}", color = c2, horizontalalignment='right', transform=ax[9].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))

ax[9].legend()

fig.tight_layout()
fig.savefig("Distribution_vars_MC090929_1.pdf")


fig, ax = plt.subplots(5,2,figsize=(15,20))
ax = ax.flatten()

ax[0].set_title("Distribution of muo_topoetconecoreConeEnergyCorrection > 500")
ax[0].hist(hf_data[hf_data["Type"] == 0]["muo_topoetconecoreConeEnergyCorrection"], bins = 100, range = (500, 10000), color = c1, label = "Background", histtype = "step")
ax[0].hist(hf_data[hf_data["Type"] == 1]["muo_topoetconecoreConeEnergyCorrection"], bins = 100, range = (500, 10000), color = c2, label = "Signal", histtype = "step")
ax[0].set(xlabel = "muo_topoetconecoreConeEnergyCorrection", ylabel = "Frequency")

ax[0].text(0.98,0.78, f"muo_topoetconecoreConeEnergyCorrection < 500", color = 'k', horizontalalignment='right', transform=ax[0].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))
ax[0].text(0.98,0.73, f"{np.round(sum(nBkg[0][:4])/sum(nBkg[0])*100,2)} %, n = {int(sum(nBkg[0][:4]))}", color = c1, horizontalalignment='right', transform=ax[0].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))
ax[0].text(0.98,0.68, f"{np.round(sum(nSig[0][:4])/sum(nSig[0])*100,2)} %, n = {int(sum(nSig[0][:4]))}", color = c2, horizontalalignment='right', transform=ax[0].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))

ax[0].legend()

ax[1].set_title("Distribution of muo_pt > 4.5 GeV")
ax[1].hist(hf_data[hf_data["Type"] == 0]["muo_pt"]/1000, bins = 100, range = (4.5, 120), color = c1, label = "Background", histtype = "step")
ax[1].hist(hf_data[hf_data["Type"] == 1]["muo_pt"]/1000, bins = 100, range = (4.5, 120), color = c2, label = "Signal", histtype = "step")
ax[1].set(xlabel = "muo_pt", ylabel = "Frequency")

ax[1].text(0.98,0.78, f"muo_pt < 4.5", color = 'k', horizontalalignment='right', transform=ax[1].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))
ax[1].text(0.98,0.73, f"{np.round(sum(nBkg[0][:4])/sum(nBkg[0])*100,2)} %, n = {int(sum(nBkg[0][:4]))}", color = c1, horizontalalignment='right', transform=ax[1].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))
ax[1].text(0.98,0.68, f"{np.round(sum(nSig[0][:4])/sum(nSig[0])*100,2)} %, n = {int(sum(nSig[0][:4]))}", color = c2, horizontalalignment='right', transform=ax[1].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))

ax[1].legend()


ax[2].set_title("Distribution of muo_scatteringCurvatureSignificance")
ax[2].hist(hf_data[hf_data["Type"] == 0]["muo_scatteringCurvatureSignificance"], bins = 100, range = (-2, 2), color = c1, label = "Background", histtype = "step")
ax[2].hist(hf_data[hf_data["Type"] == 1]["muo_scatteringCurvatureSignificance"], bins = 100, range = (-2, 2), color = c2, label = "Signal", histtype = "step")
ax[2].set(xlabel = "muo_scatteringCurvatureSignificance", ylabel = "Frequency")

ax[2].legend()


ax[3].set_title("Distribution of muo_scatteringNeighbourSignificance")
ax[3].hist(hf_data[hf_data["Type"] == 0]["muo_scatteringNeighbourSignificance"], bins = 100, range = (-2, 2), color = c1, label = "Background", histtype = "step")
ax[3].hist(hf_data[hf_data["Type"] == 1]["muo_scatteringNeighbourSignificance"], bins = 100, range = (-2, 2), color = c2, label = "Signal", histtype = "step")
ax[3].set(xlabel = "muo_scatteringNeighbourSignificance", ylabel = "Frequency")

ax[3].legend()

ax[4].set_title("Distribution of muo_momentumBalanceSignificance")
ax[4].hist(hf_data[hf_data["Type"] == 0]["muo_momentumBalanceSignificance"], bins = 100, range = (-0.4, 0.4), color = c1, label = "Background", histtype = "step")
ax[4].hist(hf_data[hf_data["Type"] == 1]["muo_momentumBalanceSignificance"], bins = 100, range = (-0.4, 0.4), color = c2, label = "Signal", histtype = "step")
ax[4].set(xlabel = "muo_momentumBalanceSignificance", ylabel = "Frequency")

ax[4].legend()


ax[5].set_title("Distribution of muo_EnergyLoss")
ax[5].hist(hf_data[hf_data["Type"] == 0]["muo_EnergyLoss"], bins = 100, range = (1500, 6000), color = c1, label = "Background", histtype = "step")
ax[5].hist(hf_data[hf_data["Type"] == 1]["muo_EnergyLoss"], bins = 100, range = (1500, 6000), color = c2, label = "Signal", histtype = "step")
ax[5].set(xlabel = "muo_EnergyLoss", ylabel = "Frequency")

ax[5].legend()


ax[6].set_title("Distribution of muo_energyLossType")
ax[6].hist(hf_data[hf_data["Type"] == 0]["muo_energyLossType"], bins = 10, range = (0, 3), color = c1, label = "Background", histtype = "step")
ax[6].hist(hf_data[hf_data["Type"] == 1]["muo_energyLossType"], bins = 10, range = (0, 3), color = c2, label = "Signal", histtype = "step")
ax[6].set(xlabel = "muo_energyLossType", ylabel = "Frequency")

ax[6].legend()

ax[7].set_title("Distribution of muo_quality")
ax[7].hist(hf_data[hf_data["Type"] == 0]["muo_quality"], bins = 20, range = (0, 11), color = c1, label = "Background", histtype = "step")
ax[7].hist(hf_data[hf_data["Type"] == 1]["muo_quality"], bins = 20, range = (0, 11), color = c2, label = "Signal", histtype = "step")
ax[7].set(xlabel = "muo_quality", ylabel = "Frequency")

ax[7].legend()


ax[8].set_title("Distribution of muo_numberOfPrecisionLayers")
ax[8].hist(hf_data[hf_data["Type"] == 0]["muo_numberOfPrecisionLayers"], bins = 10, range = (0, 5), color = c1, label = "Background", histtype = "step")
ax[8].hist(hf_data[hf_data["Type"] == 1]["muo_numberOfPrecisionLayers"], bins = 10, range = (0, 5), color = c2, label = "Signal", histtype = "step")
ax[8].set(xlabel = "muo_numberOfPrecisionLayers", ylabel = "Frequency")

ax[8].legend()

ax[9].set_title("Distribution of muo_numberOfPrecisionHoleLayers")
ax[9].hist(hf_data[hf_data["Type"] == 0]["muo_numberOfPrecisionHoleLayers"], bins = 10, range = (0, 3), color = c1, label = "Background", histtype = "step")
ax[9].hist(hf_data[hf_data["Type"] == 1]["muo_numberOfPrecisionHoleLayers"], bins = 10, range = (0, 3), color = c2, label = "Signal", histtype = "step")
ax[9].set(xlabel = "muo_numberOfPrecisionHoleLayers", ylabel = "Frequency")

ax[9].legend()

fig.tight_layout()
fig.savefig("Distribution_vars_MC090929_2.pdf")

########################
######## Ratio #########
########################

hf_MC = h5ToDf("/Users/sda/hep/work/Data model/output/MuoSingleHdf5/080920_MC/080920_MC.h5")



fig, ax = plt.subplots(1,1,figsize=(7,2))
#ax = ax.flatten()
ax.set_title("Distribution of muo_deltatheta_1")
nBkgData, bin_edges = np.histogram(hf_data[hf_data["Type"] == 0]["muo_deltatheta_1"], bins = 20)
bins = ((bin_edges[1:] + bin_edges[:-1])/2)[:]
nBkgMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 0]["muo_deltatheta_1"], bins = 20)
ratio_bkg = np.nan_to_num(nBkgData/nBkgMC)

nSigData, _ = np.histogram(hf_data[hf_data["Type"] == 1]["muo_deltatheta_1"], bins = 20)
nSigMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 1]["muo_deltatheta_1"], bins = 20)
ratio_sig = np.nan_to_num(nSigData/nSigMC)

ax.plot(bins, ratio_bkg, '.', color = c1)
ax.plot(bins, ratio_sig, '.', color = c2)
ax.set(ylim = (-0.2,1))
fig.show()



fig, ax = plt.subplots(10,2,figsize=(15,20))
ax = ax.flatten()
ax[0].set_title("Distribution of muo_deltatheta_1")
nBkgData, bin_edges = np.histogram(hf_data[hf_data["Type"] == 0]["muo_deltatheta_1"], bins = 20, range = (-1000, 2))
bins = ((bin_edges[1:] + bin_edges[:-1])/2)[:]
nBkgMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 0]["muo_deltatheta_1"], bins = 20, range = (-1000, 2))
ratio_bkg = np.nan_to_num(nBkgData/nBkgMC)

nSigData, _ = np.histogram(hf_data[hf_data["Type"] == 1]["muo_deltatheta_1"], bins = 20, range = (-1000, 2))
nSigMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 1]["muo_deltatheta_1"], bins = 20, range = (-1000, 2))
ratio_sig = np.nan_to_num(nSigData/nSigMC)

ax[0].plot(bins, ratio_bkg, '.', color = c1)
ax[0].plot(bins, ratio_sig, '.', color = c2)
ax[0].set(ylim = (-0.2,1))
ax[0].axhline(0, color = 'k', linestyle = "dashed")
ax[0].set(xlabel = "muo_deltatheta_1", ylabel = "Ratio")



ax[1].set_title("Distribution of muo_MuonSpectrometerPt")
nBkgData, bin_edges = np.histogram(hf_data[hf_data["Type"] == 0]["muo_MuonSpectrometerPt"]/1000, bins = 100, range = (0,100))
bins = ((bin_edges[1:] + bin_edges[:-1])/2)[:]
nBkgMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 0]["muo_MuonSpectrometerPt"]/1000, bins = 100, range = (0,100))
ratio_bkg = np.nan_to_num(nBkgData/nBkgMC)

nSigData, _ = np.histogram(hf_data[hf_data["Type"] == 1]["muo_MuonSpectrometerPt"]/1000, bins = 100, range = (0,100))
nSigMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 1]["muo_MuonSpectrometerPt"]/1000, bins = 100, range = (0,100))
ratio_sig = np.nan_to_num(nSigData/nSigMC)

ax[1].plot(bins, ratio_bkg, '.', color = c1)
ax[1].plot(bins, ratio_sig, '.', color = c2)
ax[1].set(ylim = (-0.2,1))
ax[1].axhline(0, color = 'k', linestyle = "dashed")
ax[1].set(xlabel = "muo_MuonSpectrometerPt", ylabel = "Ratio")

ax[2].set_title("Distribution of muo_ptcone20")
nBkgData, bin_edges = np.histogram(hf_data[hf_data["Type"] == 0]["muo_ptcone20"], bins = 100, range = (0,5000))
bins = ((bin_edges[1:] + bin_edges[:-1])/2)[:]
nBkgMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 0]["muo_ptcone20"], bins = 100, range = (0,5000))
ratio_bkg = np.nan_to_num(nBkgData/nBkgMC)

nSigData, _ = np.histogram(hf_data[hf_data["Type"] == 1]["muo_ptcone20"], bins = 100, range = (0,5000))
nSigMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 1]["muo_ptcone20"], bins = 100, range = (0,5000))
ratio_sig = np.nan_to_num(nSigData/nSigMC)

ax[2].plot(bins, ratio_bkg, '.', color = c1)
ax[2].plot(bins, ratio_sig, '.', color = c2)
ax[2].set(ylim = (-0.2,1))
ax[2].axhline(0, color = 'k', linestyle = "dashed")
ax[2].set(xlabel = "muo_ptcone20", ylabel = "Ratio")



ax[3].set_title("Distribution of muo_ET_TileCore")
nBkgData, bin_edges = np.histogram(hf_data[hf_data["Type"] == 0]["muo_ET_TileCore"], bins = 100, range = (0,3000))
bins = ((bin_edges[1:] + bin_edges[:-1])/2)[:]
nBkgMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 0]["muo_ET_TileCore"], bins = 100, range = (0,3000))
ratio_bkg = np.nan_to_num(nBkgData/nBkgMC)

nSigData, _ = np.histogram(hf_data[hf_data["Type"] == 1]["muo_ET_TileCore"], bins = 100, range = (0,3000))
nSigMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 1]["muo_ET_TileCore"], bins = 100, range = (0,3000))
ratio_sig = np.nan_to_num(nSigData/nSigMC)

ax[3].plot(bins, ratio_bkg, '.', color = c1)
ax[3].plot(bins, ratio_sig, '.', color = c2)
ax[3].set(ylim = (-0.2,1))
ax[3].axhline(0, color = 'k', linestyle = "dashed")
ax[3].set(xlabel = "muo_ET_TileCore", ylabel = "Ratio")
#
ax[4].set_title("Distribution of muo_eta")
nBkgData, bin_edges = np.histogram(hf_data[hf_data["Type"] == 0]["muo_eta"], bins = 100, range = (-3,3))
bins = ((bin_edges[1:] + bin_edges[:-1])/2)[:]
nBkgMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 0]["muo_eta"], bins = 100, range = (-3,3))
ratio_bkg = np.nan_to_num(nBkgData/nBkgMC)

nSigData, _ = np.histogram(hf_data[hf_data["Type"] == 1]["muo_eta"], bins = 100, range = (-3,3))
nSigMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 1]["muo_eta"], bins = 100, range = (-3,3))
ratio_sig = np.nan_to_num(nSigData/nSigMC)

ax[4].plot(bins, ratio_bkg, '.', color = c1)
ax[4].plot(bins, ratio_sig, '.', color = c2)
ax[4].set(ylim = (-0.2,1))
ax[4].axhline(0, color = 'k', linestyle = "dashed")
ax[4].set(xlabel = "muo_eta", ylabel = "Ratio")


ax[5].set_title("Distribution of invM")
nBkgData, bin_edges = np.histogram(hf_data[hf_data["Type"] == 0]["invM"], bins = 120, range = (0,120))
bins = ((bin_edges[1:] + bin_edges[:-1])/2)[:]
nBkgMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 0]["invM"], bins = 120, range = (0,120))
ratio_bkg = np.nan_to_num(nBkgData/nBkgMC)

nSigData, _ = np.histogram(hf_data[hf_data["Type"] == 1]["invM"], bins = 120, range = (0,120))
nSigMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 1]["invM"], bins = 120, range = (0,120))
ratio_sig = np.nan_to_num(nSigData/nSigMC)

ax[5].plot(bins, ratio_bkg, '.', color = c1)
ax[5].plot(bins, ratio_sig, '.', color = c2)
ax[5].set(ylim = (-0.2,1))
ax[5].axhline(0, color = 'k', linestyle = "dashed")
ax[5].set(xlabel = "invM", ylabel = "Ratio")



ax[6].set_title("Distribution of muo_etcone20")
nBkgData, bin_edges = np.histogram(hf_data[hf_data["Type"] == 0]["muo_etcone20"], bins = 100, range = (-15000,15000))
bins = ((bin_edges[1:] + bin_edges[:-1])/2)[:]
nBkgMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 0]["muo_etcone20"], bins = 100, range = (-15000,15000))
ratio_bkg = np.nan_to_num(nBkgData/nBkgMC)

nSigData, _ = np.histogram(hf_data[hf_data["Type"] == 1]["muo_etcone20"], bins = 100, range = (-15000,15000))
nSigMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 1]["muo_etcone20"], bins = 100, range = (-15000,15000))
ratio_sig = np.nan_to_num(nSigData/nSigMC)

ax[6].plot(bins, ratio_bkg, '.', color = c1)
ax[6].plot(bins, ratio_sig, '.', color = c2)
ax[6].set(ylim = (-0.2,1))
ax[6].axhline(0, color = 'k', linestyle = "dashed")
ax[6].set(xlabel = "muo_etcone20", ylabel = "Ratio")


ax[7].set_title("Distribution of muo_etconecoreConeEnergyCorrection")
nBkgData, bin_edges = np.histogram(hf_data[hf_data["Type"] == 0]["muo_etconecoreConeEnergyCorrection"], bins = 100, range = (-2000,10000))
bins = ((bin_edges[1:] + bin_edges[:-1])/2)[:]
nBkgMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 0]["muo_etconecoreConeEnergyCorrection"], bins = 100, range = (-2000,10000))
ratio_bkg = np.nan_to_num(nBkgData/nBkgMC)

nSigData, _ = np.histogram(hf_data[hf_data["Type"] == 1]["muo_etconecoreConeEnergyCorrection"], bins = 100, range = (-2000,10000))
nSigMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 1]["muo_etconecoreConeEnergyCorrection"], bins = 100, range = (-2000,10000))
ratio_sig = np.nan_to_num(nSigData/nSigMC)

ax[7].plot(bins, ratio_bkg, '.', color = c1)
ax[7].plot(bins, ratio_sig, '.', color = c2)
ax[7].set(ylim = (-0.2,1))
ax[7].axhline(0, color = 'k', linestyle = "dashed")
ax[7].set(xlabel = "muo_etconecoreConeEnergyCorrection", ylabel = "Ratio")


ax[8].set_title("Distribution of muo_neflowisolcoreConeEnergyCorrection")
nBkgData, bin_edges = np.histogram(hf_data[hf_data["Type"] == 0]["muo_neflowisolcoreConeEnergyCorrection"], bins = 100, range = (0,4000))
bins = ((bin_edges[1:] + bin_edges[:-1])/2)[:]
nBkgMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 0]["muo_neflowisolcoreConeEnergyCorrection"], bins = 100, range = (0,4000))
ratio_bkg = np.nan_to_num(nBkgData/nBkgMC)

nSigData, _ = np.histogram(hf_data[hf_data["Type"] == 1]["muo_neflowisolcoreConeEnergyCorrection"], bins = 100, range = (0,4000))
nSigMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 1]["muo_neflowisolcoreConeEnergyCorrection"], bins = 100, range = (0,4000))
ratio_sig = np.nan_to_num(nSigData/nSigMC)

ax[8].plot(bins, ratio_bkg, '.', color = c1)
ax[8].plot(bins, ratio_sig, '.', color = c2)
ax[8].set(ylim = (-0.2,1))
ax[8].axhline(0, color = 'k', linestyle = "dashed")
ax[8].set(xlabel = "muo_neflowisolcoreConeEnergyCorrection", ylabel = "Ratio")


ax[9].set_title("Distribution of muo_ptconecoreTrackPtrCorrection")
nBkgData, bin_edges = np.histogram(hf_data[hf_data["Type"] == 0]["muo_ptconecoreTrackPtrCorrection"], bins = 100, range = (0,40000))
bins = ((bin_edges[1:] + bin_edges[:-1])/2)[:]
nBkgMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 0]["muo_ptconecoreTrackPtrCorrection"], bins = 100, range = (0,40000))
ratio_bkg = np.nan_to_num(nBkgData/nBkgMC)

nSigData, _ = np.histogram(hf_data[hf_data["Type"] == 1]["muo_ptconecoreTrackPtrCorrection"], bins = 100, range = (0,40000))
nSigMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 1]["muo_ptconecoreTrackPtrCorrection"], bins = 100, range = (0,40000))
ratio_sig = np.nan_to_num(nSigData/nSigMC)

ax[9].plot(bins, ratio_bkg, '.', color = c1)
ax[9].plot(bins, ratio_sig, '.', color = c2)
ax[9].set(ylim = (-0.2,1))
ax[9].axhline(0, color = 'k', linestyle = "dashed")
ax[9].set(xlabel = "muo_ptconecoreTrackPtrCorrection", ylabel = "Ratio")


ax[10].set_title("Distribution of muo_topoetconecoreConeEnergyCorrection")
nBkgData, bin_edges = np.histogram(hf_data[hf_data["Type"] == 0]["muo_topoetconecoreConeEnergyCorrection"], bins = 100, range = (0,10000))
bins = ((bin_edges[1:] + bin_edges[:-1])/2)[:]
nBkgMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 0]["muo_topoetconecoreConeEnergyCorrection"], bins = 100, range = (0,10000))
ratio_bkg = np.nan_to_num(nBkgData/nBkgMC)

nSigData, _ = np.histogram(hf_data[hf_data["Type"] == 1]["muo_topoetconecoreConeEnergyCorrection"], bins = 100, range = (0,10000))
nSigMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 1]["muo_topoetconecoreConeEnergyCorrection"], bins = 100, range = (0,10000))
ratio_sig = np.nan_to_num(nSigData/nSigMC)

ax[10].plot(bins, ratio_bkg, '.', color = c1)
ax[10].plot(bins, ratio_sig, '.', color = c2)
ax[10].set(ylim = (-0.2,1))
ax[10].axhline(0, color = 'k', linestyle = "dashed")
ax[10].set(xlabel = "muo_topoetconecoreConeEnergyCorrection", ylabel = "Ratio")



ax[11].set_title("Distribution of muo_pt")
nBkgData, bin_edges = np.histogram(hf_data[hf_data["Type"] == 0]["muo_pt"]/1000, bins = 100, range = (0,120))
bins = ((bin_edges[1:] + bin_edges[:-1])/2)[:]
nBkgMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 0]["muo_pt"]/1000, bins = 100, range = (0,120))
ratio_bkg = np.nan_to_num(nBkgData/nBkgMC)

nSigData, _ = np.histogram(hf_data[hf_data["Type"] == 1]["muo_pt"]/1000, bins = 100, range = (0,120))
nSigMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 1]["muo_pt"]/1000, bins = 100, range = (0,120))
ratio_sig = np.nan_to_num(nSigData/nSigMC)

ax[11].plot(bins, ratio_bkg, '.', color = c1)
ax[11].plot(bins, ratio_sig, '.', color = c2)
ax[11].set(ylim = (-0.2,1))
ax[11].axhline(0, color = 'k', linestyle = "dashed")
ax[11].set(xlabel = "muo_pt", ylabel = "Ratio")


ax[12].set_title("Distribution of muo_scatteringCurvatureSignificance")
nBkgData, bin_edges = np.histogram(hf_data[hf_data["Type"] == 0]["muo_scatteringCurvatureSignificance"], bins = 100, range = (-2,2))
bins = ((bin_edges[1:] + bin_edges[:-1])/2)[:]
nBkgMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 0]["muo_scatteringCurvatureSignificance"], bins = 100, range = (-2,2))
ratio_bkg = np.nan_to_num(nBkgData/nBkgMC)

nSigData, _ = np.histogram(hf_data[hf_data["Type"] == 1]["muo_scatteringCurvatureSignificance"], bins = 100, range = (-2,2))
nSigMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 1]["muo_scatteringCurvatureSignificance"], bins = 100, range = (-2,2))
ratio_sig = np.nan_to_num(nSigData/nSigMC)

ax[12].plot(bins, ratio_bkg, '.', color = c1)
ax[12].plot(bins, ratio_sig, '.', color = c2)
ax[12].set(ylim = (-0.2,1))
ax[12].axhline(0, color = 'k', linestyle = "dashed")
ax[12].set(xlabel = "muo_scatteringCurvatureSignificance", ylabel = "Ratio")


ax[13].set_title("Distribution of muo_scatteringNeighbourSignificance")
nBkgData, bin_edges = np.histogram(hf_data[hf_data["Type"] == 0]["muo_scatteringNeighbourSignificance"], bins = 100, range = (-2,2))
bins = ((bin_edges[1:] + bin_edges[:-1])/2)[:]
nBkgMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 0]["muo_scatteringNeighbourSignificance"], bins = 100, range = (-2,2))
ratio_bkg = np.nan_to_num(nBkgData/nBkgMC)

nSigData, _ = np.histogram(hf_data[hf_data["Type"] == 1]["muo_scatteringNeighbourSignificance"], bins = 100, range = (-2,2))
nSigMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 1]["muo_scatteringNeighbourSignificance"], bins = 100, range = (-2,2))
ratio_sig = np.nan_to_num(nSigData/nSigMC)

ax[13].plot(bins, ratio_bkg, '.', color = c1)
ax[13].plot(bins, ratio_sig, '.', color = c2)
ax[13].set(ylim = (-0.2,1))
ax[13].axhline(0, color = 'k', linestyle = "dashed")
ax[13].set(xlabel = "muo_scatteringNeighbourSignificance", ylabel = "Ratio")


ax[14].set_title("Distribution of muo_momentumBalanceSignificance")
nBkgData, bin_edges = np.histogram(hf_data[hf_data["Type"] == 0]["muo_momentumBalanceSignificance"], bins = 100, range = (-0.4,0.4))
bins = ((bin_edges[1:] + bin_edges[:-1])/2)[:]
nBkgMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 0]["muo_momentumBalanceSignificance"], bins = 100, range = (-0.4,0.4))
ratio_bkg = np.nan_to_num(nBkgData/nBkgMC)

nSigData, _ = np.histogram(hf_data[hf_data["Type"] == 1]["muo_momentumBalanceSignificance"], bins = 100, range = (-0.4,0.4))
nSigMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 1]["muo_momentumBalanceSignificance"], bins = 100, range = (-0.4,0.4))
ratio_sig = np.nan_to_num(nSigData/nSigMC)

ax[14].plot(bins, ratio_bkg, '.', color = c1)
ax[14].plot(bins, ratio_sig, '.', color = c2)
ax[14].set(ylim = (-0.2,1))
ax[14].axhline(0, color = 'k', linestyle = "dashed")
ax[14].set(xlabel = "muo_momentumBalanceSignificance", ylabel = "Ratio")



ax[15].set_title("Distribution of muo_EnergyLoss")
nBkgData, bin_edges = np.histogram(hf_data[hf_data["Type"] == 0]["muo_EnergyLoss"], bins = 100, range = (1500,6000))
bins = ((bin_edges[1:] + bin_edges[:-1])/2)[:]
nBkgMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 0]["muo_EnergyLoss"], bins = 100, range = (1500,6000))
ratio_bkg = np.nan_to_num(nBkgData/nBkgMC)

nSigData, _ = np.histogram(hf_data[hf_data["Type"] == 1]["muo_EnergyLoss"], bins = 100, range = (1500,6000))
nSigMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 1]["muo_EnergyLoss"], bins = 100, range = (1500,6000))
ratio_sig = np.nan_to_num(nSigData/nSigMC)

ax[15].plot(bins, ratio_bkg, '.', color = c1)
ax[15].plot(bins, ratio_sig, '.', color = c2)
ax[15].set(ylim = (-0.2,1))
ax[15].axhline(0, color = 'k', linestyle = "dashed")

ax[15].set(xlabel = "muo_EnergyLoss", ylabel = "Ratio")



ax[16].set_title("Distribution of muo_energyLossType")
nBkgData, bin_edges = np.histogram(hf_data[hf_data["Type"] == 0]["muo_energyLossType"], bins = 10)
bins = ((bin_edges[1:] + bin_edges[:-1])/2)[:]
nBkgMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 0]["muo_energyLossType"], bins = 10)
ratio_bkg = np.nan_to_num(nBkgData/nBkgMC)

nSigData, _ = np.histogram(hf_data[hf_data["Type"] == 1]["muo_energyLossType"], bins = 10)
nSigMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 1]["muo_energyLossType"], bins = 10)
ratio_sig = np.nan_to_num(nSigData/nSigMC)

ax[16].plot(bins, ratio_bkg, '.', color = c1)
ax[16].plot(bins, ratio_sig, '.', color = c2)
ax[16].set(ylim = (-0.2,1))
ax[16].axhline(0, color = 'k', linestyle = "dashed")

ax[16].set(xlabel = "muo_EnergyLoss", ylabel = "Ratio")


ax[17].set_title("Distribution of muo_quality")
nBkgData, bin_edges = np.histogram(hf_data[hf_data["Type"] == 0]["muo_quality"], bins = 20)
bins = ((bin_edges[1:] + bin_edges[:-1])/2)[:]
nBkgMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 0]["muo_quality"], bins = 20)
ratio_bkg = np.nan_to_num(nBkgData/nBkgMC)

nSigData, _ = np.histogram(hf_data[hf_data["Type"] == 1]["muo_quality"], bins = 20)
nSigMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 1]["muo_quality"], bins = 20)
ratio_sig = np.nan_to_num(nSigData/nSigMC)

ax[17].plot(bins, ratio_bkg, '.', color = c1)
ax[17].plot(bins, ratio_sig, '.', color = c2)
ax[17].set(ylim = (-0.2,1))
ax[17].axhline(0, color = 'k', linestyle = "dashed")

ax[17].set(xlabel = "muo_quality", ylabel = "Ratio")



ax[18].set_title("Distribution of muo_numberOfPrecisionLayers")
nBkgData, bin_edges = np.histogram(hf_data[hf_data["Type"] == 0]["muo_numberOfPrecisionLayers"], bins = 10)
bins = ((bin_edges[1:] + bin_edges[:-1])/2)[:]
nBkgMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 0]["muo_numberOfPrecisionLayers"], bins = 10)
ratio_bkg = np.nan_to_num(nBkgData/nBkgMC)

nSigData, _ = np.histogram(hf_data[hf_data["Type"] == 1]["muo_numberOfPrecisionLayers"], bins = 10)
nSigMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 1]["muo_numberOfPrecisionLayers"], bins = 10)
ratio_sig = np.nan_to_num(nSigData/nSigMC)

ax[18].plot(bins, ratio_bkg, '.', color = c1)
ax[18].plot(bins, ratio_sig, '.', color = c2)
ax[18].set(ylim = (-0.2,1))
ax[18].axhline(0, color = 'k', linestyle = "dashed")

ax[18].set(xlabel = "muo_numberOfPrecisionLayers", ylabel = "Ratio")


ax[19].set_title("Distribution of muo_numberOfPrecisionHoleLayers")
nBkgData, bin_edges = np.histogram(hf_data[hf_data["Type"] == 0]["muo_numberOfPrecisionHoleLayers"], bins = 10)
bins = ((bin_edges[1:] + bin_edges[:-1])/2)[:]
nBkgMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 0]["muo_numberOfPrecisionHoleLayers"], bins = 10)
ratio_bkg = np.nan_to_num(nBkgData/nBkgMC)

nSigData, _ = np.histogram(hf_data[hf_data["Type"] == 1]["muo_numberOfPrecisionHoleLayers"], bins = 10)
nSigMC, _ = np.histogram(hf_MC[hf_MC["Type"] == 1]["muo_numberOfPrecisionHoleLayers"], bins = 10)
ratio_sig = np.nan_to_num(nSigData/nSigMC)

ax[19].plot(bins, ratio_bkg, '.', color = c1)
ax[19].plot(bins, ratio_sig, '.', color = c2)
ax[19].set(ylim = (-0.2,1))
ax[19].axhline(0, color = 'k', linestyle = "dashed")
ax[19].set(xlabel = "muo_numberOfPrecisionHoleLayers", ylabel = "Ratio")

fig.tight_layout()
fig.savefig("Distribution_Ratio090929.pdf")
