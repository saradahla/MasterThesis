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

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

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
from minepy import MINE
def calc_mic_score(x, y):
    mine = MINE(est="mic_e")
    # mine = MINE(alpha=9, c=5, est="mic_e")
    mine.compute_score(x, y)
    return mine.mic()

def Predict(gbm, data, training_var):
    score = gbm.predict(data[training_var], n_jobs=1)
    return logit(score)
#
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

modelISO = "/Users/sda/hep/work/Zmm model/PID_ISO_models/output/ISOModels/110820_ZbbW/lgbmISO.txt"
modelPID = "/Users/sda/hep/work/Zmm model/PID_ISO_models/output/PIDModels/010920_ZbbW/lgbmPID.txt"
modelPID_onlyATLAS = "/Users/sda/hep/work/Zmm model/PID_ISO_models/output/PIDModels/010920_ZbbW_only6/lgbmPID.txt"

PIDmod = lgb.Booster(model_file = modelPID)
PIDmod_onlyATLAS = lgb.Booster(model_file = modelPID_onlyATLAS)
ISOmod = lgb.Booster(model_file = modelISO)




hf_data = h5ToDf("/Users/sda/hep/work/Data model/output/MuoSingleHdf5/010920_3/010920_3.h5")


#hf_data = h5ToDf("/Users/sda/hep/work/Data model/output/MuoSingleHdf5/100920_Loose/100920_Loose.h5")
print(f"Shape of data before cutting values with pt < 4.5: {hf_data.shape}")
mask_pt = (hf_data["muo_pt"]/1000 > 4.5) #&  (hf_data["muo_pt"]/1000 < 15)
hf_data = hf_data[mask_pt]
print(f"Shape of data after cutting values with pt < 4.5 and -999: {hf_data.shape}")
type_data = hf_data["Type"]
#hf_MC = h5ToDf("/Users/sda/hep/work/Data model/output/MuoSingleHdf5/080920_MC/080920_MC.h5")
hf_MC = h5ToDf("/Users/sda/hep/work/Data model/output/MuoSingleHdf5/100920_MC_Loose/100920_MC_Loose.h5")

hf_MC_truth = hf_MC.copy()#h5ToDf("/Users/sda/hep/work/Zmm model/PID_ISO_models/output/PIDReweightFiles/010920_ZbbW/combined_010920_train.h5")
hf_MC_truth["Type"] = (hf_MC_truth["muo_truthOrigin"] == 13) #& (hf_MC_truth["muo_truthPdgId"] == 13) #&
#hf_MC_truth = hf_MC_truth[(hf_MC_truth["dataset"] == 1)]
print(f"Shape of MC Truth before cutting values with pt < 4.5: {hf_MC_truth.shape}")
mask_pt_MC_truth = hf_MC_truth["muo_pt"]/1000 > 4.5
hf_MC_truth = hf_MC_truth[mask_pt_MC_truth]
print(f"Shape of MC Truth after cutting values with pt < 4.5: {hf_MC_truth.shape}")
type_MC_truth = hf_MC_truth["Type"]

# fig, ax = plt.subplots(figsize=(5,5))
# ax.hist(hf_MC_truth["muo_pt"][type_MC_truth == 0]/1000, color = 'b', histtype = "step", bins = 100, range = (0,120), label = "Background");
# ax.hist(hf_MC_truth["muo_pt"][type_MC_truth == 1]/1000, color = 'r', histtype = "step", bins = 100, range = (0,120), label = "Signal");
# ax.set(xlabel = "pt", ylabel = "Frequency")
# ax.legend()
# fig.tight_layout()
# fig.savefig("PtDistMCTruth.pdf")

#hf_MC = h5ToDf("/Users/sda/hep/work/Data model/output/MuoSingleHdf5/080920_MC/080920_MC.h5")
hf_MC = h5ToDf("/Users/sda/hep/work/Data model/output/MuoSingleHdf5/100920_MC_Loose/100920_MC_Loose.h5")

print(f"Shape of MC before cutting values with pt < 4.5: {hf_MC.shape}")
mask_999_MC = hf_MC["muo_deltatheta_1"] == -999
mask_pt_MC = hf_MC["muo_pt"]/1000 > 4.5
hf_MC = hf_MC[mask_pt_MC]# & ~mask_999_MC]
print(f"Shape of MC after cutting values with pt < 4.5 and -999: {hf_MC.shape}")
type_MC = hf_MC["Type"]

# np.unique(hf_MC_truth["Type"] == hf_MC["Type"], return_counts = True)
def GetScores(hf, type, label, color = ["b", "r"], plot = False):
    hf['muo_ISO_score'] = 0
    hf['muo_PID_score'] = 0
    hf['muo_PID_score_ATLAS'] = 0

    hf['muo_ISO_score'] = Predict(ISOmod, hf, training_var_ISO)
    hf['muo_PID_score'] = Predict(PIDmod, hf, training_var_PID)
    hf['muo_PID_score_ATLAS'] = Predict(PIDmod_onlyATLAS, hf, training_var_PIDATLAS)

    nISO_cut = len(hf['muo_ISO_score'][(hf['muo_ISO_score'] > 4) & (hf['muo_ISO_score'] < -4) ])
    nISO_cut_Sig = len(hf['muo_ISO_score'][(hf['muo_ISO_score'] > 4) & (hf['muo_ISO_score'] < -4) ][type == 1])
    nPID_cut = len(hf['muo_PID_score'][(hf['muo_PID_score'] > 20) & (hf['muo_PID_score'] < -20) ])
    nPID_cut_Sig = len(hf['muo_PID_score'][(hf['muo_PID_score'] > 20) & (hf['muo_PID_score'] < -20) ][type == 1])
    nPID_ATLAS_cut = len(hf['muo_PID_score_ATLAS'][(hf['muo_PID_score_ATLAS'] > 20) & (hf['muo_PID_score_ATLAS'] < -20) ])
    nPID_ATLAS_cut_Sig = len(hf['muo_PID_score_ATLAS'][(hf['muo_PID_score_ATLAS'] > 20) & (hf['muo_PID_score_ATLAS'] < -20) ][type == 1])

    print(f"Cutting away {nISO_cut} for ISO with {nISO_cut_Sig} being signal\n")
    print(f"Cutting away {nPID_cut} for PID with {nPID_cut_Sig} being signal\n")
    print(f"Cutting away {nPID_ATLAS_cut} for PID ATLAS with {nPID_ATLAS_cut_Sig} being signal\n")

    hf['muo_ISO_score'] = hf['muo_ISO_score'][(hf['muo_ISO_score'] < 4) & (hf['muo_ISO_score'] > -4) ]
    hf['muo_PID_score'] = hf['muo_PID_score'][(hf['muo_PID_score'] < 20) & (hf['muo_PID_score'] > -20) ]
    hf['muo_PID_score_ATLAS'] = hf['muo_PID_score_ATLAS'][(hf['muo_PID_score_ATLAS'] < 20) & (hf['muo_PID_score_ATLAS'] > -20) ]
    #
    # fig, ax = plt.subplots(1,2,figsize=(10,5))
    # ax = ax.flatten()
    # ax[0].set_title(f"For 11 variables ({label})")
    # ax[0].plot(hf['muo_PID_score'][type==0],hf['muo_ISO_score'][type==0],'b.', alpha = 0.2, label = "Background")#, bins = 50, cmax = 30);
    # ax[0].plot(hf['muo_PID_score'][type==1],hf['muo_ISO_score'][type==1],'r.', alpha = 0.2, label = "Signal")#, bins = 50, cmax = 30);
    # ax[0].set(xlabel = "ML PID", ylabel = "ML ISO");
    # ax[0].legend()
    # #fig.savefig("11vars_PID.pdf")
    # ax[1].set_title(f"For 8 variables, ATLAS ({label})")
    # ax[1].plot(hf['muo_PID_score_ATLAS'][type==0],hf['muo_ISO_score'][type==0],'b.', alpha = 0.2, label = "Background")#, bins = 50, cmax = 30);
    # ax[1].plot(hf['muo_PID_score_ATLAS'][type==1],hf['muo_ISO_score'][type==1],'r.', alpha = 0.2, label = "Signal")#, bins = 50, cmax = 30);
    # ax[1].set(xlabel = "ML PID", ylabel = "ML ISO");
    # ax[1].legend()
    #
    # fig.tight_layout()

    if plot:
        c1 = color[0]
        c2 = color[1]

        fig, ax = plt.subplots(1,2,figsize=(10,5))
        ax = ax.flatten()
        ax[0].set_title(f"For 11 variables ({label})")
        ax[0].plot(hf['muo_PID_score'][type==0],hf['muo_ISO_score'][type==0],'.', color = c1, alpha = 0.2, label = "Background")#, bins = 50, cmax = 30);
        ax[0].plot(hf['muo_PID_score'][type==1],hf['muo_ISO_score'][type==1], '.', color=c2, alpha = 0.2, label = "Signal")#, bins = 50, cmax = 30);
        ax[0].axvline(0, color = "k", linestyle = "dotted")
        ax[0].axhline(0, color = "k", linestyle = "dotted")

        # Calculate percentage in each quadrant
        nSig = len(hf[type==1])
        nBkg = len(hf[type==0])

        nSigQ1 = len(hf[type==1][(hf['muo_PID_score'] > 0) & (hf['muo_ISO_score'] > 0)])
        nSigQ2 = len(hf[type==1][(hf['muo_PID_score'] < 0) & (hf['muo_ISO_score'] > 0)])
        nSigQ3 = len(hf[type==1][(hf['muo_PID_score'] > 0) & (hf['muo_ISO_score'] < 0)])
        nSigQ4 = len(hf[type==1][(hf['muo_PID_score'] < 0) & (hf['muo_ISO_score'] < 0)])

        nBkgQ1 = len(hf[type==0][(hf['muo_PID_score'] > 0) & (hf['muo_ISO_score'] > 0)])
        nBkgQ2 = len(hf[type==0][(hf['muo_PID_score'] < 0) & (hf['muo_ISO_score'] > 0)])
        nBkgQ3 = len(hf[type==0][(hf['muo_PID_score'] > 0) & (hf['muo_ISO_score'] < 0)])
        nBkgQ4 = len(hf[type==0][(hf['muo_PID_score'] < 0) & (hf['muo_ISO_score'] < 0)])

        ax[0].text(0.99,0.96, f"{np.round((nBkgQ1/nBkg)*100,2)}%, n = {nBkgQ1}", color = c1, horizontalalignment='right', transform=ax[0].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))
        ax[0].text(0.99,0.92, f"{np.round((nSigQ1/nSig)*100,2)}%, n = {nSigQ1}", color = c2, horizontalalignment='right', transform=ax[0].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))

        ax[0].text(0.01,0.96, f"{np.round((nBkgQ2/nBkg)*100,2)}%, n = {nBkgQ2}", color = c1, transform=ax[0].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))
        ax[0].text(0.01,0.92, f"{np.round((nSigQ2/nSig)*100,2)}%, n = {nSigQ2}", color = c2, transform=ax[0].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))

        ax[0].text(0.99,0.02, f"{np.round((nBkgQ3/nBkg)*100,2)}%, n = {nBkgQ3}", color = c1, horizontalalignment='right', transform=ax[0].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))
        ax[0].text(0.99,0.06, f"{np.round((nSigQ3/nSig)*100,2)}%, n = {nSigQ3}", color = c2, horizontalalignment='right', transform=ax[0].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))

        ax[0].text(0.01,0.02, f"{np.round((nBkgQ4/nBkg)*100,2)}%, n = {nBkgQ4}", color = c1, transform=ax[0].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))
        ax[0].text(0.01,0.06, f"{np.round((nSigQ4/nSig)*100,2)}%, n = {nSigQ4}", color = c2, transform=ax[0].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))



        ax[0].set(xlabel = "ML PID", ylabel = "ML ISO");
        ax[0].legend(loc = 9, prop={'size': 6})
        #fig.savefig("11vars_PID.pdf")
        ax[1].set_title(f"For 8 variables, ATLAS ({label})")
        ax[1].plot(hf['muo_PID_score_ATLAS'][type==0],hf['muo_ISO_score'][type==0],'.', color = c1, alpha = 0.2, label = "Background")#, bins = 50, cmax = 30);
        ax[1].plot(hf['muo_PID_score_ATLAS'][type==1],hf['muo_ISO_score'][type==1],'.', color=c2, alpha = 0.2, label = "Signal")#, bins = 50, cmax = 30);
        ax[1].axvline(0, color = "k", linestyle = "dotted")
        ax[1].axhline(0, color = "k", linestyle = "dotted")
        ax[1].set(xlabel = "ML PID", ylabel = "ML ISO");
        ax[1].legend(loc = 9, prop={'size': 6})

        # Calculate percentage in each quadrant
        nSig = len(hf[type==1])
        nBkg = len(hf[type==0])
        nSigQ1 = len(hf[type==1][(hf['muo_PID_score_ATLAS'] > 0) & (hf['muo_ISO_score'] > 0)])
        nSigQ2 = len(hf[type==1][(hf['muo_PID_score_ATLAS'] < 0) & (hf['muo_ISO_score'] > 0)])
        nSigQ3 = len(hf[type==1][(hf['muo_PID_score_ATLAS'] > 0) & (hf['muo_ISO_score'] < 0)])
        nSigQ4 = len(hf[type==1][(hf['muo_PID_score_ATLAS'] < 0) & (hf['muo_ISO_score'] < 0)])

        nBkgQ1 = len(hf[type==0][(hf['muo_PID_score_ATLAS'] > 0) & (hf['muo_ISO_score'] > 0)])
        nBkgQ2 = len(hf[type==0][(hf['muo_PID_score_ATLAS'] < 0) & (hf['muo_ISO_score'] > 0)])
        nBkgQ3 = len(hf[type==0][(hf['muo_PID_score_ATLAS'] > 0) & (hf['muo_ISO_score'] < 0)])
        nBkgQ4 = len(hf[type==0][(hf['muo_PID_score_ATLAS'] < 0) & (hf['muo_ISO_score'] < 0)])

        ax[1].text(0.99,0.96, f"{np.round((nBkgQ1/nBkg)*100,2)}%, n = {nBkgQ1}", color = c1, horizontalalignment='right', transform=ax[1].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))
        ax[1].text(0.99,0.92, f"{np.round((nSigQ1/nSig)*100,2)}%, n = {nSigQ1}", color = c2, horizontalalignment='right', transform=ax[1].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))

        ax[1].text(0.01,0.96, f"{np.round((nBkgQ2/nBkg)*100,2)}%, n = {nBkgQ2}", color = c1, transform=ax[1].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))
        ax[1].text(0.01,0.92, f"{np.round((nSigQ2/nSig)*100,2)}%, n = {nSigQ2}", color = c2, transform=ax[1].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))

        ax[1].text(0.99,0.02, f"{np.round((nBkgQ3/nBkg)*100,2)}%, n = {nBkgQ3}", color = c1, horizontalalignment='right', transform=ax[1].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))
        ax[1].text(0.99,0.06, f"{np.round((nSigQ3/nSig)*100,2)}%, n = {nSigQ3}", color = c2, horizontalalignment='right', transform=ax[1].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))

        ax[1].text(0.01,0.02, f"{np.round((nBkgQ4/nBkg)*100,2)}%, n = {nBkgQ4}", color = c1, transform=ax[1].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))
        ax[1].text(0.01,0.06, f"{np.round((nSigQ4/nSig)*100,2)}%, n = {nSigQ4}", color = c2, transform=ax[1].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))


        fig.tight_layout()
        fig.savefig("PidIsoDistScatter_" + label + ".png", dpi = 600)

        fig, ax = plt.subplots(1,3,figsize=(15,5))
        ax = ax.flatten()
        ax[0].set_title(f"For 11 variables ({label})")
        ax[0].hist(hf['muo_PID_score'][type==0], bins = 100, color = c1, label = "Background", histtype = "step")#, bins = 50, cmax = 30);
        ax[0].hist(hf['muo_PID_score'][type==1],bins = 100, color = c2,label = "Signal", histtype = "step")#, bins = 50, cmax = 30);
        ax[0].set(xlabel = "ML PID (11 variables)", ylabel = "Frequency");
        ax[0].legend()

        ax[1].set_title(f"Isolation")
        ax[1].hist(hf['muo_ISO_score'][type==0], bins = 100, color = c1, label = "Background", histtype = "step")#, bins = 50, cmax = 30);
        ax[1].hist(hf['muo_ISO_score'][type==1],bins = 100, color = c2,label = "Signal", histtype = "step")#, bins = 50, cmax = 30);
        ax[1].set(xlabel = "ML ISO", ylabel = "Frequency");
        ax[1].legend()
        #fig.savefig("11vars_PID.pdf")
        ax[2].set_title(f"For 8 variables, ATLAS ({label})")
        ax[2].hist(hf['muo_PID_score_ATLAS'][type==0], bins = 100, color = c1, label = "Background", histtype = "step")#, bins = 50, cmax = 30);
        ax[2].hist(hf['muo_PID_score_ATLAS'][type==1],bins = 100, color = c2,label = "Signal", histtype = "step")#, bins = 50, cmax = 30);
        ax[2].set(xlabel = "ML PID (8 variables)", ylabel = "Frequency");
        ax[2].legend(prop={'size': 6})

        fig.tight_layout()
        fig.savefig("PidIsoHist_" + label + ".pdf")

    return hf

hf_MC = GetScores(hf_MC, type_MC, "MC_new, pt cut > 4.5 GeV", plot = False)
hf_MC_truth = GetScores(hf_MC_truth, type_MC_truth, "MC truth, pt cut > 4.5 GeV", color = ["tab:brown", "tab:pink"], plot = False)

#ax_MC[0].axhline(0, linestyle = "dashed", color="grey", label = "Suggested cut")
#ax_MC[0].axvline(3.5, linestyle = "dashed", color="grey")
#ax_MC[1].axhline(0, linestyle = "dashed", color="grey", label = "Suggested cut")
#ax_MC[1].axvline(3.5, linestyle = "dashed", color="grey")
#ax_MC[0].legend()
#ax_MC[1].legend()
#
# fig, ax = plt.subplots(1,2,figsize=(10,5))
# #mask_iso = (hf_MC['muo_PID_score'][type_MC!=1] > 5) #& (hf_MC['muo_ISO_score'][type_MC!=1] > 0)
# #mask_pid = (hf_MC['muo_ISO_score'][type_MC!=1] > 0.5) #& (hf_MC['muo_ISO_score'][type_MC!=1] > 0)
#
# ax = ax.flatten()
# ax[0].set_title(f"For 11 variables pt cut > 4.5 GeV")
# ax[0].hist(hf_MC['muo_PID_score'][type_MC==0], bins = 100, color = 'b', label = "Background MC T&P", histtype = "step")#, bins = 50, cmax = 30);
# #ax[0].hist(hf_MC['muo_PID_score'][type_MC==2], bins = 100, color = 'b', label = "Background MC T&P, type 2 (trash)", histtype = "stepfilled", alpha = 0.2)#, bins = 50, cmax = 30);
# #ax[0].axvline(5, color = 'k', linestyle = "dashed")
# ax[0].hist(hf_MC['muo_PID_score'][type_MC==1],bins = 100, color = 'r',label = "Signal MC T&P", histtype = "step")#, bins = 50, cmax = 30);
# ax[0].hist(hf_MC_truth['muo_PID_score'][type_MC_truth==0], bins = 100, color = 'g', label = "Background MC truth", histtype = "step")#, bins = 50, cmax = 30);
# ax[0].hist(hf_MC_truth['muo_PID_score'][type_MC_truth==1],bins = 100, color = 'tab:purple',label = "Signal MC truth", histtype = "step")#, bins = 50, cmax = 30);
# ax[0].set(xlabel = "ML PID (11 variables)", ylabel = "Frequency");
# ax[0].legend(loc=2)
#
# ax[1].set_title(f"Isolation")
# ax[1].hist(hf_MC['muo_ISO_score'][type_MC==0], bins = 100, color = 'b', label = "Background MC T&P", histtype = "step")#, bins = 50, cmax = 30);
# #ax[1].hist(hf_MC['muo_ISO_score'][type_MC==2], bins = 100, color = 'b', label = "Background MC T&P, type 2 (trash)", histtype = "stepfilled", alpha = 0.2)#, bins = 50, cmax = 30);
# #ax[1].axvline(0.5, color = 'k', linestyle = "dashed")
# ax[1].hist(hf_MC['muo_ISO_score'][type_MC==1],bins = 100, color = 'r',label = "Signal MC T&P", histtype = "step")#, bins = 50, cmax = 30);
# ax[1].hist(hf_MC_truth['muo_ISO_score'][type_MC_truth==0], bins = 100, color = 'g', label = "Background MC truth", histtype = "step")#, bins = 50, cmax = 30);
# ax[1].hist(hf_MC_truth['muo_ISO_score'][type_MC_truth==1],bins = 100, color = 'tab:purple',label = "Signal MC truth", histtype = "step")#, bins = 50, cmax = 30);
# ax[1].set(xlabel = "ML ISO", ylabel = "Frequency");
# ax[1].legend(loc=2)
# #fig.savefig("11vars_PID.pdf")
# # ax[2].set_title(f"For 8 variables, ATLAS ({label})")
# # ax[2].hist(hf['muo_PID_score_ATLAS'][type==0], bins = 100, color = 'b', label = "Background", histtype = "step")#, bins = 50, cmax = 30);
# # ax[2].hist(hf['muo_PID_score_ATLAS'][type==1],bins = 100, color = 'r',label = "Signal", histtype = "step")#, bins = 50, cmax = 30);
# # ax[2].set(xlabel = "ML PID (8 variables)", ylabel = "Frequency");
# # ax[2].legend()
#
# fig.tight_layout()
# fig.savefig("PidIsoHist_MC_MCTruth.pdf" + ".pdf")


hf_data = GetScores(hf_data, type_data, "Data,  pt > 15 GeV", color = ["g", "tab:purple"], plot = True)

# ax_Data[0].axhline(0, linestyle = "dashed", color="grey", label = "Suggested cut")
# ax_Data[0].axvline(3.5, linestyle = "dashed", color="grey")
# ax_Data[1].axhline(0, linestyle = "dashed", color="grey", label = "Suggested cut")
# ax_Data[1].axvline(3.5, linestyle = "dashed", color="grey")
# ax_Data[0].legend()
# ax_Data[1].legend()


#### Plot the distribution of Variables
#
# training_vars = [training_var_ISO, training_var_PID, training_var_PIDATLAS]
# training_vars_name = ["training_var_ISO", "training_var_PID", "training_var_PIDATLAS"]
#
# data_types = [hf_MC, hf_data]
# data_types_name = ["MC", "Data"]
#
# for k, (data_type, label) in enumerate(zip(data_types, [type_MC, type_data])):
#     for j, training_var in enumerate(training_vars):
#         if training_vars_name[j] == "training_var_PID":
#             fig, ax = plt.subplots(4,3,figsize=(10,15))
#         else:
#             fig, ax = plt.subplots(3,3,figsize=(10,10))
#         ax = ax.flatten()
#         for i,var in enumerate(training_var):
#             data_bkg = data_type[var][label == 0]
#             data_sig = data_type[var][label == 1]
#
#             ax[i].hist(data_bkg, histtype = "step", color = "b", bins = 100, label = "Background", range = (np.percentile(data_bkg, 5),np.percentile(data_bkg, 95)));
#             ax[i].hist(data_sig, histtype = "step", color = "r", bins = 100, label = "Signal", range = (np.percentile(data_sig, 5),np.percentile(data_sig, 95)));
#             ax[i].legend()
#             ax[i].set(xlabel = var, ylabel = "Frequency")
#
#         fig.tight_layout()
#         fig.savefig(f"{training_vars_name[j]}_{data_types_name[k]}" + ".pdf")





## Mutual information
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import pearsonr
hf_data = clean_dataset(hf_data)
hf_MC = clean_dataset(hf_MC)

allCorrVars =  training_var_PID + ["muo_PID_score"] + ["muo_ISO_score"] + training_var_ISO
cmap_green=sns.diverging_palette(10, 130, sep=5, n=20, as_cmap=True)


def GetCorrPlot(data, label, outname, Datatype, mic=False):
    corrTotal = np.empty(shape=(len(allCorrVars), len(allCorrVars)))
    for i in range(corrTotal.shape[0]):
        for j in range(corrTotal.shape[1]):
            if i < j:
                corrTotal[i,j] = pearsonr(data[label][allCorrVars[i]], data[label][allCorrVars[j]])[0]
            elif i > j:
                if mic:
                    corrTotal[i,j] = calc_mic_score(data[label][allCorrVars[i]][:2000], data[label][allCorrVars[j]][:2000])
                else:
                    corrTotal[i,j] = normalized_mutual_info_score(data[label][allCorrVars[i]], data[label][allCorrVars[j]])

    mask = np.zeros_like(corrTotal, dtype=np.bool)
    mask_zeros = (np.abs(corrTotal) < 0.05)
    mask[mask_zeros] = True
    mask[np.triu_indices_from(mask)] = True
    #mask = mask[mask_zeros]
    mask2 = np.zeros_like(corrTotal, dtype=np.bool)
    mask2[mask_zeros] = True
    mask2[np.tril_indices_from(mask2)] = True
    #mask2 = mask2[mask_zeros]

    fig, ax = plt.subplots(figsize=(12,12))
    ax = sns.heatmap(corrTotal,
                xticklabels=allCorrVars,
                cmap="Blues",
                annot=True,
                square=True,
                mask = mask,
                cbar = False,
                fmt=".2f",
                annot_kws={'size':11})
    ax = sns.heatmap(corrTotal, # pearson
                annot=True,
                cmap=cmap_green,
                mask = mask2,
                cbar = False,
                fmt=".2f",
                annot_kws={'size':11},
                vmin = -1,
                vmax = 1)
    ax.hlines([11,13], *ax.get_xlim(), linewidth = 3)
    ax.hlines([12], *ax.get_xlim(), linewidth = 1, linestyle = "dashed")
    ax.vlines([11,13], *ax.get_ylim(), linewidth = 3)
    ax.vlines([12], *ax.get_ylim(), linewidth = 1, linestyle = "dashed")

    ax.set_title("Correlation plot for " + outname + " " + Datatype, size=14)
    #if Datatype[0] == "D":
#        ax.xaxis.tick_top()
#        ax.set_xticklabels(allCorrVars, size=12, rotation = 90);
    ax.set_xticklabels(allCorrVars, size=12);
    #if outname[0] == "s":
    ax.yaxis.tick_right()
    ax.set_yticklabels(allCorrVars, rotation = 0, size=12);

    if mic:
        ax.set_ylabel('Maximal Information Coefficient (MIC)', size=12)
    else:
        ax.set_ylabel('Mutual Information (MI)', size=12)
    #if outname[0] == "b":
    #    ax.yaxis.set_label_position('right')

    ax.set_xlabel("Pearson correlation", size=12);
    #if Datatype[0] == "M":
    ax.xaxis.set_label_position('top')

    fig.tight_layout()
    if mic:
        fig.savefig("CorrPlot_MIC_" + outname + "("+Datatype+")" + ".pdf")
    else:
        fig.savefig("CorrPlot_" + outname + "("+Datatype+")" + ".pdf")

GetCorrPlot(hf_data, type_data==0, "background", "Data_new, pt cut > 4.5 GeV", mic = False)
GetCorrPlot(hf_data, type_data==1, "signal", "Data_new, pt cut > 4.5 GeV", mic = False)
GetCorrPlot(hf_MC, type_MC==0, "background", "MC, pt cut > 4.5 GeV", mic = False)
GetCorrPlot(hf_MC, type_MC==1, "signal", "MC, pt cut > 4.5 GeV", mic = False)

hf_MC_truth = clean_dataset(hf_MC_truth)
type_MC_truth = hf_MC_truth["Type"]
GetCorrPlot(hf_MC_truth, type_MC_truth==0, "background", "MC Truth, pt cut > 4.5 GeV", mic = False)
GetCorrPlot(hf_MC_truth, type_MC_truth==1, "signal", "MC Truth, pt cut > 4.5 GeV", mic = False)

### Cutting the data

hf_data_cut = hf_data.copy()
type_data_cut = hf_data_cut["Type"]
hf_data_cut[type_data_cut == 1]
b = hf_data_cut[(type_data_cut == 1) & (hf_data_cut["muo_ISO_score"] < 0) & (hf_data_cut["muo_PID_score"] < 0)]


hf_MC_cut = hf_MC.copy()
type_MC_cut = hf_MC_cut["Type"]

#a = hf_MC_cut[(type_MC_cut == 1) & (hf_MC_cut["muo_ISO_score"] < 0) & (hf_MC_cut["muo_PID_score"] < 0)]
McSigCut = hf_MC_cut[(type_MC_cut == 1) & (hf_MC_cut["muo_ISO_score"] > 0)]
#hf_MC_cut[type_MC == 0] = hf_MC_cut[type_MC == 0][((hf_MC_cut["muo_ISO_score"] < 0) & (hf_MC_cut["muo_PID_score"] < 3.5))]
hf_MC_cut[type_MC_cut == 1]




fig, ax = plt.subplots(1,2,figsize=(10,5))
ax = ax.flatten()
ax[0].set_title(f"For 11 variables \n (Data, cut pt > 4.5 GeV and ISO > 0 for signal)")
ax[0].plot(hf_data_cut['muo_PID_score'][type_data_cut==0],hf_data_cut['muo_ISO_score'][type_data_cut==0],'b.', alpha = 0.2, label = "Background")#, bins = 50, cmax = 30);
ax[0].plot(hf_data_cut['muo_PID_score'][type_data_cut==1],hf_data_cut['muo_ISO_score'][type_data_cut==1],'r.', alpha = 0.2, label = "Signal")#, bins = 50, cmax = 30);
ax[0].set(xlabel = "ML PID", ylabel = "ML ISO");
ax[0].legend()
#fig.savefig("11vars_PID.pdf")
ax[1].set_title(f"For 8 variables, ATLAS \n (Data, cut pt > 4.5 GeV and ISO > 0 for signal)")
ax[1].plot(hf_data_cut['muo_PID_score_ATLAS'][type_data_cut==0],hf_data_cut['muo_ISO_score'][type_data_cut==0],'b.', alpha = 0.2, label = "Background")#, bins = 50, cmax = 30);
ax[1].plot(hf_data_cut['muo_PID_score_ATLAS'][type_data_cut==1],hf_data_cut['muo_ISO_score'][type_data_cut==1],'r.', alpha = 0.2, label = "Signal")#, bins = 50, cmax = 30);
ax[1].set(xlabel = "ML PID", ylabel = "ML ISO");
ax[1].legend()

fig.tight_layout()
fig.savefig("PidIsoDist_Data_CuttedIsoSignal.png", dpi = 600)

fig, ax = plt.subplots(1,2,figsize=(10,5))
ax = ax.flatten()
ax[0].set_title(f"For 11 variables \n (MC, cut pt > 4.5 GeV and ISO > 0 for signal)")
ax[0].plot(hf_MC_cut['muo_PID_score'][type_MC_cut==0],hf_MC_cut['muo_ISO_score'][type_MC_cut==0],'b.', alpha = 0.2, label = "Background")#, bins = 50, cmax = 30);
ax[0].plot(hf_MC_cut['muo_PID_score'][type_MC_cut==1],hf_MC_cut['muo_ISO_score'][type_MC_cut==1],'r.', alpha = 0.2, label = "Signal")#, bins = 50, cmax = 30);
ax[0].set(xlabel = "ML PID", ylabel = "ML ISO");
ax[0].legend()
#fig.savefig("11vars_PID.pdf")
ax[1].set_title(f"For 8 variables, ATLAS \n (MC, cut pt > 4.5 GeV and ISO > 0 for signal)")
ax[1].plot(hf_MC_cut['muo_PID_score_ATLAS'][type_MC_cut==0],hf_MC_cut['muo_ISO_score'][type_MC_cut==0],'b.', alpha = 0.2, label = "Background")#, bins = 50, cmax = 30);
ax[1].plot(hf_MC_cut['muo_PID_score_ATLAS'][type_MC_cut==1],hf_MC_cut['muo_ISO_score'][type_MC_cut==1],'r.', alpha = 0.2, label = "Signal")#, bins = 50, cmax = 30);
ax[1].set(xlabel = "ML PID", ylabel = "ML ISO");
ax[1].legend()

fig.tight_layout()
fig.savefig("PidIsoDist_MC_CuttedIsoSignal.png", dpi = 600)

hf_data_cut = clean_dataset(hf_data_cut)
hf_MC_cut = clean_dataset(hf_MC_cut)
