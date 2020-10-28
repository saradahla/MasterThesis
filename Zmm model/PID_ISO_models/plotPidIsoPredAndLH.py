"""
Created on Tuesday 30 June
@author: Sara Dahl Andersen

Plotting the ROC scores for PID and ISO models and comparing them with the likelihood from ATLAS

nohup python -u muoISOapply.py --tag 110820_ZbbW  output/ISOReweightFiles/110820_ZbbW/combined_110820_train.h5 2>&1 &> output/logIsoApply.txt & disown

"""
print("Program running...")

import os
import argparse
import logging as log

from time import time
from datetime import timedelta
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import randint


from utils import mkdir
from hep_ml.reweight import GBReweighter

from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc, roc_auc_score
import shap
import lightgbm as lgb
from scipy.special import logit


# ================================================ #
#                   Functions                      #
# ================================================ #

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

def GetISOscore(gbm, data):
    training_var = [f'muo_etcone20',
                    f'muo_ptcone20',
                    f'muo_pt',
                    f'muo_etconecoreConeEnergyCorrection',
                    f'muo_neflowisolcoreConeEnergyCorrection',
                    f'muo_ptconecoreTrackPtrCorrection',
                    f'muo_topoetconecoreConeEnergyCorrection']
    score = gbm.predict(data[training_var], n_jobs=1)
    return logit(score)

def GetPIDscore(gbm, data):
    training_var = [f'muo_numberOfPrecisionLayers',
                    f'muo_numberOfPrecisionHoleLayers',
                    f'muo_quality',
                    f'muo_ET_TileCore',
                    f'muo_MuonSpectrometerPt',
                    f'muo_deltatheta_1',
                    f'muo_scatteringCurvatureSignificance', # PID
                    f'muo_scatteringNeighbourSignificance', # PID
                    f'muo_momentumBalanceSignificance', # PID
                    f'muo_EnergyLoss', # PID
                    f'muo_energyLossType']

    score = gbm.predict(data[training_var], n_jobs=1)
    return logit(score)


dataPID = h5ToDf("/Users/sda/hep/work/Zmm model/PID_ISO_models/output/PIDReweightFiles/071020_ZbbW/combined_010920_train.h5")
dataPID = dataPID[dataPID["dataset"] == 1] #only validation set
dataISO = h5ToDf("/Users/sda/hep/work/Zmm model/PID_ISO_models/output/ISOReweightFiles/110820_ZbbW/combined_110820_train.h5")
dataISO = dataISO[dataISO["dataset"] == 1] #only validation set
#modelPID = "/groups/hep/sda/work/Zmm model/PID_ISO_models/output/PIDModels/010920_ZbbW/lgbmPID.txt"
modelPID = "/Users/sda/hep/work/Zmm model/PID_ISO_models/output/PIDModels/071020_ZbbW/lgbmPID.txt"
modelISO = "/Users/sda/hep/work/Zmm model/PID_ISO_models/output/ISOModels/110820_ZbbW/lgbmISO.txt"
#modelISO = "/groups/hep/sda/work/Zmm model/PID_ISO_models/output/ISOModels/110820_ZbbW/lgbmISO.txt"

PIDmod = lgb.Booster(model_file = modelPID)
ISOmod = lgb.Booster(model_file = modelISO)

dataPID['muo_PID_score'] = GetPIDscore(PIDmod,dataPID)
dataISO['muo_ISO_score'] = GetISOscore(ISOmod,dataISO)

dataPID = dataPID[(dataPID['muo_PID_score'] > -40) & (dataPID['muo_PID_score'] < 20)]
fpr_PID, tpr_PID, thresholds_PID = roc_curve(dataPID['label'], dataPID['muo_PID_score'], sample_weight=dataPID["weightEst15"])
auc_PID = auc(fpr_PID, tpr_PID)
fpr_PID_Loose, tpr_PID_Loose, thresholds_PID_Loose = roc_curve(dataPID['label'], dataPID['muo_LHLoose'], sample_weight=dataPID["weightEst15"])
auc_PID_Loose = auc(fpr_PID_Loose, tpr_PID_Loose)
fpr_PID_Medium, tpr_PID_Medium, thresholds_PID_Medium = roc_curve(dataPID['label'], dataPID['muo_LHMedium'], sample_weight=dataPID["weightEst15"])
auc_PID_Medium = auc(fpr_PID_Medium, tpr_PID_Medium)
fpr_PID_Tight, tpr_PID_Tight, thresholds_PID_Tight = roc_curve(dataPID['label'], dataPID['muo_LHTight'], sample_weight=dataPID["weightEst15"])
auc_PID_Tight = auc(fpr_PID_Tight, tpr_PID_Tight)


fig, ax = plt.subplots(1,1, figsize=(6,5))
ax.plot(tpr_PID, fpr_PID, color = 'C0', label=f"AUC PID model = {np.round(auc_PID,3)}")
ax.plot(tpr_PID_Loose[1], fpr_PID_Loose[1], color = 'C2', marker='*', markersize=10, label=f"AUC LH Loose = {np.round(auc_PID_Loose,3)}")
ax.plot(tpr_PID_Medium[1], fpr_PID_Medium[1], color = 'C9' ,marker='*', markersize=10,label=f"AUC LH Medium = {np.round(auc_PID_Medium,3)}")
ax.plot(tpr_PID_Tight[1], fpr_PID_Tight[1], color = 'C8', marker='*', markersize=10,label=f"AUC LH Tight = {np.round(auc_PID_Tight,3)}")
# ax.plot(tpr_ISO, fpr_ISO, color = 'C3', label=f"AUC ISO model = {np.round(auc_ISO,3)}")
# ax.plot(tpr_ISO_Loose[1], fpr_ISO_Loose[1], color = 'C1', marker='*',markersize=10,label=f"AUC LH Loose (ISO) = {np.round(auc_ISO_Loose,3)}")
# ax.plot(tpr_ISO_Medium[1], fpr_ISO_Medium[1], color = 'C6', marker='*',markersize=10,label=f"AUC LH Medium (ISO) = {np.round(auc_ISO_Medium,3)}")
# ax.plot(tpr_ISO_Tight[1], fpr_ISO_Tight[1], color = 'C4', marker='*',markersize=10,label=f"AUC LH Tight (ISO) = {np.round(auc_ISO_Tight,3)}")
ax.set_ylabel('Background efficiency')
ax.set_xlabel('Signal efficiency')
ax.set_yscale('log', nonposy='clip')
ax.legend(loc='best', prop={'size': 13})
ax.set(xlim = (0.6,1))
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)
fig.tight_layout()
fig.savefig("ROC_train_valid_LH_ML" + ".pdf")
plt.close()



fpr_ISO, tpr_ISO, thresholds_ISO = roc_curve(dataISO['label'], dataISO['muo_ISO_score'], sample_weight=dataISO["weightEst20"])
auc_ISO = auc(fpr_ISO, tpr_ISO)

dataISO["Isolated_ptcone"] = dataISO['muo_ptvarcone30']/dataISO['muo_pt']

# fig, ax = plt.subplots(figsize=(5,5))
# ax.hist(dataISO["Isolated_ptcone"], bins = 50,range = (0, 0.1));
# ax.set_yscale('log')
# ax.set(xlabel = "ptvarcone30/pt", ylabel = "Frequency")
# fig.tight_layout()
# fig.savefig("ptvarcone_pt.pdf")

fpr_ISO_Isolation, tpr_ISO_Isolation, thresholds_ISO_Isolation = roc_curve(dataISO['label'], dataISO["Isolated_ptcone"], sample_weight=dataISO["weightEst20"])
auc_ISO_Isolation = auc(fpr_ISO_Isolation, tpr_ISO_Isolation)
fpr_ISO_Isolation_cut, tpr_ISO_Isolation_cut, thresholds_ISO_Isolation_cut = roc_curve(dataISO['label'], dataISO["Isolated_ptcone"] < 0.06, sample_weight=dataISO["weightEst20"])
auc_ISO_Isolation_cut = auc(fpr_ISO_Isolation_cut, tpr_ISO_Isolation_cut)

fig, ax = plt.subplots(1,1, figsize=(6,5))
# ax.plot(tpr_PID, fpr_PID, color = 'C0', label=f"AUC PID model = {np.round(auc_PID,3)}")
# ax.plot(tpr_PID_Loose[1], fpr_PID_Loose[1], color = 'C2', marker='*', markersize=10, label=f"AUC LH Loose = {np.round(auc_PID_Loose,3)}")
# ax.plot(tpr_PID_Medium[1], fpr_PID_Medium[1], color = 'C9' ,marker='*', markersize=10,label=f"AUC LH Medium = {np.round(auc_PID_Medium,3)}")
# ax.plot(tpr_PID_Tight[1], fpr_PID_Tight[1], color = 'C8', marker='*', markersize=10,label=f"AUC LH Tight = {np.round(auc_PID_Tight,3)}")
ax.plot(tpr_ISO, fpr_ISO, color = 'C3', label=f"AUC ISO model = {np.round(auc_ISO,3)}")
ax.plot(1-tpr_ISO_Isolation, 1-fpr_ISO_Isolation, color = 'C1', label=f"AUC ptvarcone30/muo_pt = {np.round(auc_ISO_Isolation,3)}") #marker='*',markersize=10,
ax.plot(tpr_ISO_Isolation_cut[1], fpr_ISO_Isolation_cut[1], color = 'C1', marker='*',markersize=10, label=f"ptvarcone30/muo_pt < 0.06") #,
# ax.plot(tpr_ISO_Medium[1], fpr_ISO_Medium[1], color = 'C6', marker='*',markersize=10,label=f"AUC LH Medium (ISO) = {np.round(auc_ISO_Medium,3)}")
# ax.plot(tpr_ISO_Tight[1], fpr_ISO_Tight[1], color = 'C4', marker='*',markersize=10,label=f"AUC LH Tight (ISO) = {np.round(auc_ISO_Tight,3)}")
ax.set_ylabel('Background efficiency')
ax.set_xlabel('Signal efficiency')
ax.set_yscale('log', nonposy='clip')
ax.legend(loc='best', prop={'size': 13})
ax.set(xlim = (0.6,1), ylim = (10**(-2), 1))
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)
fig.tight_layout()
fig.savefig("ROC_train_valid_iso_ML" + ".pdf")
plt.close()

tpr_ISO_Isolation_cut
