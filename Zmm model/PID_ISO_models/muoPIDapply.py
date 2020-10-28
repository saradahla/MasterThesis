#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Tuesday 30 June
@author: Sara Dahl Andersen

Running my PID algorithm for muons and outputting the score
to be used in the Z model

nohup python -u muoPIDapply.py --tag 010920_ZbbW  output/PIDReweightFiles/010920_ZbbW/combined_010920_train.h5 2>&1 &> output/logPidApply.txt & disown

nohup python -u muoPIDapply.py --tag 300920_ZbbW  output/PIDReweightFiles/010920_ZbbW/combined_010920_train.h5 2>&1 &> output/logPidApply.txt & disown

nohup python -u muoPIDapply.py --tag 081020_ZbbW  output/PIDReweightFiles/071020_ZbbW/combined_010920_train.h5 2>&1 &> output/logPidApply.txt & disown

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

from utils import mkdir, Plot, Histogram
from hep_ml.reweight import GBReweighter

from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc, roc_auc_score
import shap
import lightgbm as lgb
from scipy.special import logit


# Logging style and level
log.basicConfig(format='[%(levelname)s] %(message)s', level=log.INFO)
log.info("Packages imported")

# Start "timer"
t_start = time()


# Command line options
parser = argparse.ArgumentParser(description="Combine datafiles, reweigh data and add columns.")
parser.add_argument('--outdir', action='store', default="output/PIDModels/", type=str,
                    help='Output directory.')
parser.add_argument('path', type=str, nargs='+',
                    help='HDF5 file(s) to use for PID.')
parser.add_argument('--tag', action='store', type=str, required=False, default="",
                    help='Tag the data category (Zee, Wev, etc.).')
parser.add_argument('--njobs', action='store', default=10, type=int,
                    help='Maximum number of concurrent processes to use.')


args = parser.parse_args()

# Validate arguments
if not args.path:
    log.error("No HDF5 file was specified.")
    quit()

if args.njobs > 20:
    log.error("The requested number of jobs ({}) is excessive (>20). Exiting.".format(args.njobs))
    quit()

# Make and set the output directory to tag, if it doesn't already exist
# Will stop if the output already exists since re-running is either not needed or unwanted
# If it's wanted, delete the output first yourself
args.outdir = args.outdir+args.tag+f"/"
if os.path.exists(args.outdir):
    log.error(f"Output already exists - please remove yourself. Output: {args.outdir}")
    quit()
else:
    log.info(f"Creating output folder: {args.outdir}")
    mkdir(args.outdir)

# File number counter (incremented first in loop)
counter = -1

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

# ================================================ #
#                End of functions                  #
# ================================================ #


# Data
data_get = h5ToDf(args.path[0])

data_train = data_get[data_get["dataset"] == 0]
data_valid = data_get[data_get["dataset"] == 1]

# Check shapes
shapeAll = np.shape(data_get)
shapeTrain = np.shape(data_train)
shapeValid = np.shape(data_valid)

log.info(f"Shape all:       {shapeAll}")
log.info(f"Shape train:     {shapeTrain}")
log.info(f"Shape valid:     {shapeValid}")

# =========================
#       Variables
# =========================

truth_var = "label"
training_var = [
# 'correctedScaledAverageMu',
# 'NvtxReco',
# 'muo_truthPdgId',
# 'muo_truthType',
# 'muo_truthOrigin',
# 'muo_etcone20',
# 'muo_etcone30',
# 'muo_etcone40',
# 'muo_ptcone20',
# 'muo_ptcone30',
# 'muo_ptcone40',
# 'muo_ptvarcone20',
# 'muo_ptvarcone30',
# 'muo_ptvarcone40',
# 'muo_pt',
# 'muo_eta',
# 'muo_phi',
# 'muo_muonType',
'muo_numberOfPrecisionLayers',
'muo_numberOfPrecisionHoleLayers', # Potentially used: # of missing precision hits
'muo_quality',
# 'muo_LHMedium',
# 'muo_LHTight',
'muo_ET_TileCore', #MS Segments associated with the Muon
'muo_MuonSpectrometerPt',
# 'muo_deltaphi_0',
# 'muo_deltaphi_1',
# 'muo_deltatheta_0',
'muo_deltatheta_1', #direction relative to track in ID
# 'muo_etconecoreConeEnergyCorrection',
# 'muo_sigmadeltaphi_0',
# 'muo_sigmadeltaphi_1',
# 'muo_sigmadeltatheta_0',
# 'muo_sigmadeltatheta_1',
# 'muo_neflowisolcoreConeEnergyCorrection',
# 'muo_ptconecoreTrackPtrCorrection',
# 'muo_topoetconecoreConeEnergyCorrection',
'muo_scatteringCurvatureSignificance', # PID
'muo_scatteringNeighbourSignificance', # PID
'muo_momentumBalanceSignificance', # PID
'muo_EnergyLoss', # PID
'muo_energyLossType', # PID
]

#============================================================================
# LGBM dataset and parameters
#============================================================================

log.info(f"LGBM")
t_start = time()


X_train = data_train[training_var]
y_train = data_train[truth_var]
X_valid = data_valid[training_var]
y_valid = data_valid[truth_var]

# create LGBM dataset
print("The dataset is created with the weightEst15")
train_dataset = lgb.Dataset(X_train, label=y_train, weight=data_train["weightEst15"])
valid_dataset = lgb.Dataset(X_valid, label=y_valid, weight=data_valid["weightEst15"])


params = {
              'boosting_type': 'gbdt',        # Default gbdt (traditional Gradient Boosting Decision Tree)
              'objective': 'binary',          # Probability labeĺs in [0,1]
              'boost_from_average': True,
              'verbose': 0,                   # < 0: Fatal, = 0: Error (Warning), = 1: Info, > 1: Debug
              'num_threads': args.njobs,
              'learning_rate':0.1,
              'num_leaves': 30,
              'max_depth': -1,
              }

print(f"Parameters:")
keys = list(params.keys())
for i in range(len(keys)):
    print(f"        {keys[i]}: {params[keys[i]]}")



#============================================================================
# Train the model
#============================================================================

print(f"Training...")

num_round=500
bst = lgb.train(params, train_dataset, num_round)

print('Saving model...')
bst.save_model(args.outdir + "lgbmPID" + ".txt")

#============================================================================
# Predict
#============================================================================

y_pred_train = bst.predict(X_train, num_iteration=bst.best_iteration)
y_pred_valid = bst.predict(X_valid, num_iteration=bst.best_iteration)

print('AUC score of prediction:')
print(f"        Training:   {roc_auc_score(y_train, y_pred_train):.6f}")
print(f"        Validation: {roc_auc_score(y_valid, y_pred_valid):.6f}")
#print('AUC score of prediction (weighted):')
#print(f"        Training:   {roc_auc_score(y_train, y_pred_train, sample_weight=data_train["weight"]):.6f}")
#print(f"        Validation: {roc_auc_score(y_valid, y_pred_valid, sample_weight=data_valid["weight"]):.6f}")



#============================================================================
# Plotting ROC curve
#============================================================================


fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_pred_train, sample_weight=data_train["weightEst15"])
auc_train = auc(fpr_train, tpr_train)

fpr_valid, tpr_valid, thresholds_valid = roc_curve(y_valid, y_pred_valid, sample_weight=data_valid["weightEst15"])
auc_valid = auc(fpr_valid, tpr_valid)

fprs = [fpr_train, fpr_valid]
tprs = [tpr_train, tpr_valid]
aucs = [auc_train, auc_valid]
names = ["train", "valid"]

fig, ax = plt.subplots(1,1, figsize=(8,5))
for fpr, tpr, auc, name in zip(fprs, tprs, aucs, names):
    ax.plot(tpr, fpr, label="LGBM {:s} (area = {:.3f})".format(name, auc))
ax.set_ylabel('Background efficiency')
ax.set_xlabel('Signal efficiency')
ax.set_yscale('log', nonposy='clip')
ax.legend(loc='best')
fig.tight_layout()
fig.savefig(args.outdir + "ROC_train_valid" + ".pdf")
plt.close()

#============================================================================
# Plotting LGBM score
#============================================================================
fig, ax = plt.subplots(1,1, figsize=(8,5))
ax.hist(logit(y_pred_valid)[(y_valid == 1) & (logit(y_pred_valid) < 20) & (logit(y_pred_valid) > -20)], bins = 100, histtype = 'step', color = 'r', label="Signal")
ax.hist(logit(y_pred_valid)[(y_valid == 0) & (logit(y_pred_valid) < 20) & (logit(y_pred_valid) > -20)], bins = 100, histtype = 'step', color = 'b', label="Background")
ax.set_ylabel('Frequency')
ax.set_xlabel('LGBM score')
ax.legend(loc='best')
fig.tight_layout()
fig.savefig(args.outdir + "LGBM_valid" + ".pdf")
plt.close()
#============================================================================
# Plotting SHAP values
#============================================================================
#
# shap_values_train = shap.TreeExplainer(bst).shap_values(X_train)
#
# shap_values_df = pd.DataFrame(shap_values_train[0], columns = training_var)
# shap_val = shap_values_df.abs().mean(0)
# shap_val_sort = shap_val.sort_values(0, ascending = False)
#
# # Get names of values and positions for plot
# shap_name_sort = list(shap_val_sort.index.values)
# shap_pos = np.arange(len(shap_name_sort))
#
# # Plot shap values
# print('Plotting SHAP values for training set...')
# fig_shap, ax_shap = plt.subplots(figsize=(6,3))
# ax_shap.barh(shap_pos, shap_val_sort, align='center')
# ax_shap.set_yticks(shap_pos)
# ax_shap.set_yticklabels(shap_name_sort)
# ax_shap.invert_yaxis()  # labels read top-to-bottom
# ax_shap.set_xlabel('mean(|SHAP|)')
# #ax_shap.set_title('SHAP values - Average impact on model output')
# plt.tight_layout()
# fig_shap.savefig(args.outdir + 'featureImportance_SHAP_train' + '.pdf')
#
# shap_values_valid = shap.TreeExplainer(bst).shap_values(X_valid)
#
# shap_values_df = pd.DataFrame(shap_values_valid[0], columns = training_var)
# shap_val = shap_values_df.abs().mean(0)
# shap_val_sort = shap_val.sort_values(0, ascending = False)
#
# # Get names of values and positions for plot
# shap_name_sort = list(shap_val_sort.index.values)
# shap_pos = np.arange(len(shap_name_sort))
#
# # Plot shap values
# print('Plotting SHAP values for validation set...')
# fig_shap, ax_shap = plt.subplots(figsize=(6,3))
# ax_shap.barh(shap_pos, shap_val_sort, align='center')
# ax_shap.set_yticks(shap_pos)
# ax_shap.set_yticklabels(shap_name_sort)
# ax_shap.invert_yaxis()  # labels read top-to-bottom
# ax_shap.set_xlabel('mean(|SHAP|)')
# #ax_shap.set_title('SHAP values - Average impact on model output')
# plt.tight_layout()
# fig_shap.savefig(args.outdir + 'featureImportance_SHAP_valid' + '.pdf')


log.info(f"Done! Total time: {timedelta(seconds=time() - t_start)}")
