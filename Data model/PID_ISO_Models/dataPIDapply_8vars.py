#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Tuesday 30 June
@author: Sara Dahl Andersen

Running my PID algorithm for muons and outputting the score for data
to be used in the Z model

nohup python -u dataPIDapply_8vars.py --tag 160920_Data_8vars  output/dataReweightFiles/110920_Data/combined_combined_train.h5 2>&1 &> output/logPidApply_8vars.txt & disown

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
# 'muo_numberOfPrecisionLayers',
'muo_numberOfPrecisionHoleLayers',
# 'muo_quality',
# 'muo_LHMedium',
# 'muo_LHTight',
'muo_ET_TileCore',
# 'muo_MuonSpectrometerPt',
# 'muo_deltaphi_0',
# 'muo_deltaphi_1',
# 'muo_deltatheta_0',
'muo_deltatheta_1',
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
'muo_energyLossType'
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
print("The dataset is created with the weightEst40")

train_dataset = lgb.Dataset(X_train, label=y_train, weight=data_train["weightEst40"])
valid_dataset = lgb.Dataset(X_valid, label=y_valid, weight=data_valid["weightEst40"])


params = {
              'boosting_type': 'gbdt',        # Default gbdt (traditional Gradient Boosting Decision Tree)
              'objective': 'binary',          # Probability labeĺs in [0,1]
              'boost_from_average': True,
              'verbose': 0,                   # < 0: Fatal, = 0: Error (Warning), = 1: Info, > 1: Debug
              'num_threads': args.njobs,
              'num_leaves': 31,
              'learning_rate':0.1
              }

print(f"Parameters:")
keys = list(params.keys())
for i in range(len(keys)):
    print(f"        {keys[i]}: {params[keys[i]]}")
#
# m, arr = hyp_lgbm(train_dataset)
#
#
# plt.plot(arr, 'b.');
#
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.model_selection import ParameterSampler
#
#
# def RandomSearch(model, dataset, label):
#     params_set = {
#                    'boosting_type': 'gbdt',        # Default gbdt (traditional Gradient Boosting Decision Tree)
#                    'objective': 'binary',          # Probability labeĺs in [0,1]
#                    'boost_from_average': True,
#                    'verbose': 0#,                   # < 0: Fatal, = 0: Error (Warning), = 1: Info, > 1: Debug
#                    #'num_threads': args.njobs,
#                    }
#     params_grid = {
#                 'num_leaves': randint(20,40),                    # Important! Default: 31, set to less than 2^(max_depth)
#                 'max_depth': randint(-20,20),                       # Important! Default: -1, <= 0 means no limit
#                 'min_data_in_leaf': randint(10,100),              # Important! Default: 20
#                 'feature_fraction': [1.0],                        # Default: 1.0, random selection if it is under 1 eg. 80% of features for 0.8, helps with: speed up, over-fitting
#                 'bagging_fraction': [1.0],                        # Default: 1.0
#                 'bagging_freq': [0],                              # Default: 0, bags data, should be combined vith bagging_fraction
#                 }
#
#     n_iter = 5
#     cv_scores = np.zeros((n_iter,6))
#     best_scores = np.zeros(n_iter)
#     param_list = list(ParameterSampler(params_grid, n_iter=n_iter, random_state=0))
#     rounded_list = [dict((k, round(v, 3)) for (k, v) in d.items()) for d in param_list]
#
#     for i in range(n_iter):
#         params_cv = params_set
#         params_cv.update(rounded_list[i])
#
#         if (params_cv['max_depth']>0) & (params_cv['num_leaves'] > 2**(params_cv['max_depth'])):
#             cv_scores[i] = np.array([999,999,999,params_cv['num_leaves'],params_cv['min_data_in_leaf'],params_cv['max_depth']])
#             continue
#
#         RandomSearch = RandomizedSearchCV(model,
#                                       param_distributions=params_grid,
#                                       n_iter=2,
#                                       cv=5,
#                                       iid=True,
#                                       return_train_score=True,
#                                       random_state=42)
#         RandomSearch.fit(dataset, label);
#
#         best_results = np.array([v for k,v in RandomSearch.best_params_.items()])
#         best_results_name = np.array([k for k,v in RandomSearch.best_params_.items()])
#         best_score = RandomSearch.best_score_
#
#         cv_scores[i] = best_results
#         best_scores[i] = best_score
#
#         print(f"Done with {i+1} out of {n_iter + 1} iterations")
#
#     return cv_scores, best_scores
# lgm = lgb.LGBMClassifier()
# a, b = RandomSearch(lgm, X_train, y_train)
#
# h = np.array([v for k,v in a.items()])
# j = np.array([k for k,v in a.items()])
# b
# np.column_stack((h,j))
#
# def hyp_lgbm(dataset):
#
#         params_set = {
#                        'boosting_type': 'gbdt',        # Default gbdt (traditional Gradient Boosting Decision Tree)
#                        'objective': 'binary',          # Probability labeĺs in [0,1]
#                        'boost_from_average': True,
#                        'verbose': 0#,                   # < 0: Fatal, = 0: Error (Warning), = 1: Info, > 1: Debug
#                        #'num_threads': args.njobs,
#                        }
#         params_grid = {
#                     'num_leaves': randint(20,40),                    # Important! Default: 31, set to less than 2^(max_depth)
#                     'max_depth': randint(-20,20),                       # Important! Default: -1, <= 0 means no limit
#                     'min_data_in_leaf': randint(10,100),              # Important! Default: 20
#                     'feature_fraction': [1.0],                        # Default: 1.0, random selection if it is under 1 eg. 80% of features for 0.8, helps with: speed up, over-fitting
#                     'bagging_fraction': [1.0],                        # Default: 1.0
#                     'bagging_freq': [0],                              # Default: 0, bags data, should be combined vith bagging_fraction
#                     }
#
#         n_iter = 5
#         cv_scores = np.zeros((n_iter,4))
#         best_scores = np.zeros(n_iter)
#         param_list = list(ParameterSampler(params_grid, n_iter=n_iter, random_state=0))
#         rounded_list = [dict((k, round(v, 3)) for (k, v) in d.items()) for d in param_list]
#
#         for i in range(n_iter):
#             params_cv = params_set
#             params_cv.update(rounded_list[i])
#             params_cv.update(learning_rate=0.1)
#
#             if (params_cv['max_depth']>0) & (params_cv['num_leaves'] > 2**(params_cv['max_depth'])):
#                 cv_scores[i] = np.array([999,params_cv['num_leaves'],params_cv['min_data_in_leaf'],params_cv['max_depth']])
#                 continue
#
#             cv_result = lgb.cv( params_cv,
#                                 dataset,
#                                 nfold=5,
#                                 #n_iter = 20,
#                                 num_boost_round=500,
#                                 early_stopping_rounds=100,
#                                 verbose_eval = False,
#                                 metrics = ['l1'])
#
#             best_results = np.array([-np.min(cv_result['l1-mean']),
#                                      params_cv['num_leaves'],
#                                      params_cv['min_data_in_leaf'],
#                                      params_cv['max_depth']
#                                      ])
#
#             cv_scores[i] = best_results
#             print(f"Done with {i+1} out of {n_iter} iterations")
#
#         return cv_scores
#
# arr = hyp_lgbm(train_dataset)
# arr[np.argmin(arr[arr[:,0] != 999][:,0])]
#============================================================================
# Train the model
#============================================================================

print(f"Training...")

num_round=50
bst = lgb.train(params, train_dataset, num_round)

print('Saving model...')
bst.save_model(args.outdir + "lgbmPID_8vars" + ".txt")

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


fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_pred_train, sample_weight=data_train["weightEst40"])
auc_train = auc(fpr_train, tpr_train)

fpr_valid, tpr_valid, thresholds_valid = roc_curve(y_valid, y_pred_valid, sample_weight=data_valid["weightEst40"])
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
# Plotting SHAP values
#============================================================================
# class names
classes = ['b', 'b']

# set RGB tuple per class
colors = [(0, 0, 1), (0, 0, 1)]

# get class ordering from shap values

# create listed colormap
from matplotlib import colors as plt_colors

shap_values_train = shap.TreeExplainer(bst).shap_values(X_train, tree_limit = -1)
class_inds = np.argsort([-np.abs(shap_values_train[i]).mean() for i in range(len(shap_values_train))])
cmap = plt_colors.ListedColormap(np.array(colors)[class_inds])

shap.summary_plot(shap_values_train, training_var, plot_type = 'bar', color = cmap, show = False, color_bar_label=None)
f = plt.gcf()
f.tight_layout()
f.savefig(args.outdir + "SHAPvalues_train" + ".pdf")
plt.close()

shap_values_valid = shap.TreeExplainer(bst).shap_values(X_valid, tree_limit = -1)
class_inds = np.argsort([-np.abs(shap_values_valid[i]).mean() for i in range(len(shap_values_valid))])
cmap = plt_colors.ListedColormap(np.array(colors)[class_inds])

shap.summary_plot(shap_values_valid, training_var, plot_type = 'bar', color = cmap, show = False, color_bar_label=None)
f = plt.gcf()
f.tight_layout()
f.savefig(args.outdir + "SHAPvalues_valid" + ".pdf")

log.info(f"Done! Total time: {timedelta(seconds=time() - t_start)}")
