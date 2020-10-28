#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Tuesday 30 June
@author: Sara Dahl Andersen

Running my ISO algorithm for photons and outputting the score
to be used in the Z model

nohup python -u phoISOmodel.py --tag 20201027  output/phoReweightFiles/20201027/combined_20201027_train.h5 2>&1 &> output/logIsoApply.txt & disown
nohup python -u phoISOmodel.py --tag 20201027_orig3  output/phoReweightFiles/20201027_cutOrig/combined_20201027_train.h5 2>&1 &> output/logIsoApply.txt & disown
nohup python -u phoISOmodel.py --tag 20201028 --hypopt 1  output/phoReweightFiles/20201028/combined_20201028_train.h5 2>&1 &> output/logIsoApply.txt & disown

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
x

from utils import mkdir
from hep_ml.reweight import GBReweighter

from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc, roc_auc_score
import shap
import lightgbm as lgb
from scipy.special import logit
from Zee_functions_HKLE import h5ToDf, accuracy, HyperOpt_RandSearch, auc_eval, HeatMap_rand, PlotRandomSearch




# Logging style and level
log.basicConfig(format='[%(levelname)s] %(message)s', level=log.INFO)
log.info("Packages imported")

# Start "timer"
t_start = time()


# Command line options
parser = argparse.ArgumentParser(description="Combine datafiles, reweigh data and add columns.")
parser.add_argument('--outdir', action='store', default="output/phoISOModels/", type=str,
                    help='Output directory.')
parser.add_argument('path', type=str, nargs='+',
                    help='HDF5 file(s) to use for ISO.')
parser.add_argument('--tag', action='store', type=str, required=False, default="",
                    help='Tag the data category (Zee, Wev, etc.).')
parser.add_argument('--njobs', action='store', default=10, type=int,
                    help='Maximum number of concurrent processes to use.')
parser.add_argument('--hypopt', action='store', default=0, type=int,
                    help='Should hyperparameters be optimized?')
parser.add_argument('--hypParam', action='store', type=str, required=False, default="",
                    help='Input hyperparameters: What directory to load hyperparameters from, used in combination with --hypopt = 2.')

# parser.add_argument('--conv', action='store', default=1, type=int,
#                     help='Should the training run on converted or unconverted photons?')


args = parser.parse_args()

# Validate arguments
if not args.path:
    log.error("No HDF5 file was specified.")
    quit()

if args.njobs > 20:
    log.error("The requested number of jobs ({}) is excessive (>20). Exiting.".format(args.njobs))
    quit()

if ( (args.hypopt == 2) & (args.hypParam == "")):
    log.error("Path to input hyperparameters missing (--hypParam)")
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
#
# if (args.conv == 0):
#     log.info(f"Creating model for unconverted photons")
# elif (args.conv == 1):
#     log.info(f"Creating model for converted photons")
# else:
#     log.error("--conv should be zero or one. Exiting.")
#     quit()

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
#data_get = h5ToDf("/Users/sda/hep/work/Zmm model/PID_ISO_models/output/ISOReweightFiles/110820_ZbbW/combined_110820_train.h5")
# data_get = data_get[data_get["pho_ConversionType"] == args.conv]

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
'correctedScaledActualMu',
'NvtxReco',
'pho_et',
'pho_topoetcone20',
'pho_topoetcone40',
'pho_ptvarcone20'
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
print("The dataset is created with the weightEst10")

train_dataset = lgb.Dataset(X_train, label=y_train, weight=data_train["weightEst10"])
valid_dataset = lgb.Dataset(X_valid, label=y_valid, weight=data_valid["weightEst10"], reference=train_dataset)



params_set = {
              'boosting_type': 'gbdt',        # Default gbdt (traditional Gradient Boosting Decision Tree)
              'objective': 'binary',          # Probability labeĺs in [0,1]
              'boost_from_average': True,
              'verbose': 0,                   # < 0: Fatal, = 0: Error (Warning), = 1: Info, > 1: Debug
              'num_threads': args.njobs,
              }


hyperopt_options = {0 : "Use standard hyperparameters",
                    1 : "Run hyperparameter optimization",
                    2 : f"Use preoptimized hyperparameters from {args.hypParam}"}


#============================================================================
# Hyperparameter optimization
#============================================================================

if (args.hypopt == 0):
    print("USING STANDARD HYPERPARAMETERS ...")
    # header(f"Hyper Parameter Tuning: {hyperopt_options[args.hypopt]}")
    params_add = {
        'num_leaves': 30,
        'max_depth': -1,
        'min_data_in_leaf': 30,
        'feature_fraction': 1.0,
        'bagging_fraction': 1.0,
        'bagging_freq': 0}

elif (args.hypopt == 1):
    print("RUNNING HYPERPARAMETER OPTIMIZATION ...")

    t = time()

    # Set ranges for random search
    n_iter = 20
    n_fold = 5
    l_rate = 0.1
    n_boost_round = 500
    e_stop_round = 100
    params_grid = {
        'num_leaves': randint(20,40),                    # Important! Default: 31, set to less than 2^(max_depth)
        'max_depth': randint(-20,20),                       # Important! Default: -1, <= 0 means no limit
        'min_data_in_leaf': randint(10,100),              # Important! Default: 20
        'feature_fraction': [1.0],                        # Default: 1.0, random selection if it is under 1 eg. 80% of features for 0.8, helps with: speed up, over-fitting
        'bagging_fraction': [1.0],                        # Default: 1.0
        'bagging_freq': [0],                              # Default: 0, bags data, should be combined vith bagging_fraction
        }

    # Perform random search
    best, df_cv = HyperOpt_RandSearch(train_dataset,
                             params_set,
                             params_grid,
                             learning_rate=l_rate,
                             n_iter=n_iter,
                             n_fold=n_fold,
                             n_boost_round=n_boost_round,
                             e_stop_round=e_stop_round,
                             verbose=False)

    params_add = df_cv['params'][best]
    print(f"Best iteration: [{best}/{n_iter}]")

    # Save best hyperparameters
    print('Saving best hyperparameters...')
    np.save(args.outdir + "hypParam" + args.tag + "_best.npy", params_add)

    # Save all hyperparameters
    print('Saving best hyperparameters...')
    params_return = np.array(df_cv[['cv_mean', 'cv_stdv', 'num_boost_round','num_leaves','min_data_in_leaf','max_depth']])
    np.savetxt(args.outdir + "hypParam" + args.tag + ".txt", params_return)

    # Plot all hyperparameters
    print('Plotting hyperparameters...')
    rand_param = df_cv[['cv_mean', 'cv_stdv', 'num_boost_round','num_leaves','min_data_in_leaf','max_depth']]

    range_num_leaves = np.arange(20, 41, 1).astype(int)
    range_min_data_in_leaf = np.arange(10, 101, 1).astype(int)
    range_max_depth = np.arange(-20,21,1).astype(int)

    # fig, ax = plt.subplots(1,2,figsize=(7,5),sharey=True)
    # ax = ax.flatten()
    # HeatMap_rand(fig,ax[0],rand_param["cv_mean"],rand_param["cv_stdv"],rand_param["min_data_in_leaf"],rand_param["num_leaves"], range_min_data_in_leaf, range_num_leaves)
    # HeatMap_rand(fig,ax[1],rand_param["cv_mean"],rand_param["cv_stdv"],rand_param["max_depth"],rand_param["num_leaves"], range_max_depth, range_num_leaves)
    # fig.suptitle("Random search - negative AUC")
    # ax[0].set_ylabel("num_leaves")
    # ax[0].set_xlabel("min_data_in_leaf")
    # ax[1].set_xlabel("max_depth")
    #
    # fig.tight_layout(h_pad=0.3, w_pad=0.3)
    fig = PlotRandomSearch(rand_param["cv_mean"],rand_param["cv_stdv"], rand_param["min_data_in_leaf"], rand_param["max_depth"], rand_param["num_leaves"], range_min_data_in_leaf, range_max_depth, range_num_leaves)
    fig.savefig(args.outdir + "RandSearch.png")


elif (args.hypopt == 2):
    print("IMPORTING HYPERPARAMETERS ...")

    print(f"Importing parameters from: {args.hypParam}")
    params_add = np.load( args.hypParam, allow_pickle='TRUE').item()

params = params_set
params.update(params_add)

print(f"Parameters:")
keys = list(params.keys())
for i in range(len(keys)):
    print(f"        {keys[i]}: {params[keys[i]]}")

#============================================================================
# Train the model
#============================================================================

print(f"Training...")

num_round=50
bst = lgb.train(params, train_dataset, num_round)

print('Saving model...')
bst.save_model(args.outdir + "lgbm_phoISO" + ".txt")

#============================================================================
# Predict
#============================================================================

y_pred_train = bst.predict(X_train, num_iteration=bst.best_iteration)
y_pred_valid = bst.predict(X_valid, num_iteration=bst.best_iteration)

# print('AUC score of prediction:')
# print(f"        Training:   {roc_auc_score(y_train, y_pred_train, sample_weight = data_train["weightEst10"])}")
# print(f"        Validation: {roc_auc_score(y_valid, y_pred_valid, sample_weight = data_valid["weightEst10"])}")
#print('AUC score of prediction (weighted):')
#print(f"        Training:   {roc_auc_score(y_train, y_pred_train, sample_weight=data_train["weight"]):.6f}")
#print(f"        Validation: {roc_auc_score(y_valid, y_pred_valid, sample_weight=data_valid["weight"]):.6f}")



#============================================================================
# Plotting ROC curve
#============================================================================


fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_pred_train, sample_weight=data_train["weightEst10"])
auc_train = auc(fpr_train, tpr_train)

fpr_valid, tpr_valid, thresholds_valid = roc_curve(y_valid, y_pred_valid, sample_weight=data_valid["weightEst10"])
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
ax.set_yscale('log')
fig.tight_layout()
fig.savefig(args.outdir + "LGBM_valid" + ".pdf")
plt.close()

#============================================================================
# Plotting SHAP values
#============================================================================

shap_values_train = shap.TreeExplainer(bst).shap_values(X_train)

shap_values_df = pd.DataFrame(shap_values_train[0], columns = training_var)
shap_val = shap_values_df.abs().mean(0)
shap_val_sort = shap_val.sort_values(0, ascending = False)

# Get names of values and positions for plot
shap_name_sort = list(shap_val_sort.index.values)
shap_pos = np.arange(len(shap_name_sort))

# Plot shap values
print('Plotting SHAP values for training set...')
fig_shap, ax_shap = plt.subplots(figsize=(6,3))
ax_shap.barh(shap_pos, shap_val_sort, align='center')
ax_shap.set_yticks(shap_pos)
ax_shap.set_yticklabels(shap_name_sort)
ax_shap.invert_yaxis()  # labels read top-to-bottom
ax_shap.set_xlabel('mean(|SHAP|)')
#ax_shap.set_title('SHAP values - Average impact on model output')
plt.tight_layout()
fig_shap.savefig(args.outdir + 'featureImportance_SHAP_train' + '.pdf')

shap_values_valid = shap.TreeExplainer(bst).shap_values(X_valid)

shap_values_df = pd.DataFrame(shap_values_valid[0], columns = training_var)
shap_val = shap_values_df.abs().mean(0)
shap_val_sort = shap_val.sort_values(0, ascending = False)

# Get names of values and positions for plot
shap_name_sort = list(shap_val_sort.index.values)
shap_pos = np.arange(len(shap_name_sort))

# Plot shap values
print('Plotting SHAP values for validation set...')
fig_shap, ax_shap = plt.subplots(figsize=(6,3))
ax_shap.barh(shap_pos, shap_val_sort, align='center')
ax_shap.set_yticks(shap_pos)
ax_shap.set_yticklabels(shap_name_sort)
ax_shap.invert_yaxis()  # labels read top-to-bottom
ax_shap.set_xlabel('mean(|SHAP|)')
#ax_shap.set_title('SHAP values - Average impact on model output')
plt.tight_layout()
fig_shap.savefig(args.outdir + 'featureImportance_SHAP_valid' + '.pdf')


log.info(f"Done! Total time: {timedelta(seconds=time() - t_start)}")
