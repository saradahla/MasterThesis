'pt'#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Tuesday 30 June
@author: Sara Dahl Andersen

Reweighing the data and adding the PID and ISO scores to the data

nohup python -u Reweighing.py --tag 160920  output/MuoPairHdf5/160920/160920.h5 2>&1 &> output/logZReweight.txt & disown
nohup python -u Reweighing.py --tag 170920  output/MuoPairHdf5/170920/170920.h5 2>&1 &> output/logZReweight170920.txt & disown
nohup python -u Reweighing.py --tag 141020_MC  output/MuoPairHdf5/131020_MC/131020_MC.h5 2>&1 &> output/logZReweight141020.txt & disown

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
import lightgbm as lgb
from scipy.special import logit

# Logging style and level
log.basicConfig(format='[%(levelname)s] %(message)s', level=log.INFO)
log.info("Packages imported")

# Start "timer"
t_start = time()


# Command line options
parser = argparse.ArgumentParser(description="Combine datafiles, reweigh data and add columns.")
parser.add_argument('--outdir', action='store', default="output/ZReweightFiles/", type=str,
                    help='Output directory.')
parser.add_argument('paths', type=str, nargs='+',
                    help='HDF5 file(s) to reweight.')
parser.add_argument('--testSize', action='store', default=0.2, type=float,
                    help='Size of test set from 0.0 to 1.0. (Default = 0.2)')
parser.add_argument('--validSize', action='store', default=0.2, type=float,
                    help='Size of validation set from 0.0 to 1.0. Split after separation into test and training set. (Default = 0.2)')
parser.add_argument('--tag', action='store', type=str, required=False, default="",
                    help='Tag the data category (Zee, Wev, etc.).')
parser.add_argument('--max-processes', action='store', default=10, type=int,
                    help='Maximum number of concurrent processes to use.')


args = parser.parse_args()

# Validate arguments
if not args.paths:
    log.error("No HDF5 file was specified.")
    quit()

if args.max_processes > 20:
    log.error("The requested number of processes ({}) is excessive (>20). Exiting.".format(args.max_processes))
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

modelPID = "/groups/hep/sda/work/Data model/PID_ISO_Models/output/PIDModels/160920_Data/lgbmPID.txt"
modelPID8 = "/groups/hep/sda/work/Data model/PID_ISO_Models/output/PIDModels/160920_Data_8vars/lgbmPID_8vars.txt"
modelPID6 = "/groups/hep/sda/work/Data model/PID_ISO_Models/output/PIDModels/170920_Data_6vars/lgbmPID_6vars.txt"
modelISO = "/groups/hep/sda/work/Data model/PID_ISO_Models/output/ISOModels/110920_Data/lgbmISO.txt"

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

def getRegularWeights(datatype, reweighter, data):
    # Inspired by reweightFile.py from Dnielsen
    # predict the weights
    total_weight = reweighter.predict_weights(np.array([data['eta'][data['label'] < 0.5],
                                                        data['pt'][data['label'] < 0.5],
                                                        data['invM'][data['label'] < 0.5],
                                                        data['correctedScaledAverageMu'][data['label'] < 0.5]]).T)
    log.info(f'Prediction of weights for {datatype} is done')

    # Get the ratio of sig and bkg weights to scale the bkg to have the same number of events ( after weighting )
    log.info(f"[Data]   shape: {np.shape(data['label'])}, sum = {np.sum(data['label'])}, sum[>=0.5] = {np.sum(data['label'] >= 0.5)}")
    log.info(f"[Weight] shape: {np.shape(total_weight)}, sum = {np.sum(total_weight)}")

    ratio = np.sum(data['label'] >= 0.5) / np.sum(total_weight)
    log.info(f"Ratio: {ratio}")
    if (datatype == "store"):
        # Stored data has no signal
        ratio = 1

    # Set array for weights to 1 (Signal gets weight 1)
    weight = np.ones(len(data['eta']))

    # Get weights for background
    weight[data['label'] < 0.5] = total_weight * ratio

    # Return the weights normalized to a mean of one, since this is how keras likes it
    return weight#, (weight / np.mean(weight))

def getReverseWeights(datatype, reweighter, data):
    # Inspired by reweightFile.py from Dnielsen
    # predict the weights
    total_weight = reweighter.predict_weights(np.array([data['eta'][data['label'] > 0.5],
                                                        data['pt'][data['label'] > 0.5],
                                                        data['invM'][data['label'] > 0.5],
                                                        data['correctedScaledAverageMu'][data['label'] > 0.5]]).T)
    log.info(f'Prediction of weights for {datatype} is done')

    # Get the ratio of sig and bkg weights to scale the signal to have the same number of events ( after weighting )
    log.info(f"[Data]   shape: {np.shape(data['label'])}, sum = {np.sum(data['label'])}, sum[<=0.5] = {np.sum(data['label'] <= 0.5)}")
    log.info(f"[Weight] shape: {np.shape(total_weight)}, sum = {np.sum(total_weight)}")

    ratio = np.sum(data['label'] <= 0.5) / np.sum(total_weight)
    log.info(f"Ratio: {ratio}")
    if (datatype == "store"):
        # Stored data has no signal
        ratio = 1

    # Set array for weights to 1 (Background gets weight 1)
    weight = np.ones(len(data['eta']))

    # Get weights for signal
    weight[data['label'] > 0.5] = total_weight * ratio

    # Return the weights normalized to a mean of one, since this is how keras likes it
    return weight#, (weight / np.mean(weight))

def GetISOscore(gbm, data, muoNr):
    training_var = [f'muo{muoNr}_etcone20',
                    f'muo{muoNr}_ptcone20',
                    f'muo{muoNr}_pt',
                    f'muo{muoNr}_etconecoreConeEnergyCorrection',
                    f'muo{muoNr}_neflowisolcoreConeEnergyCorrection',
                    f'muo{muoNr}_ptconecoreTrackPtrCorrection',
                    f'muo{muoNr}_topoetconecoreConeEnergyCorrection']
    score = gbm.predict(data[training_var], n_jobs=args.max_processes)
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

    score = gbm.predict(data[training_var], n_jobs=args.max_processes)
    return logit(score)

def GetPID8score(gbm, data, muoNr):
    training_var = [#f'muo{muoNr}_numberOfPrecisionLayers',
                    f'muo{muoNr}_numberOfPrecisionHoleLayers',
                    #f'muo{muoNr}_quality',
                    f'muo{muoNr}_ET_TileCore',
                    #f'muo{muoNr}_MuonSpectrometerPt',
                    f'muo{muoNr}_deltatheta_1',
                    f'muo{muoNr}_scatteringCurvatureSignificance', # PID
                    f'muo{muoNr}_scatteringNeighbourSignificance', # PID
                    f'muo{muoNr}_momentumBalanceSignificance', # PID
                    f'muo{muoNr}_EnergyLoss', # PID
                    f'muo{muoNr}_energyLossType']

    score = gbm.predict(data[training_var], n_jobs=args.max_processes)
    return logit(score)

def GetPID6score(gbm, data, muoNr):
    training_var = [#f'muo{muoNr}_numberOfPrecisionLayers',
                    # f'muo{muoNr}_numberOfPrecisionHoleLayers',
                    #f'muo{muoNr}_quality',
                    f'muo{muoNr}_ET_TileCore',
                    #f'muo{muoNr}_MuonSpectrometerPt',
                    # f'muo{muoNr}_deltatheta_1',
                    f'muo{muoNr}_scatteringCurvatureSignificance', # PID
                    f'muo{muoNr}_scatteringNeighbourSignificance', # PID
                    f'muo{muoNr}_momentumBalanceSignificance', # PID
                    f'muo{muoNr}_EnergyLoss', # PID
                    f'muo{muoNr}_energyLossType']

    score = gbm.predict(data[training_var], n_jobs=args.max_processes)
    return logit(score)

# ================================================ #
#                End of functions                  #
# ================================================ #

# Data
filenames = []
data_name = []
data_list = []
fname = "combined_"
for path in args.paths:
    # Name of data file
    filename_base = os.path.basename(path)
    filename = os.path.splitext(filename_base)[0]
    filenames.append(filename)

    # Name of process
    name = filename.split("_")
    data_name.append(name[0])

    # Data
    data_get = h5ToDf(path)
    data_list.append(data_get)

    # Combine names for new filename
    fname = fname + name[0]

data_all = pd.concat(data_list, ignore_index=True)
data_all = data_all.sample(frac=1, random_state=0).reset_index(drop=True) # Shuffle
data = data_all.copy()

# Add label
log.info(f"Add label to data")
#data = data.drop(data[data["type"] == 2].index, axis = 0) #removes all type 2 muons (opposite sign, wrong mass)
data["label"] = 0
data.loc[data["type"]==1,"label"] = 1
#data.loc[data["type"]==2,"label"] = 2 #to remove trash from background

# Check shapes
shapeAll = np.shape(data_all)
shapeSig = np.shape(data[data["label"] == 1])
shapeBkg = np.shape(data[data["label"] == 0])

log.info(f"Shape all:        {shapeAll}")
log.info(f"Shape signal:     {shapeSig}")
log.info(f"Shape background: {shapeBkg}")


#============================================================================
# Add columns
#============================================================================

# Add Z_sig
data['Z_sig'] = (data['muo1_priTrack_z0'] - data['muo2_priTrack_z0'])/np.sqrt(data['muo1_priTrack_z0Sig']**2 + data['muo2_priTrack_z0Sig']**2)
data['muo1_d0_d0Sig'] = data['muo1_priTrack_d0']/data['muo1_priTrack_d0Sig']
data['muo2_d0_d0Sig'] = data['muo2_priTrack_d0']/data['muo2_priTrack_d0Sig']

# Add ML isolation
log.info(f"Add ML models score to data")
log.info(f"        Muon models: {modelPID}")
log.info(f"                     {modelPID8}")
log.info(f"                     {modelISO}")

PIDmod = lgb.Booster(model_file = modelPID)
PIDmod8 = lgb.Booster(model_file = modelPID8)
PIDmod6 = lgb.Booster(model_file = modelPID6)
ISOmod = lgb.Booster(model_file = modelISO)

data['muo1_PID_score'] = GetPIDscore(PIDmod,data,1)
data['muo2_PID_score'] = GetPIDscore(PIDmod,data,2)

data['muo1_PID_score_ATLAS'] = GetPID8score(PIDmod8,data,1)
data['muo2_PID_score_ATLAS'] = GetPID8score(PIDmod8,data,2)

data['muo1_PID_score_ATLAS6'] = GetPID6score(PIDmod6,data,1)
data['muo2_PID_score_ATLAS6'] = GetPID6score(PIDmod6,data,2)

data['muo1_ISO_score'] = GetISOscore(ISOmod,data,1)
data['muo2_ISO_score'] = GetISOscore(ISOmod,data,2)


#============================================================================
# Split in train, valid and test set
#============================================================================
log.info(f"Split data in training and test with split: {args.testSize}")
data_train, data_test = train_test_split(data, test_size=args.testSize, random_state=0)

TrainNSig = np.shape(data_train[data_train['label']==1])[0]
TrainNBkg = np.shape(data_train[data_train['label']==0])[0]
TestNSig = np.shape(data_test[data_test['label']==1])[0]
TestNBkg = np.shape(data_test[data_test['label']==0])[0]

log.info(f"        Shape of training data:  {np.shape(data_train)}")
log.info(f"                Signal:          {TrainNSig} ({( (TrainNSig) / (TrainNSig+TrainNBkg) )*100:.2f}%)")
log.info(f"                Background:      {TrainNBkg} ({( (TrainNBkg) / (TrainNSig+TrainNBkg) )*100:.2f}%)")
log.info(f"        Shape of test data:      {np.shape(data_test)}")
log.info(f"                Signal:          {TestNSig} ({( (TestNSig) / (TestNSig+TestNBkg) )*100:.2f}%)")
log.info(f"                Background:      {TestNBkg} ({( (TestNBkg) / (TestNSig+TestNBkg) )*100:.2f}%)")

# Copy data to avoid SettingWithCopyWarning
data_train = data_train.copy()
data_test = data_test.copy()

# Set dataset type: 0 = train, 1 = valid, 2 = test, 3 = store
datatype = {0 : "train",
            1 : "valid",
            2 : "test",
            3 : "store"}

# Set dataset
data_test["dataset"] = 2

# Split training data into train and valid
log.info(f"Split training data in training and validation with split: {args.validSize}")

data_train["dataset"] = 0
data_train.loc[data_train.sample(frac = args.validSize, random_state=3).index,"dataset"] = 1 #

# Create masks
trainMask = (data_train["dataset"] == 0)
validMask = (data_train["dataset"] == 1)

trainNSig = np.shape(data_train[trainMask & (data_train['label']==1)])[0]
trainNBkg = np.shape(data_train[trainMask & (data_train['label']==0)])[0]
validNSig = np.shape(data_train[validMask & (data_train['label']==1)])[0]
validNBkg = np.shape(data_train[validMask & (data_train['label']==0)])[0]

# Print
log.info(f"        Shape of training set:   {np.shape(data_train[trainMask])}")
log.info(f"                Signal:          {trainNSig} ({( (trainNSig) / (trainNSig+trainNBkg) )*100:.2f}%)")
log.info(f"                Background:      {trainNBkg} ({( (trainNBkg) / (trainNSig+trainNBkg) )*100:.2f}%)")
log.info(f"        Shape of validation set: {np.shape(data_train[validMask])}")
log.info(f"                Signal:          {validNSig} ({( (validNSig) / (validNSig+validNBkg) )*100:.2f}%)")
log.info(f"                Background:      {validNBkg} ({( (validNBkg) / (validNSig+validNBkg) )*100:.2f}%)")


#============================================================================
# Reweigh
#============================================================================
log.info(f"Reweigh background data using GBReweighter on training set")

reweightNames = ["nEst10", "nEst15", "nEst20", "nEst25", "nEst30"]#, "nEst100", "nEst200"]

#reweightNames = ["nEst10", "nEst40", "nEst100", "nEst200"]

# Set parameters: Default {'n_estimators' : 40, 'learning_rate' : 0.2, 'max_depth' : 3, 'min_samples_leaf' : 200, 'loss_regularization' : 5.0}
# reweightParams = [ {'n_estimators' : 10, 'learning_rate' : 0.2, 'max_depth' : 3, 'min_samples_leaf' : 200, 'loss_regularization' : 5.0 },
#                    {'n_estimators' : 40, 'learning_rate' : 0.2, 'max_depth' : 3, 'min_samples_leaf' : 200, 'loss_regularization' : 5.0 },
#                    {'n_estimators' : 100, 'learning_rate' : 0.2, 'max_depth' : 3, 'min_samples_leaf' : 200, 'loss_regularization' : 5.0 },
#                    {'n_estimators' : 200, 'learning_rate' : 0.2, 'max_depth' : 3, 'min_samples_leaf' : 200, 'loss_regularization' : 5.0 }
#                  ]
reweightParams = [ {'n_estimators' : 10, 'learning_rate' : 0.2, 'max_depth' : 3, 'min_samples_leaf' : 200, 'loss_regularization' : 5.0 },
                   {'n_estimators' : 15, 'learning_rate' : 0.2, 'max_depth' : 3, 'min_samples_leaf' : 200, 'loss_regularization' : 5.0 },
                   {'n_estimators' : 20, 'learning_rate' : 0.2, 'max_depth' : 3, 'min_samples_leaf' : 200, 'loss_regularization' : 5.0 },
                   {'n_estimators' : 25, 'learning_rate' : 0.2, 'max_depth' : 3, 'min_samples_leaf' : 200, 'loss_regularization' : 5.0 },
                   {'n_estimators' : 30, 'learning_rate' : 0.2, 'max_depth' : 3, 'min_samples_leaf' : 200, 'loss_regularization' : 5.0 }
                 ]

log.info(f"Regular reweights")
for iWeight, weightName in enumerate(reweightNames):
    t = time()
    # Print parameters
    log.info(f"Parameters for GBReweighter:")
    params = reweightParams[iWeight]
    for param in params:
        log.info(f"        {param} : {params[param]}")

    # Setup reweighter: https://arogozhnikov.github.io/hep_ml/reweight.html#
    reweighter  = GBReweighter(n_estimators=params['n_estimators'],
                               learning_rate=params['learning_rate'],
                               max_depth=params['max_depth'],
                               min_samples_leaf=params['min_samples_leaf'],
                               loss_regularization=params['loss_regularization'])

    # Create weight estimators and fit them to the data
    log.info(f"Fitting weights...")
    reweighter.fit(original = np.array([data_train['eta'][trainMask & (data_train["label"] < 0.5)],
                                        data_train['pt'][trainMask & (data_train["label"] < 0.5)],
                                        data_train['invM'][trainMask & (data_train["label"] < 0.5)],
                                        data_train['correctedScaledAverageMu'][trainMask & (data_train["label"] < 0.5)]]).T,
                   target   = np.array([data_train['eta'][trainMask & (data_train["label"] >= 0.5)],
                                        data_train['pt'][trainMask & (data_train["label"] >= 0.5)],
                                        data_train['invM'][trainMask & (data_train["label"] >= 0.5)],
                                        data_train['correctedScaledAverageMu'][trainMask & (data_train["label"] >= 0.5)]]).T)
    log.info(f"Fitting of weights is done (time: {timedelta(seconds=time() - t)})")

    # Get weights
    log.info(f"Get weights for training, validation and test set")
    weight_train = getRegularWeights("train", reweighter, data_train[trainMask])
    weight_valid = getRegularWeights("valid", reweighter, data_train[validMask])
    weight_test  = getRegularWeights("test",  reweighter, data_test)

    # Add weights to data
    log.info(f"Add weights for training, validation and test set to data")
    data_train["regWeight_"+weightName] = 0
    data_train.loc[trainMask,"regWeight_"+weightName] = weight_train
    data_train.loc[validMask,"regWeight_"+weightName] = weight_valid
    data_test["regWeight_"+weightName] = weight_test


#
# log.info(f"Reverse reweights")
# for iWeight, weightName in enumerate(reweightNames):
#     t = time()
#     # Print parameters
#     log.info(f"Parameters for GBReweighter:")
#     params = reweightParams[iWeight]
#     for param in params:
#         log.info(f"        {param} : {params[param]}")
#
#     # Setup reweighter: https://arogozhnikov.github.io/hep_ml/reweight.html#
#     reweighter  = GBReweighter(n_estimators=params['n_estimators'],
#                                learning_rate=params['learning_rate'],
#                                max_depth=params['max_depth'],
#                                min_samples_leaf=params['min_samples_leaf'],
#                                loss_regularization=params['loss_regularization'])
#
#     # Create weight estimators and fit them to the data
#     log.info(f"Fitting weights...")
#     reweighter.fit(original = np.array([data_train['eta'][trainMask & (data_train["label"] > 0.5)],
#                                         data_train['pt'][trainMask & (data_train["label"] > 0.5)],
#                                         data_train['invM'][trainMask & (data_train["label"] > 0.5)],
#                                         data_train['correctedScaledAverageMu'][trainMask & (data_train["label"] > 0.5)]]).T,
#                    target   = np.array([data_train['eta'][trainMask & (data_train["label"] <= 0.5)],
#                                         data_train['pt'][trainMask & (data_train["label"] <= 0.5)],
#                                         data_train['invM'][trainMask & (data_train["label"] <= 0.5)],
#                                         data_train['correctedScaledAverageMu'][trainMask & (data_train["label"] <= 0.5)]]).T)
#     log.info(f"Fitting of weights is done (time: {timedelta(seconds=time() - t)})")
#
#     # Get weights
#     log.info(f"Get weights for training, validation and test set")
#     weight_train = getReverseWeights("train", reweighter, data_train[trainMask])
#     weight_valid = getReverseWeights("valid", reweighter, data_train[validMask])
#     weight_test  = getReverseWeights("test",  reweighter, data_test)
#
#     # Add weights to data
#     log.info(f"Add weights for training, validation and test set to data")
#     data_train["revWeight_"+weightName] = 0
#     data_train.loc[trainMask,"revWeight_"+weightName] = weight_train
#     data_train.loc[validMask,"revWeight_"+weightName] = weight_valid
#     data_test["revWeight_"+weightName] = weight_test
#


#============================================================================
# Save to hdf5
#============================================================================
column_names = data_train.columns
filename_train = args.outdir+fname+"_train.h5"
filename_test = args.outdir+fname+"_test.h5"

log.info("Saving training data to {}".format(filename_train))
with h5py.File(filename_train, 'w') as hf:
    for var in column_names:
        hf.create_dataset( f'{var}', data=np.array(data_train[var]), chunks=True, maxshape= (None,), compression='lzf')

log.info("Saving test data to {}".format(filename_test))
with h5py.File(filename_test, 'w') as hf:
    for var in column_names:
        hf.create_dataset( f'{var}', data=np.array(data_test[var]), chunks=True, maxshape= (None,), compression='lzf')

#============================================================================
# Plot reweighted data
#============================================================================
# weights_train = [weight_trainEst10, weight_trainEst20, weight_trainEst7, weight_trainEst5]
# weights_train_names = ["weight_trainEst10", "weight_trainEst20", "weight_trainEst7", "weight_trainEst5"]

#reweightNames = ["nEst10", "nEst40", "nEst100", "nEst200"]

masks = [trainMask, validMask]
maskNames = ["train", "valid"]
maskLabel = ["Training set", "Validation set"]
variables = ["eta", "pt", "invM", "correctedScaledAverageMu"]
bins = [120, 120, 120, 80]
ranges = [(-4, 4), (-5, 120), (50, 110), (-2, 80)]
xlabel = [r"$\eta$", "pt", "invM", r"$\langle\mu\rangle$"]

weightTypes = ["regWeight"]#, "revWeight"]
weightTypeNames = ["regular"]#, "reverse"]

#weightLinestyle = ['dotted', 'dashed', 'dashdot','solid']
#
# for iMask, mask in enumerate(masks):
#     for iType, weightType in enumerate(weightTypes):
#         for iWeight, weightName in enumerate(reweightNames):
#             fig, ax = plt.subplots(2,3,figsize=(20,10))
#             ax = ax.flatten()
#
#             if weightType == "revWeight": #here we have the signal reweighted
#                 for i, (var, bin, rang, xlab) in enumerate(zip(variables, bins, ranges, xlabel)):
#                     counts_sig, edges_sig = np.histogram(data_train[var][mask][data_train["label"][mask]>0.5], bins=bin, range=rang)
#                     counts_bkg, edges_bkg = np.histogram(data_train[var][mask][data_train["label"][mask]<0.5], bins=bin, range=rang)
#                     counts_sigrw, edges_sigrw = np.histogram(data_train[var][mask][data_train["label"][mask]>0.5], bins=bin, weights = data_train[weightType+"_"+weightName][mask][data_train["label"][mask] > 0.5], range=rang)
#
#                     bw = edges_sig[1] - edges_sig[0]
#
#                     ax[i].step(x=edges_sig, y=np.append(counts_sig, 0), where="post", color = "tab:purple", alpha = 1, label = "Signal");
#                     ax[i].step(x=edges_bkg, y=np.append(counts_bkg, 0), where="post", color = "g", alpha = 1, label = "Background");
#                     ax[i].step(x=edges_sigrw, y=np.append(counts_sigrw, 0), where="post", color = "k", linestyle = 'dashed', alpha = 1, label = "Signal reweighted");
#
#                     ax[i].set(xlim = (edges_sig[0], edges_sig[-1]), xlabel = xlab, ylabel = f"Events/{bw:4.2f}");
#
#                     ax[i].legend()
#                     for item in ([ax[i].title, ax[i].xaxis.label, ax[i].yaxis.label] +
#                                  ax[i].get_xticklabels() + ax[i].get_yticklabels()):
#                         item.set_fontsize(15)
#
#
#             else:
#                 for i, (var, bin, rang, xlab) in enumerate(zip(variables, bins, ranges, xlabel)):
#
#                     fig, ax[i] = Plot(Histogram(data_train[var][mask], data_train["label"][mask], data_train[weightType+"_"+weightName][mask], bin, rang[0], rang[1]), fig, ax[i], xlab, includeN = True)
#                     #fig, ax[1] = Plot(Histogram(data_train['pt'][mask], data_train["label"][mask], data_train[weightType+"_"+weightName][mask], 120, -5, 120), fig, ax[1], "pt", includeN = False)
#                     #fig, ax[2] = Plot(Histogram(data_train['invM'][mask], data_train["label"][mask], data_train[weightType+"_"+weightName][mask], 120, 50, 110), fig, ax[2], "invM", includeN = False)
#                     #fig, ax[3] = Plot(Histogram(data_train['correctedScaledAverageMu'][mask], data_train["label"][mask], data_train[weightType+"_"+weightName][mask], 80, -2, 80), fig, ax[3], r"$\langle\mu\rangle$", includeN = False)
#
#                     for item in ([ax[i].title, ax[i].xaxis.label, ax[i].yaxis.label] +
#                                  ax[i].get_xticklabels() + ax[i].get_yticklabels()):
#                         item.set_fontsize(15)
#             counts_weight, edges_weight = np.histogram(data_train[weightType+"_"+weightName][mask], bins=120, range=(0, 40))
#             ax[4].step(x=edges_weight, y=np.append(counts_weight, 0), where="post", color = "k");
#             ax[4].set_yscale('log', nonposy='clip')
#             ax[4].set(xlabel = weightName, ylabel = "Events per bin")
#             for item in ([ax[4].title, ax[4].xaxis.label, ax[4].yaxis.label] +
#                          ax[4].get_xticklabels() + ax[4].get_yticklabels()):
#                 item.set_fontsize(15)
#             fig.tight_layout()
#             fig.savefig(args.outdir + maskNames[iMask] + weightType + "_" + weightName + ".pdf")
#

# Test plotting


for iMask, mask in enumerate(masks):
    for iType, weightType in enumerate(weightTypes):
        for iWeight, weightName in enumerate(reweightNames):

            fig = plt.figure(figsize=(20,10))

            ax1 = fig.add_subplot(2,3,1)
            ax2 = fig.add_subplot(2,3,2)
            ax3 = fig.add_subplot(2,3,3)
            ax4 = fig.add_subplot(2,2,3)
            ax5 = fig.add_subplot(2,2,4)

            axs = [ax1, ax2, ax3, ax4, ax5]

            for i, (var, bin, rang, xlab, ax) in enumerate(zip(variables, bins, ranges, xlabel, axs)):

                fig, ax = Plot(Histogram(data_train[var][mask], data_train["label"][mask], data_train[weightType+"_"+weightName][mask], bin, rang[0], rang[1]), fig, ax, xlab, includeN = True)
                #fig, ax[1] = Plot(Histogram(data_train['pt'][mask], data_train["label"][mask], data_train[weightType+"_"+weightName][mask], 120, -5, 120), fig, ax[1], "pt", includeN = False)
                #fig, ax[2] = Plot(Histogram(data_train['invM'][mask], data_train["label"][mask], data_train[weightType+"_"+weightName][mask], 120, 50, 110), fig, ax[2], "invM", includeN = False)
                #fig, ax[3] = Plot(Histogram(data_train['correctedScaledAverageMu'][mask], data_train["label"][mask], data_train[weightType+"_"+weightName][mask], 80, -2, 80), fig, ax[3], r"$\langle\mu\rangle$", includeN = False)

                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(20)
            counts_weight, edges_weight = np.histogram(data_train[weightType+"_"+weightName][mask], bins=120, range=(0, 40))
            ax5.step(x=edges_weight, y=np.append(counts_weight, 0), where="post", color = "k");
            ax5.set_yscale('log', nonposy='clip')
            ax5.set(xlabel = weightName, ylabel = "Events per bin")
            for item in ([ax5.title, ax5.xaxis.label, ax5.yaxis.label] +
                     ax5.get_xticklabels() + ax5.get_yticklabels()):
                item.set_fontsize(20)
            #fig.tight_layout()
            fig.savefig(args.outdir + maskNames[iMask] + weightType + "_" + weightName + ".pdf")




### Validation data
# weights_valid = [weight_validEst10, weight_validEst20, weight_validEst7, weight_validEst5]
# weights_valid_names = ["weight_validEst10", "weight_validEst20", "weight_validEst7", "weight_validEst5"]
#
# for w, wn in zip(weights_valid, weights_valid_names):
#     fig, ax = plt.subplots(2,3,figsize=(15,10))
#     ax = ax.flatten()
#     fig, ax[0] = Plot(Histogram(data_train['eta'][validMask], data_train["label"][validMask], w, 120, -4, 4), fig, ax[0], r"$\eta$", includeN = True)
#     fig, ax[1] = Plot(Histogram(data_train['pt'][validMask], data_train["label"][validMask], w, 120, -5, 120), fig, ax[1], "pt", includeN = False)
#     fig, ax[2] = Plot(Histogram(data_train['invM'][validMask], data_train["label"][validMask], w, 120, 50, 110), fig, ax[2], "invM", includeN = False)
#     fig, ax[3] = Plot(Histogram(data_train['correctedScaledAverageMu'][validMask], data_train["label"][validMask], w, 80, -2, 80), fig, ax[3], r"$\langle\mu\rangle$", includeN = False)
#
#     counts_weight, edges_weight = np.histogram(w, bins=120, range=(0, 40))
#     ax[4].step(x=edges_weight, y=np.append(counts_weight, 0), where="post", color = "k");
#     ax[4].set_yscale('log', nonposy='clip')
#     ax[4].set(xlabel = wn, ylabel = "Events per bin")
#     fig.savefig(args.outdir + "valid_reweighted" + wn + ".pdf")


log.info(f"Done! Total time: {timedelta(seconds=time() - t_start)}")
