#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Tuesday 30 June
@author: Sara Dahl Andersen

Reweighing the photon file

nohup python -u phoReweight.py --tag 20201027_cutOrig  output/pho_Dataset/20201027/20201027.h5 2>&1 &> output/logPhoReweight.txt & disown
nohup python -u phoReweight.py --tag 20201027_cutOrig2  output/pho_Dataset/20201027_2/20201027_2.h5 2>&1 &> output/logPhoReweight2.txt & disown

nohup python -u phoReweight.py --tag 20201028  output/pho_Dataset/20201028/20201028.h5 2>&1 &> output/logPhoReweight.txt & disown

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


# Logging style and level
log.basicConfig(format='[%(levelname)s] %(message)s', level=log.INFO)
log.info("Packages imported")

# Start "timer"
t_start = time()


# Command line options
parser = argparse.ArgumentParser(description="Combine datafiles, reweigh data and add columns.")
parser.add_argument('--outdir', action='store', default="output/phoReweightFiles/", type=str,
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

def getWeights(datatype, reweighter, data):
    # Inspired by reweightFile.py from Dnielsen
    # predict the weights
    total_weight = reweighter.predict_weights(np.array([data['pho_eta'][data['label'] < 0.5],
                                                        data['pho_et'][data['label'] < 0.5],
                                                        data['correctedScaledActualMu'][data['label'] < 0.5]]).T)
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
    weight = np.ones(len(data['pho_eta']))

    # Get weights for background
    weight[data['label'] < 0.5] = total_weight * ratio

    # Return the weights normalized to a mean of one, since this is how keras likes it
    return weight#, (weight / np.mean(weight))


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

# Cut all data below 4.5 GeV
# log.info(f"Cut all data below 4.5 GeV")
# data = data[data["pho_et"] > 4.5]

# Add label isolated photons are signal
log.info(f"Add label to data")
data["label"] = 0
# data.loc[((data["pho_truthOrigin"] == 3) & (data["pho_et"] > 4.5)), "label"] = 1
data.loc[((data["pho_truthType"] == 14) & (data["pho_et"] > 4.5)), "label"] = 1

# Check shapes
shapeAll = np.shape(data_all)
shapeSig = np.shape(data[data["label"] == 1])
shapeBkg = np.shape(data[data["label"] == 0])

log.info(f"Shape all:        {shapeAll}")
log.info(f"Shape signal:     {shapeSig}")
log.info(f"Shape background: {shapeBkg}")


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
t = time()
# Create weight estimators and fit them to the data

reweighterEst10  = GBReweighter(n_estimators=10,
                           #learning_rate=params['learning_rate'],
                           max_depth=5,
                           #min_samples_leaf=params['min_samples_leaf'],
                           #loss_regularization=params['loss_regularization']
                           )
reweighterEst20  = GBReweighter(n_estimators=20,
                           #learning_rate=params['learning_rate'],
                           max_depth=5,
                           #min_samples_leaf=params['min_samples_leaf'],
                           #loss_regularization=params['loss_regularization']
                           )
reweighterEst40  = GBReweighter(n_estimators=40,
                           #learning_rate=params['learning_rate'],
                           max_depth=5,
                           #min_samples_leaf=params['min_samples_leaf'],
                           #loss_regularization=params['loss_regularization']
                           )
log.info(f"Fitting weights...")
reweighterEst10.fit(original = np.array([data_train['pho_eta'][trainMask & (data_train["label"] < 0.5)],
                                    data_train['pho_et'][trainMask & (data_train["label"] < 0.5)],
                                    data_train['correctedScaledActualMu'][trainMask & (data_train["label"] < 0.5)]]).T,
               target   = np.array([data_train['pho_eta'][trainMask & (data_train["label"] >= 0.5)],
                                    data_train['pho_et'][trainMask & (data_train["label"] >= 0.5)],
                                    data_train['correctedScaledActualMu'][trainMask & (data_train["label"] >= 0.5)]]).T)
reweighterEst20.fit(original = np.array([data_train['pho_eta'][trainMask & (data_train["label"] < 0.5)],
                                    data_train['pho_et'][trainMask & (data_train["label"] < 0.5)],
                                    data_train['correctedScaledActualMu'][trainMask & (data_train["label"] < 0.5)]]).T,
               target   = np.array([data_train['pho_eta'][trainMask & (data_train["label"] >= 0.5)],
                                    data_train['pho_et'][trainMask & (data_train["label"] >= 0.5)],
                                    data_train['correctedScaledActualMu'][trainMask & (data_train["label"] >= 0.5)]]).T)
reweighterEst40.fit(original = np.array([data_train['pho_eta'][trainMask & (data_train["label"] < 0.5)],
                                    data_train['pho_et'][trainMask & (data_train["label"] < 0.5)],
                                    data_train['correctedScaledActualMu'][trainMask & (data_train["label"] < 0.5)]]).T,
               target   = np.array([data_train['pho_eta'][trainMask & (data_train["label"] >= 0.5)],
                                    data_train['pho_et'][trainMask & (data_train["label"] >= 0.5)],
                                    data_train['correctedScaledActualMu'][trainMask & (data_train["label"] >= 0.5)]]).T)
log.info(f"Fitting of weights is done (time: {timedelta(seconds=time() - t)})")


log.info(f"Get weights for training, validation and test set")
weight_trainEst10 = getWeights("train", reweighterEst10, data_train[trainMask])
weight_trainEst20 = getWeights("train", reweighterEst20, data_train[trainMask])
weight_trainEst40 = getWeights("train", reweighterEst40, data_train[trainMask])

weight_validEst10 = getWeights("valid", reweighterEst10, data_train[validMask])
weight_validEst20 = getWeights("valid", reweighterEst20, data_train[validMask])
weight_validEst40 = getWeights("valid", reweighterEst40, data_train[validMask])

weight_testEst10  = getWeights("test",  reweighterEst10, data_test)
weight_testEst20  = getWeights("test",  reweighterEst20, data_test)
weight_testEst40  = getWeights("test",  reweighterEst40, data_test)

# Add weights to data
log.info(f"Add weights for training, validation and test set to data")
data_train["weightEst10"] = 0
data_train["weightEst20"] = 0
data_train["weightEst40"] = 0

data_train.loc[trainMask, "weightEst10"] = weight_trainEst10
data_train.loc[trainMask, "weightEst20"] = weight_trainEst20
data_train.loc[trainMask, "weightEst40"] = weight_trainEst40

data_train.loc[validMask, "weightEst10"] = weight_validEst10
data_train.loc[validMask, "weightEst20"] = weight_validEst20
data_train.loc[validMask, "weightEst40"] = weight_validEst40

data_test["weightEst10"] = weight_testEst10
data_test["weightEst20"] = weight_testEst20
data_test["weightEst40"] = weight_testEst40



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
weights_train = [weight_trainEst10, weight_trainEst20, weight_trainEst40]
weights_train_names = ["weight_trainEst10", "weight_trainEst20", "weight_trainEst40"]

### Training data
for w, wn in zip(weights_train, weights_train_names):
    fig, ax = plt.subplots(2,2,figsize=(15,10))
    ax = ax.flatten()
    fig, ax[0] = Plot(Histogram(data_train['pho_eta'][trainMask], data_train["label"][trainMask], w, 120, -4, 4), fig, ax[0], r"$\eta$", includeN = True)
    ax[0].set_yscale('log')
    fig, ax[1] = Plot(Histogram(data_train['pho_et'][trainMask], data_train["label"][trainMask], w, 90, 0, 50), fig, ax[1], r"$E_T$", includeN = False)
    ax[1].set_yscale('log')
    fig, ax[2] = Plot(Histogram(data_train['correctedScaledActualMu'][trainMask], data_train["label"][trainMask], w, 90, 0, 90), fig, ax[2], r"$\langle\mu\rangle$", includeN = False)
    ax[2].set_yscale('log')

    counts_weight, edges_weight = np.histogram(w, bins=120, range=(0, 40))
    ax[3].step(x=edges_weight, y=np.append(counts_weight, 0), where="post", color = "k");
    ax[3].set_yscale('log', nonposy='clip')
    ax[3].set(xlabel = wn, ylabel = "Events per bin")
    fig.savefig(args.outdir + "train_reweighted" + wn + ".pdf")


### Validation data
weights_valid = [weight_validEst10, weight_validEst20, weight_validEst40]
weights_valid_names = ["weight_validEst10", "weight_validEst20", "weight_validEst40"]

for w, wn in zip(weights_valid, weights_valid_names):
    fig, ax = plt.subplots(2,2,figsize=(15,10))
    ax = ax.flatten()
    fig, ax[0] = Plot(Histogram(data_train['pho_eta'][validMask], data_train["label"][validMask], w, 120, -4, 4), fig, ax[0], r"$\eta$", includeN = True)
    ax[0].set_yscale('log')
    fig, ax[1] = Plot(Histogram(data_train['pho_et'][validMask], data_train["label"][validMask], w, 90, 0, 50), fig, ax[1], r"$E_T$", includeN = False)
    ax[1].set_yscale('log')
    fig, ax[2] = Plot(Histogram(data_train['correctedScaledActualMu'][validMask], data_train["label"][validMask], w, 90, 0, 90), fig, ax[2], r"$\langle\mu\rangle$", includeN = False)
    ax[2].set_yscale('log')

    counts_weight, edges_weight = np.histogram(w, bins=120, range=(0, 40))
    ax[3].step(x=edges_weight, y=np.append(counts_weight, 0), where="post", color = "k");
    ax[3].set_yscale('log', nonposy='clip')
    ax[3].set(xlabel = wn, ylabel = "Events per bin")
    fig.savefig(args.outdir + "valid_reweighted" + wn + ".pdf")

log.info(f"Done! Total time: {timedelta(seconds=time() - t_start)}")
