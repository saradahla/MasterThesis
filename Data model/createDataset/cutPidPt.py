'pt'#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Tuesday 30 June
@author: Sara Dahl Andersen


nohup python -u cutPidPt.py --tag 230920_Data  output/MuoSingleHdf5/010920_3/010920_3.h5 2>&1 &> output/logCutPidPt.txt & disown


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
parser.add_argument('--outdir', action='store', default="output/MuoCutFilesForISO/", type=str,
                    help='Output directory.')
parser.add_argument('paths', type=str, nargs='+',
                    help='HDF5 file(s) to reweight.')
# parser.add_argument('--testSize', action='store', default=0.2, type=float,
#                     help='Size of test set from 0.0 to 1.0. (Default = 0.2)')
# parser.add_argument('--validSize', action='store', default=0.2, type=float,
#                     help='Size of validation set from 0.0 to 1.0. Split after separation into test and training set. (Default = 0.2)')
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

modelPID = "/groups/hep/sda/work/Zmm model/PID_ISO_models/output/PIDModels/010920_ZbbW/lgbmPID.txt"
modelPID_onlyATLAS = "/groups/hep/sda/work/Zmm model/PID_ISO_models/output/PIDModels/010920_ZbbW_only6/lgbmPID.txt"
modelISO = "/groups/hep/sda/work/Zmm model/PID_ISO_models/output/ISOModels/110820_ZbbW/lgbmISO.txt"

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
    score = gbm.predict(data[training_var], n_jobs=args.max_processes)
    return logit(score)

def GetPIDscore(gbm, data):
    training_var = [f'muo_numberOfPrecisionLayers',
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

    score = gbm.predict(data[training_var], n_jobs=args.max_processes)
    return logit(score)

def GetPIDscoreATLAS(gbm, data):
    training_var = [#f'muo_numberOfPrecisionLayers',
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
data = data.drop(data[data["Type"] == 2].index, axis = 0) #removes all type 2 muons (opposite sign, wrong mass)

# Add label
log.info(f"Add label to data")
data["label"] = 0
data.loc[data["Type"]==1,"label"] = 1

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

# # Add Z_sig
# data['Z_sig'] = (data['muo1_priTrack_z0'] - data['muo2_priTrack_z0'])/np.sqrt(data['muo1_priTrack_z0Sig']**2 + data['muo2_priTrack_z0Sig']**2)
# data['muo1_d0_d0Sig'] = data['muo1_priTrack_d0']/data['muo1_priTrack_d0Sig']
# data['muo2_d0_d0Sig'] = data['muo2_priTrack_d0']/data['muo2_priTrack_d0Sig']

# Add ML isolation
log.info(f"Add ML models score to data")
log.info(f"        Muon models: {modelPID}")
log.info(f"                     {modelPID_onlyATLAS}")
log.info(f"                     {modelISO}")

PIDmod = lgb.Booster(model_file = modelPID)
PIDmod_onlyATLAS = lgb.Booster(model_file = modelPID_onlyATLAS)
ISOmod = lgb.Booster(model_file = modelISO)

data['muo_PID_score'] = GetPIDscore(PIDmod,data)

data['muo_PID_score_ATLAS'] = GetPIDscoreATLAS(PIDmod_onlyATLAS,data)

data['muo_ISO_score'] = GetISOscore(ISOmod,data)

#============================================================================
# Cut on pt
#============================================================================

mask_pt = data["muo_pt"]/1000 > 4.5
data = data[mask_pt]

#============================================================================
# Plot the PID and ISO distributions
#============================================================================

type = data["label"]

c1 = 'g'
c2 = 'tab:purple'

maskPID = (data['muo_PID_score'] > -20)

fig, ax = plt.subplots(1,2,figsize=(10,5))
ax = ax.flatten()
ax[0].set_title(f"For 11 variables")
ax[0].plot(data['muo_PID_score'][(type==0) & maskPID],data['muo_ISO_score'][(type==0) & maskPID],'.', color = c1, alpha = 0.2, label = "Background")#, bins = 50, cmax = 30);
ax[0].plot(data['muo_PID_score'][(type==1) & maskPID],data['muo_ISO_score'][(type==1) & maskPID], '.', color=c2, alpha = 0.2, label = "Signal")#, bins = 50, cmax = 30);
ax[0].axvline(0, color = "k", linestyle = "dotted")
ax[0].axhline(0, color = "k", linestyle = "dotted")

# Calculate percentage in each quadrant
nSig = len(data[type==1])
nBkg = len(data[type==0])

nSigQ1 = len(data[type==1][(data['muo_PID_score'] > 0) & (data['muo_ISO_score'] > 0)])
nSigQ2 = len(data[type==1][(data['muo_PID_score'] < 0) & (data['muo_ISO_score'] > 0)])
nSigQ3 = len(data[type==1][(data['muo_PID_score'] > 0) & (data['muo_ISO_score'] < 0)])
nSigQ4 = len(data[type==1][(data['muo_PID_score'] < 0) & (data['muo_ISO_score'] < 0)])

nBkgQ1 = len(data[type==0][(data['muo_PID_score'] > 0) & (data['muo_ISO_score'] > 0)])
nBkgQ2 = len(data[type==0][(data['muo_PID_score'] < 0) & (data['muo_ISO_score'] > 0)])
nBkgQ3 = len(data[type==0][(data['muo_PID_score'] > 0) & (data['muo_ISO_score'] < 0)])
nBkgQ4 = len(data[type==0][(data['muo_PID_score'] < 0) & (data['muo_ISO_score'] < 0)])

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


ax[1].set_title(f"For 8 variables")
ax[1].plot(data['muo_PID_score_ATLAS'][type==0],data['muo_ISO_score'][type==0],'.', color = c1, alpha = 0.2, label = "Background")#, bins = 50, cmax = 30);
ax[1].plot(data['muo_PID_score_ATLAS'][type==1],data['muo_ISO_score'][type==1],'.', color=c2, alpha = 0.2, label = "Signal")#, bins = 50, cmax = 30);
ax[1].axvline(0, color = "k", linestyle = "dotted")
ax[1].axhline(0, color = "k", linestyle = "dotted")
ax[1].set(xlabel = "ML PID", ylabel = "ML ISO");
ax[1].legend(loc = 9, prop={'size': 6})

# Calculate percentage in each quadrant
nSig = len(data[type==1])
nBkg = len(data[type==0])
nSigQ1 = len(data[type==1][(data['muo_PID_score_ATLAS'] > 0) & (data['muo_ISO_score'] > 0)])
nSigQ2 = len(data[type==1][(data['muo_PID_score_ATLAS'] < 0) & (data['muo_ISO_score'] > 0)])
nSigQ3 = len(data[type==1][(data['muo_PID_score_ATLAS'] > 0) & (data['muo_ISO_score'] < 0)])
nSigQ4 = len(data[type==1][(data['muo_PID_score_ATLAS'] < 0) & (data['muo_ISO_score'] < 0)])

nBkgQ1 = len(data[type==0][(data['muo_PID_score_ATLAS'] > 0) & (data['muo_ISO_score'] > 0)])
nBkgQ2 = len(data[type==0][(data['muo_PID_score_ATLAS'] < 0) & (data['muo_ISO_score'] > 0)])
nBkgQ3 = len(data[type==0][(data['muo_PID_score_ATLAS'] > 0) & (data['muo_ISO_score'] < 0)])
nBkgQ4 = len(data[type==0][(data['muo_PID_score_ATLAS'] < 0) & (data['muo_ISO_score'] < 0)])

ax[1].text(0.99,0.96, f"{np.round((nBkgQ1/nBkg)*100,2)}%, n = {nBkgQ1}", color = c1, horizontalalignment='right', transform=ax[1].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))
ax[1].text(0.99,0.92, f"{np.round((nSigQ1/nSig)*100,2)}%, n = {nSigQ1}", color = c2, horizontalalignment='right', transform=ax[1].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))

ax[1].text(0.01,0.96, f"{np.round((nBkgQ2/nBkg)*100,2)}%, n = {nBkgQ2}", color = c1, transform=ax[1].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))
ax[1].text(0.01,0.92, f"{np.round((nSigQ2/nSig)*100,2)}%, n = {nSigQ2}", color = c2, transform=ax[1].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))

ax[1].text(0.99,0.02, f"{np.round((nBkgQ3/nBkg)*100,2)}%, n = {nBkgQ3}", color = c1, horizontalalignment='right', transform=ax[1].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))
ax[1].text(0.99,0.06, f"{np.round((nSigQ3/nSig)*100,2)}%, n = {nSigQ3}", color = c2, horizontalalignment='right', transform=ax[1].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))

ax[1].text(0.01,0.02, f"{np.round((nBkgQ4/nBkg)*100,2)}%, n = {nBkgQ4}", color = c1, transform=ax[1].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))
ax[1].text(0.01,0.06, f"{np.round((nSigQ4/nSig)*100,2)}%, n = {nSigQ4}", color = c2, transform=ax[1].transAxes, bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad = 0.1))

fig.tight_layout()
fig.savefig(args.outdir + "PidIsoDistScatter_" + args.tag + ".png", dpi = 600)

fig, ax = plt.subplots(3,1,figsize=(5,15))
ax = ax.flatten()

ax[0].set_title(f"For 11 variables")
ax[0].hist(data['muo_PID_score'][(type==0) & maskPID], bins = 100, color = c1, label = "Background", histtype = "step")#, bins = 50, cmax = 30);
ax[0].hist(data['muo_PID_score'][(type==1) & maskPID],bins = 100, color = c2,label = "Signal", histtype = "step")#, bins = 50, cmax = 30);
ax[0].axvline(0, color = 'k', linestyle = "dashed", label = "Cut on signal")#, bins = 50, cmax = 30);
ax[0].set(xlabel = "ML PID (11 variables)", ylabel = "Frequency");
ax[0].legend(prop={'size': 15})
for item in ([ax[0].title, ax[0].xaxis.label, ax[0].yaxis.label] +
             ax[0].get_xticklabels() + ax[0].get_yticklabels()):
    item.set_fontsize(15)

ax[1].set_title(f"Isolation")
ax[1].hist(data['muo_ISO_score'][type==0], bins = 100, color = c1, label = "Background", histtype = "step")#, bins = 50, cmax = 30);
ax[1].hist(data['muo_ISO_score'][type==1],bins = 100, color = c2,label = "Signal", histtype = "step")#, bins = 50, cmax = 30);
ax[1].set(xlabel = "ML ISO", ylabel = "Frequency");
ax[1].legend(prop={'size': 15})
for item in ([ax[1].title, ax[1].xaxis.label, ax[1].yaxis.label]  +
             ax[1].get_xticklabels() + ax[1].get_yticklabels()):
    item.set_fontsize(15)

#fig.savefig("11vars_PID.pdf")
ax[2].set_title(f"For 8 variables")
ax[2].hist(data['muo_PID_score_ATLAS'][type==0], bins = 100, color = c1, label = "Background", histtype = "step")#, bins = 50, cmax = 30);
ax[2].hist(data['muo_PID_score_ATLAS'][type==1],bins = 100, color = c2,label = "Signal", histtype = "step")#, bins = 50, cmax = 30);
ax[2].axvline(0, color = 'k', linestyle = "dashed", label = "Cut on signal")#, bins = 50, cmax = 30);
ax[2].set(xlabel = "ML PID (8 variables)", ylabel = "Frequency");
ax[2].legend(prop={'size': 15})
for item in ([ax[2].title, ax[2].xaxis.label, ax[2].yaxis.label] +
             ax[2].get_xticklabels() + ax[2].get_yticklabels()):
    item.set_fontsize(15)

fig.tight_layout()
fig.savefig(args.outdir + "PidIsoHist_" + args.tag + ".pdf", dpi = 600)


#============================================================================
# Cut on Isolation
#============================================================================
mask_pid = (data["muo_PID_score"] > 0)
data[data["label"] == 1] = data[(data["label"] == 1) & mask_pid]
data = data.dropna(how = "all") #removes rows where all values are nan


#============================================================================
# Save to hdf5
#============================================================================
column_names = data.columns
filename = args.outdir+fname+".h5"

log.info("Saving data to {}".format(filename))
with h5py.File(filename, 'w') as hf:
    for var in column_names:
        hf.create_dataset( f'{var}', data=np.array(data[var]), chunks=True, maxshape= (None,), compression='lzf')


log.info(f"Done! Total time: {timedelta(seconds=time() - t_start)}")
