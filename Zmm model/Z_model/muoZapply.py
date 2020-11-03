#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Tuesday 30 June
@author: Sara Dahl Andersen

Running my Z model for muons and outputting the score

nohup python -u muoZapply.py --tag 110820_ZbbW  output/ZReweightFiles/110820_ZbbW/combined_110820_train.h5 2>&1 &> output/logZApply.txt & disown
nohup python -u muoZapply.py --tag 150920_ZbbW  output/ZReweightFiles/260820_ZbbW/combined_260820_train.h5 2>&1 &> output/logZApply_150920.txt & disown
nohup python -u muoZapply.py --tag 220920_ZbbW  output/ZReweightFiles/220920_ZbbW/combined_220920_train.h5 2>&1 &> output/logZApply_220920.txt & disown
nohup python -u muoZapply.py --tag 250920  output/ZReweightFiles/220920_ZbbW/combined_220920_train.h5 2>&1 &> output/logZApply_250920.txt & disown
nohup python -u muoZapply.py --tag 081020  output/ZReweightFiles/220920_ZbbW/combined_220920_train.h5 2>&1 &> output/logZApply_081020.txt & disown

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
parser.add_argument('--outdir', action='store', default="output/ZModels/", type=str,
                    help='Output directory.')
parser.add_argument('path', type=str, nargs='+',
                    help='HDF5 file(s) to use for Z model.')
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

def getSameFpr(fprArray, tprArray, fprGoal, thresholds):
    # Round fpr to compare
    fprMask = (np.around(fprArray,decimals=6) == np.around(fprGoal,decimals=6))
    # If round to 6 decimals does not give any results round to 4 decimals
    if np.sum(fprMask) == 0:
        fprMask = (np.around(fprArray,decimals=4) == np.around(fprGoal,decimals=4))

    if np.sum(fprMask) == 0:
        fprMask = (np.around(fprArray,decimals=2) == np.around(fprGoal,decimals=2))

    # Possible fpr and tpr values
    fprChosen = fprArray[fprMask]
    tprChosen = tprArray[fprMask]
    thresholdsChosen = thresholds[fprMask]

    # Number of possible fpr values to choose from
    nfprMask = np.sum(fprMask)

    # Calculate difference between the possible fpr and the goal fpr
    fprDiff = fprChosen - fprGoal

    # Choose index: More than one possibility
    if nfprMask>1:
        # If there all possible fpr are the same, choose half way point
        if np.sum(fprDiff)==0:
            idx = int(nfprMask/2) # Half way point
        # If the possible fpr are not the same, get minimum difference
        else:
            idx = np.argmin(fprDiff)
    # Choose index: Only one possibility
    else:
        idx = 0

    return fprMask, idx, fprChosen[idx], tprChosen[idx], thresholdsChosen[idx]

def mask_LGBM(data, sigSel):
    return (data["predLGBM"]>sigSel)

def mask_LGBM_LH(data, sigSel):
    return (data["predLGBM_LH"]>sigSel)

def getSameBkg(data, maskFunction, selStart, decimals=4):
    Sel = selStart
    i = 0

    nBkgATLAS = np.sum(data[(data[truth_var]==0)]['isATLAS'])
    nSigATLAS = np.sum(data[(data[truth_var]==1)]['isATLAS'])

    print(f"    ATLAS selection:  nBkg = {nBkgATLAS}, nSig = {nSigATLAS}")

    nBkgCompare = np.sum( (data_get[truth_var]==0) & maskFunction(data, Sel) )
    nSigCompare = np.sum( (data_get[truth_var]==1) & maskFunction(data, Sel) )

    print(f"   Initial selection data at selection {selStart}:    nBkg = {nBkgCompare}, nSig = {nSigCompare}")



    while nBkgCompare > nBkgATLAS:
        # Increase selection
        Sel = Sel + 10**(-decimals)
        nBkgBefore = nBkgCompare

        nBkgCompare = np.sum( (  (data[truth_var]==0) & maskFunction(data, Sel) ) )
        nSigCompare = np.sum( ( (data[truth_var]==1) & maskFunction(data, Sel) ) )

        i += 1
        if (i % 100) == 0:
            print(f"After {i} iterations with selection = {Sel}:")
            print(f"    Selection (valid):         nBkg = {nBkgCompare}, nSig = {nSigCompare}")
        if i > 300 and nBkgCompare == nBkgBefore:
            break

    Sel = Sel - 10**(-decimals) #get prediction right before we are below ATLAS

    nBkg = np.sum( (  (data[truth_var]==0) & maskFunction(data, Sel) ) )
    nSig = np.sum( (  (data[truth_var]==1) & maskFunction(data, Sel) ) )

    print(f"    Final selection data:    nBkg = {nBkg}, nSig = {nSig}")
    print(f"    Final selection: {Sel}\n")
    print(f"    Yielding increase in signal of {np.round(((nSig-nSigATLAS)/nSig)*100,2)}")

    Sel = round(Sel,decimals)

    if Sel == selStart:
        # Check if signal selection had no effect
        print("    Initial selection too high... Exiting.")
        quit()

    return Sel


def GetATLASCut(data):
    return ( ( np.sign(data['muo1_truthPdgId'])*np.sign(data['muo2_truthPdgId']) == -1 ) & #opposite sign
             ( (data['muo1_pt'] / 1000) > 10 ) &
             ( (data['muo2_pt'] / 1000) > 10 ) &
             ( (np.abs( data['muo1_eta']) < 2.7)) &
             ( (np.abs( data['muo2_eta']) < 2.7)) &
             ( data['muo1_LHMedium'] * data['muo2_LHMedium'] ) &
             ( abs(data['muo1_d0_d0Sig']) < 3 ) &
             ( abs(data['muo2_d0_d0Sig']) < 3 )# &
            # ( np.sin(2*np.arctan(np.exp(-data['muo1_eta']))) < 0.5 ) &
            # ( np.sin(2*np.arctan(np.exp(-data['muo2_eta']))) < 0.5 )
             )

def GetPIDCut(data, MLpidSel):
    return ( ( np.sign(data['muo1_truthPdgId'])*np.sign(data['muo2_truthPdgId']) == -1 ) & #opposite sign
             ( data['muo1_pt']/1000 > 10 ) &
             ( data['muo2_pt']/1000 > 10 ) &
             ( (np.abs( data['muo1_eta'])<2.7)) &
             ( (np.abs( data['muo2_eta'])<2.7)) &
             ( data['muo1_PID_score'] > MLpidSel ) &
             ( data['muo2_PID_score'] > MLpidSel ) &
             ( abs(data['muo1_d0_d0Sig']) < 3 ) &
             ( abs(data['muo2_d0_d0Sig']) < 3 )
             )

def GetPIDISOCut(data, MLisoSel):
    return ( ( np.sign(data['muo1_truthPdgId'])*np.sign(data['muo2_truthPdgId']) == -1 ) & #opposite sign
             ( data['muo1_pt']/1000 > 10 ) &
             ( data['muo2_pt']/1000 > 10 ) &
             ( (np.abs( data['muo1_eta'])<2.7)) &
             ( (np.abs( data['muo2_eta'])<2.7)) &
             ( data['muo1_PID_score'] > MLpidSel ) &
             ( data['muo2_PID_score'] > MLpidSel ) &
             ( abs(data['muo1_d0_d0Sig']) < 3 ) &
             ( abs(data['muo2_d0_d0Sig']) < 3 ) &
             ( data['muo1_ISO_score'] > MLisoSel ) &
             ( data['muo2_ISO_score'] > MLisoSel )
             )


def getMLcut(data, maskFunction,selStart,decimals=4):
    # Number of background pairs in ATLAS selection
    nBkgATLAS = np.sum(data[(data[truth_var]==0)]['isATLAS'])
    nSigATLAS = np.sum(data[(data[truth_var]==1)]['isATLAS'])
    print(f"    ATLAS selection:   fpr = {fprCompare}, tpr = {tprCompare} | nBkg = {nBkgATLAS}, nSig = {nSigATLAS}")

    # Initiate signal selection
    Sel = selStart
    i = 0

    # Choose data
    mask = (data["dataset"]==1)
    #dataSel = data[mask][:].copy().reset_index(drop=True)
    #predSel = pd.DataFrame(data=np.zeros(len(dataSel)))
    #predSel.loc[dataSel[maskFunction(dataSel,Sel)].index,0] = 1

    #fprSel, tprSel, thresholdsSel = roc_curve(y_valid, predSel[:][0], sample_weight=data_valid["regWeight_nEst10"])
    #aucSel = auc(fprSel, tprSel)

    #fprSelCompare = fprSel[1]; tprSelCompare = tprSel[1]
    #print(f"    Selection (valid):         fpr = {fprSelCompare}, tpr = {tprSel[1]}")
    nBkgCompare = np.sum( ( (data[truth_var]==0) & maskFunction(data, Sel) ) )
    nSigCompare = np.sum( ( (data[truth_var]==1) & maskFunction(data, Sel) ) )
    print(f"    Selection (valid):         nBkg = {nBkgCompare}, nSig = {nSigCompare}")

    # Find signal selection
    while nBkgCompare > nBkgATLAS:
        # Increase selection
        Sel = Sel + 10**(-decimals)
        nBkgBefore = nBkgCompare

        # Calculate new fpr and tpr
        # predSel = pd.DataFrame(data=np.zeros(len(dataSel)))
        # predSel.loc[dataSel[maskFunction(dataSel,Sel)].index,0] = 1
        # fprSel, tprSel, thresholdsSel = roc_curve(y_valid, predSel[:][0], sample_weight=data_valid["regWeight_nEst10"])
        # aucSel = auc(fprSel, tprSel)
        nBkgCompare = np.sum( (  (data[truth_var]==0) & maskFunction(data, Sel) ) )
        nSigCompare = np.sum( (  (data[truth_var]==1) & maskFunction(data, Sel) ) )

        #fprSel, tprSel, aucSel, thresholdsSel = rocVars(predSel[:][0], mask, weightName)
        # fprSelCompare = fprSel[1]; tprSelCompare = tprSel[1]
        i += 1
        if (i % 100) == 0:
            print(f"After {i} iterations with selection = {Sel}:")
            print(f"    Selection (valid):         nBkg = {nBkgCompare}, nSig = {nSigCompare}")
        if i > 300 and nBkgCompare == nBkgBefore:
            break

    Sel = Sel - 10**(-decimals) #get prediction right before we are below ATLAS

    nBkg = np.sum( ( (data[truth_var]==0) & maskFunction(data, Sel) ) )
    nSig = np.sum( ( (data[truth_var]==1) & maskFunction(data, Sel) ) )

    print(f"    Final selection (valid):    nBkg = {nBkg}, nSig = {nSig}")
    #fpr = {fprSelCompare}, tpr = {tprSel[1]}
    print(f"    Final selection: {Sel}\n")
    Sel = round(Sel,decimals)

    # if Sel == selStart:
    #     # Check if signal selection had no effect
    #     print("    Initial selection too high... Exiting.")
    #     quit()

    return Sel


# ================================================ #
#                End of functions                  #
# ================================================ #


# Data
data_get = h5ToDf(args.path[0])
#change ATLAS cut to include sign
data_get["isATLAS"] = GetATLASCut(data_get)

#data = h5ToDf("/Users/sda/hep/work/Zmm model/Z_model/output/ZReweightFiles/260820_ZbbW/combined_260820_train.h5")
#plt.hist(data["pt"][data["label"] == 1], bins = 120, range = (0,20));

#
# np.sign(-13)*np.sign(-13)
#
# data["isATLAS"].value_counts()
# plt.hist(data[GetATLASCut(data)], bins = 100)
#
# GetATLASCut(data).astype(float).value_counts()
#
#
# plt.title("Muon 1, PID score")
# plt.hist(data["muo1_PID_score"][data["label"] == 1], bins = 100, color = 'r', range = (-10,20), label = "Signal");
# plt.hist(data["muo1_PID_score"][data["label"] == 0], alpha = 0.5, bins = 100, color = 'b', range = (-10,10), label = "Background");
# plt.legend()
# plt.savefig("Muon1PID.png")
#
# plt.title("Muon 2, PID score")
# plt.hist(data["muo2_PID_score"][data["label"] == 1], bins = 100, color = 'r', range = (-10,20), label = "Signal");
# plt.hist(data["muo2_PID_score"][data["label"] == 0], alpha = 0.5, bins = 100, color = 'b', range = (-10,10), label = "Background");
# plt.legend()
# plt.savefig("Muon2PID.png")
#
# plt.title("Muon 1, ISO score")
# plt.hist(data["muo1_ISO_score"][data["label"] == 1], bins = 100, color = 'r', range = (-3,3), label = "Signal");
# plt.hist(data["muo1_ISO_score"][data["label"] == 0], alpha = 0.5, bins = 100, color = 'b', range = (-3,3), label = "Background");
# plt.legend()
# plt.savefig("Muon1ISO.png")
#
# plt.title("Muon 2, ISO score")
# plt.hist(data["muo2_ISO_score"][data["label"] == 1], bins = 100, color = 'r', range = (-3,3), label = "Signal");
# plt.hist(data["muo2_ISO_score"][data["label"] == 0], alpha = 0.5, bins = 100, color = 'b', range = (-3,3), label = "Background");
# plt.legend()
# plt.savefig("Muon2ISO.png")

# data[GetPIDCut(data, 1)]
# len(data[GetPIDCut(data, 1)]["label"] == 1)
# len(data[data[GetPIDCut(data, 1)]["label"] == 0])
#
# MLpidSel = 8.3
# data[GetPIDCut(data, 8.3)]["label"].value_counts()
#
# data[GetPIDISOCut(data, -4)]["label"].value_counts()
# data[GetPIDCut(data, -20)]["label"].hist(bins = 2)
#plt.hist(data["muo1_ISO_score"][data["label"] == 1], color = 'b', bins = 100, range = (-3, 3));
#plt.hist(data["muo1_ISO_score"][data["label"] == 0], color = 'r', alpha = 0.5, bins = 100, range = (-3, 3));

#plt.hist(data["label"], bins = 2);

data_train = data_get[data_get["dataset"] == 0]
data_valid = data_get[data_get["dataset"] == 1]

# Check shapes
shapeAll = np.shape(data_get)
shapeTrain = np.shape(data_train)
shapeTrainSig = np.shape(data_train[data_train["label"] == 1])
shapeTrainBkg = np.shape(data_train[data_train["label"] == 0])
shapeValid = np.shape(data_valid)
shapeValidSig = np.shape(data_valid[data_valid["label"] == 1])
shapeValidBkg = np.shape(data_valid[data_valid["label"] == 0])

log.info(f"Shape all:       {shapeAll}")
log.info(f"Shape train:     {shapeTrain}")
log.info(f"Shape valid:     {shapeValid}")
log.info(f"Shape train:     sig: {shapeTrainSig}, bkg: {shapeTrainBkg}")
log.info(f"Shape valid:      sig: {shapeValidSig}, bkg: {shapeValidBkg}")

#log.info(f"Just checking if my ATLAS function is the same as old one (1 = True, 0 = False):")
#log.info(f"TRAINING SET: {np.all( GetATLASCut(data_train) == data_train['isATLAS'] )}")
#log.info(f"VALIDATION SET: {np.all( GetATLASCut(data_valid) == data_valid['isATLAS'] )}")

# =========================
#       Variables
# =========================

truth_var = "label"
training_var = [
'correctedScaledActualMu',
'NvtxReco',
# "invM" ,
# "pt" ,
# "eta" ,
# "phi" ,
# "type" ,
'Z_sig',
# "isATLAS" ,
####
####
'muo1_PID_score',
'muo1_ISO_score',
'muo1_d0_d0Sig',
# 'muo1_truthPdgId',
# 'muo1_truthType',
# 'muo1_truthOrigin',
# 'muo1_truth_eta',
# 'muo1_truth_phi',
# 'muo1_truth_m',
# 'muo1_truth_px',
# 'muo1_truth_py',
# 'muo1_truth_pz',
# 'muo1_truth_E',
# 'muo1_etcone20',
# 'muo1_etcone30',
# 'muo1_etcone40',
# 'muo1_ptcone20',
# 'muo1_ptcone30',
# 'muo1_ptcone40',
# 'muo1_ptvarcone20',
# 'muo1_ptvarcone30',
# 'muo1_ptvarcone40',
# 'muo1_pt',
# 'muo1_eta',
# 'muo1_phi',
# 'muo1_charge',
# 'muo1_innerSmallHits',
# 'muo1_innerLargeHits',
# 'muo1_middleSmallHits',
# 'muo1_middleLargeHits',
# 'muo1_outerSmallHits',
# 'muo1_outerLargeHits',
# 'muo1_extendedSmallHits',
# 'muo1_extendedLargeHits',
# 'muo1_cscEtaHits',
# 'muo1_cscUnspoiledEtaHits',
# 'muo1_innerSmallHoles',
# 'muo1_innerLargeHoles',
# 'muo1_middleSmallHoles',
# 'muo1_middleLargeHoles',
# 'muo1_outerSmallHoles',
# 'muo1_outerLargeHoles',
# 'muo1_extendedSmallHoles',
# 'muo1_extendedLargeHoles',
# 'muo1_author',
# 'muo1_allAuthors',
# 'muo1_muonType',
# 'muo1_numberOfPrecisionLayers',
# 'muo1_numberOfPrecisionHoleLayers',
# 'muo1_quality',
# 'muo1_energyLossType',
# 'muo1_spectrometerFieldIntegral',
# 'muo1_scatteringCurvatureSignificance',
# 'muo1_scatteringNeighbourSignificance',
# 'muo1_momentumBalanceSignificance',
# 'muo1_segmentDeltaEta',
# 'muo1_CaloLRLikelihood',
# 'muo1_EnergyLoss',
# 'muo1_CaloMuonIDTag',
# 'muo1_DFCommonGoodMuon',
# 'muo1_DFCommonMuonsPreselection',
# 'muo1_LHLoose',
# 'muo1_LHMedium',
# 'muo1_LHTight',
'muo1_priTrack_d0',
'muo1_priTrack_z0',
# 'muo1_priTrack_d0Sig',
# 'muo1_priTrack_z0Sig',
# 'muo1_priTrack_theta',
# 'muo1_priTrack_qOverP',
# 'muo1_priTrack_vx',
# 'muo1_priTrack_vy',
# 'muo1_priTrack_vz',
# 'muo1_priTrack_phi',
# 'muo1_priTrack_chiSquared',
# 'muo1_priTrack_numberDoF',
# 'muo1_priTrack_radiusOfFirstHit',
# 'muo1_priTrack_trackFitter',
# 'muo1_priTrack_particleHypothesis',
# 'muo1_priTrack_numberOfUsedHitsdEdx',
# 'muo1_priTrack_numberOfContribPixelLayers',
# 'muo1_priTrack_numberOfInnermostPixelLayerHits',
# 'muo1_priTrack_expectInnermostPixelLayerHit',
# 'muo1_priTrack_numberOfNextToInnermostPixelLayerHits',
# 'muo1_priTrack_expectNextToInnermostPixelLayerHit',
# 'muo1_priTrack_numberOfPixelHits',
# 'muo1_priTrack_numberOfGangedPixels',
# 'muo1_priTrack_numberOfGangedFlaggedFakes',
# 'muo1_priTrack_numberOfPixelSpoiltHits',
# 'muo1_priTrack_numberOfDBMHits',
# 'muo1_priTrack_numberOfSCTHits',
# 'muo1_priTrack_numberOfTRTHits',
# 'muo1_priTrack_numberOfOutliersOnTrack',
# 'muo1_priTrack_standardDeviationOfChi2OS',
# 'muo1_priTrack_pixeldEdx',
# 'muo1_IDTrack_d0',
# 'muo1_IDTrack_z0',
# 'muo1_IDTrack_d0Sig',
# 'muo1_IDTrack_z0Sig',
# 'muo1_IDTrack_theta',
# 'muo1_IDTrack_qOverP',
# 'muo1_IDTrack_vx',
# 'muo1_IDTrack_vy',
# 'muo1_IDTrack_vz',
# 'muo1_IDTrack_phi',
# 'muo1_IDTrack_chiSquared',
# 'muo1_IDTrack_numberDoF',
# 'muo1_IDTrack_radiusOfFirstHit',
# 'muo1_IDTrack_trackFitter',
# 'muo1_IDTrack_particleHypothesis',
# 'muo1_IDTrack_numberOfUsedHitsdEdx',
# 'muo1_ET_Core',
# 'muo1_ET_EMCore',
# 'muo1_ET_HECCore',
# 'muo1_ET_TileCore',
# 'muo1_FSR_CandidateEnergy',
# 'muo1_InnerDetectorPt',
# 'muo1_MuonSpectrometerPt',
# 'muo1_combinedTrackOutBoundsPrecisionHits',
# 'muo1_coreMuonEnergyCorrection',
# 'muo1_deltaphi_0',
# 'muo1_deltaphi_1',
# 'muo1_deltatheta_0',
# 'muo1_deltatheta_1',
# 'muo1_etconecoreConeEnergyCorrection',
# 'muo1_extendedClosePrecisionHits',
# 'muo1_extendedOutBoundsPrecisionHits',
# 'muo1_innerClosePrecisionHits',
# 'muo1_innerOutBoundsPrecisionHits',
# 'muo1_isEndcapGoodLayers',
# 'muo1_isSmallGoodSectors',
# 'muo1_middleClosePrecisionHits',
# 'muo1_middleOutBoundsPrecisionHits',
# 'muo1_numEnergyLossPerTrack',
# 'muo1_numberOfGoodPrecisionLayers',
# 'muo1_outerClosePrecisionHits',
# 'muo1_outerOutBoundsPrecisionHits',
# 'muo1_sigmadeltaphi_0',
# 'muo1_sigmadeltaphi_1',
# 'muo1_sigmadeltatheta_0',
# 'muo1_sigmadeltatheta_1',
# 'muo1_etconeCorrBitset',
# 'muo1_neflowisol20',
# 'muo1_neflowisol30',
# 'muo1_neflowisol40',
# 'muo1_neflowisolCorrBitset',
# 'muo1_neflowisolcoreConeEnergyCorrection',
# 'muo1_ptconeCorrBitset',
# 'muo1_ptconecoreTrackPtrCorrection',
# 'muo1_topoetconeCorrBitset',
# 'muo1_topoetconecoreConeEnergyCorrection',
# 'muo1_CT_EL_Type',
# 'muo1_CT_ET_Core',
# 'muo1_CT_ET_FSRCandidateEnergy',
# 'muo1_CT_ET_LRLikelihood',
# 'muo1_d0_staco',
# 'muo1_phi0_staco',
# 'muo1_qOverPErr_staco',
# 'muo1_qOverP_staco',
# 'muo1_theta_staco',
# 'muo1_z0_staco',
####
####
'muo2_PID_score',
'muo2_ISO_score',
'muo2_d0_d0Sig',
# 'muo2_truthPdgId',
# 'muo2_truthType',
# 'muo2_truthOrigin',
# 'muo2_truth_eta',
# 'muo2_truth_phi',
# 'muo2_truth_m',
# 'muo2_truth_px',
# 'muo2_truth_py',
# 'muo2_truth_pz',
# 'muo2_truth_E',
# 'muo2_etcone20',
# 'muo2_etcone30',
# 'muo2_etcone40',
# 'muo2_ptcone20',
# 'muo2_ptcone30',
# 'muo2_ptcone40',
# 'muo2_ptvarcone20',
# 'muo2_ptvarcone30',
# 'muo2_ptvarcone40',
# 'muo2_pt',
# 'muo2_eta',
# 'muo2_phi',
# 'muo2_charge',
# 'muo2_innerSmallHits',
# 'muo2_innerLargeHits',
# 'muo2_middleSmallHits',
# 'muo2_middleLargeHits',
# 'muo2_outerSmallHits',
# 'muo2_outerLargeHits',
# 'muo2_extendedSmallHits',
# 'muo2_extendedLargeHits',
# 'muo2_cscEtaHits',
# 'muo2_cscUnspoiledEtaHits',
# 'muo2_innerSmallHoles',
# 'muo2_innerLargeHoles',
# 'muo2_middleSmallHoles',
# 'muo2_middleLargeHoles',
# 'muo2_outerSmallHoles',
# 'muo2_outerLargeHoles',
# 'muo2_extendedSmallHoles',
# 'muo2_extendedLargeHoles',
# 'muo2_author',
# 'muo2_allAuthors',
# 'muo2_muonType',
# 'muo2_numberOfPrecisionLayers',
# 'muo2_numberOfPrecisionHoleLayers',
# 'muo2_quality',
# 'muo2_energyLossType',
# 'muo2_spectrometerFieldIntegral',
# 'muo2_scatteringCurvatureSignificance',
# 'muo2_scatteringNeighbourSignificance',
# 'muo2_momentumBalanceSignificance',
# 'muo2_segmentDeltaEta',
# 'muo2_CaloLRLikelihood',
# 'muo2_EnergyLoss',
# 'muo2_CaloMuonIDTag',
# 'muo2_DFCommonGoodMuon',
# 'muo2_DFCommonMuonsPreselection',
# 'muo2_LHLoose',
# 'muo2_LHMedium',
# 'muo2_LHTight',
'muo2_priTrack_d0',
'muo2_priTrack_z0',
# 'muo2_priTrack_d0Sig',
# 'muo2_priTrack_z0Sig',
# 'muo2_priTrack_theta',
# 'muo2_priTrack_qOverP',
# 'muo2_priTrack_vx',
# 'muo2_priTrack_vy',
# 'muo2_priTrack_vz',
# 'muo2_priTrack_phi',
# 'muo2_priTrack_chiSquared',
# 'muo2_priTrack_numberDoF',
# 'muo2_priTrack_radiusOfFirstHit',
# 'muo2_priTrack_trackFitter',
# 'muo2_priTrack_particleHypothesis',
# 'muo2_priTrack_numberOfUsedHitsdEdx',
# 'muo2_priTrack_numberOfContribPixelLayers',
# 'muo2_priTrack_numberOfInnermostPixelLayerHits',
# 'muo2_priTrack_expectInnermostPixelLayerHit',
# 'muo2_priTrack_numberOfNextToInnermostPixelLayerHits',
# 'muo2_priTrack_expectNextToInnermostPixelLayerHit',
# 'muo2_priTrack_numberOfPixelHits',
# 'muo2_priTrack_numberOfGangedPixels',
# 'muo2_priTrack_numberOfGangedFlaggedFakes',
# 'muo2_priTrack_numberOfPixelSpoiltHits',
# 'muo2_priTrack_numberOfDBMHits',
# 'muo2_priTrack_numberOfSCTHits',
# 'muo2_priTrack_numberOfTRTHits',
# 'muo2_priTrack_numberOfOutliersOnTrack',
# 'muo2_priTrack_standardDeviationOfChi2OS',
# 'muo2_priTrack_pixeldEdx',
# 'muo2_IDTrack_d0',
# 'muo2_IDTrack_z0',
# 'muo2_IDTrack_d0Sig',
# 'muo2_IDTrack_z0Sig',
# 'muo2_IDTrack_theta',
# 'muo2_IDTrack_qOverP',
# 'muo2_IDTrack_vx',
# 'muo2_IDTrack_vy',
# 'muo2_IDTrack_vz',
# 'muo2_IDTrack_phi',
# 'muo2_IDTrack_chiSquared',
# 'muo2_IDTrack_numberDoF',
# 'muo2_IDTrack_radiusOfFirstHit',
# 'muo2_IDTrack_trackFitter',
# 'muo2_IDTrack_particleHypothesis',
# 'muo2_IDTrack_numberOfUsedHitsdEdx',
# 'muo2_ET_Core',
# 'muo2_ET_EMCore',
# 'muo2_ET_HECCore',
# 'muo2_ET_TileCore',
# 'muo2_FSR_CandidateEnergy',
# 'muo2_InnerDetectorPt',
# 'muo2_MuonSpectrometerPt',
# 'muo2_combinedTrackOutBoundsPrecisionHits',
# 'muo2_coreMuonEnergyCorrection',
# 'muo2_deltaphi_0',
# 'muo2_deltaphi_1',
# 'muo2_deltatheta_0',
# 'muo2_deltatheta_1',
# 'muo2_etconecoreConeEnergyCorrection',
# 'muo2_extendedClosePrecisionHits',
# 'muo2_extendedOutBoundsPrecisionHits',
# 'muo2_innerClosePrecisionHits',
# 'muo2_innerOutBoundsPrecisionHits',
# 'muo2_isEndcapGoodLayers',
# 'muo2_isSmallGoodSectors',
# 'muo2_middleClosePrecisionHits',
# 'muo2_middleOutBoundsPrecisionHits',
# 'muo2_numEnergyLossPerTrack',
# 'muo2_numberOfGoodPrecisionLayers',
# 'muo2_outerClosePrecisionHits',
# 'muo2_outerOutBoundsPrecisionHits',
# 'muo2_sigmadeltaphi_0',
# 'muo2_sigmadeltaphi_1',
# 'muo2_sigmadeltatheta_0',
# 'muo2_sigmadeltatheta_1',
# 'muo2_etconeCorrBitset',
# 'muo2_neflowisol20',
# 'muo2_neflowisol30',
# 'muo2_neflowisol40',
# 'muo2_neflowisolCorrBitset',
# 'muo2_neflowisolcoreConeEnergyCorrection',
# 'muo2_ptconeCorrBitset',
# 'muo2_ptconecoreTrackPtrCorrection',
# 'muo2_topoetconeCorrBitset',
# 'muo2_topoetconecoreConeEnergyCorrection',
# 'muo2_CT_EL_Type',
# 'muo2_CT_ET_Core',
# 'muo2_CT_ET_FSRCandidateEnergy',
# 'muo2_CT_ET_LRLikelihood',
# 'muo2_d0_staco',
# 'muo2_phi0_staco',
# 'muo2_qOverPErr_staco',
# 'muo2_qOverP_staco',
# 'muo2_theta_staco',
# 'muo2_z0_staco'
]

training_varLH = [
'correctedScaledActualMu',
'NvtxReco',
# "invM" ,
# "pt" ,
# "eta" ,
# "phi" ,
# "type" ,
'Z_sig',
# "isATLAS" ,
####
####
# 'muo1_PID_score',
# 'muo1_ISO_score',
'muo1_d0_d0Sig',
# 'muo1_truthPdgId',
# 'muo1_truthType',
# 'muo1_truthOrigin',
# 'muo1_truth_eta',
# 'muo1_truth_phi',
# 'muo1_truth_m',
# 'muo1_truth_px',
# 'muo1_truth_py',
# 'muo1_truth_pz',
# 'muo1_truth_E',
# 'muo1_etcone20',
# 'muo1_etcone30',
# 'muo1_etcone40',
# 'muo1_ptcone20',
# 'muo1_ptcone30',
# 'muo1_ptcone40',
# 'muo1_ptvarcone20',
# 'muo1_ptvarcone30',
# 'muo1_ptvarcone40',
# 'muo1_pt',
# 'muo1_eta',
# 'muo1_phi',
# 'muo1_charge',
# 'muo1_innerSmallHits',
# 'muo1_innerLargeHits',
# 'muo1_middleSmallHits',
# 'muo1_middleLargeHits',
# 'muo1_outerSmallHits',
# 'muo1_outerLargeHits',
# 'muo1_extendedSmallHits',
# 'muo1_extendedLargeHits',
# 'muo1_cscEtaHits',
# 'muo1_cscUnspoiledEtaHits',
# 'muo1_innerSmallHoles',
# 'muo1_innerLargeHoles',
# 'muo1_middleSmallHoles',
# 'muo1_middleLargeHoles',
# 'muo1_outerSmallHoles',
# 'muo1_outerLargeHoles',
# 'muo1_extendedSmallHoles',
# 'muo1_extendedLargeHoles',
# 'muo1_author',
# 'muo1_allAuthors',
# 'muo1_muonType',
# 'muo1_numberOfPrecisionLayers',
# 'muo1_numberOfPrecisionHoleLayers',
# 'muo1_quality',
# 'muo1_energyLossType',
# 'muo1_spectrometerFieldIntegral',
# 'muo1_scatteringCurvatureSignificance',
# 'muo1_scatteringNeighbourSignificance',
# 'muo1_momentumBalanceSignificance',
# 'muo1_segmentDeltaEta',
# 'muo1_CaloLRLikelihood',
# 'muo1_EnergyLoss',
# 'muo1_CaloMuonIDTag',
# 'muo1_DFCommonGoodMuon',
# 'muo1_DFCommonMuonsPreselection',
'muo1_LHLoose',
'muo1_LHMedium',
'muo1_LHTight',
'muo1_priTrack_d0',
'muo1_priTrack_z0',
# 'muo1_priTrack_d0Sig',
# 'muo1_priTrack_z0Sig',
# 'muo1_priTrack_theta',
# 'muo1_priTrack_qOverP',
# 'muo1_priTrack_vx',
# 'muo1_priTrack_vy',
# 'muo1_priTrack_vz',
# 'muo1_priTrack_phi',
# 'muo1_priTrack_chiSquared',
# 'muo1_priTrack_numberDoF',
# 'muo1_priTrack_radiusOfFirstHit',
# 'muo1_priTrack_trackFitter',
# 'muo1_priTrack_particleHypothesis',
# 'muo1_priTrack_numberOfUsedHitsdEdx',
# 'muo1_priTrack_numberOfContribPixelLayers',
# 'muo1_priTrack_numberOfInnermostPixelLayerHits',
# 'muo1_priTrack_expectInnermostPixelLayerHit',
# 'muo1_priTrack_numberOfNextToInnermostPixelLayerHits',
# 'muo1_priTrack_expectNextToInnermostPixelLayerHit',
# 'muo1_priTrack_numberOfPixelHits',
# 'muo1_priTrack_numberOfGangedPixels',
# 'muo1_priTrack_numberOfGangedFlaggedFakes',
# 'muo1_priTrack_numberOfPixelSpoiltHits',
# 'muo1_priTrack_numberOfDBMHits',
# 'muo1_priTrack_numberOfSCTHits',
# 'muo1_priTrack_numberOfTRTHits',
# 'muo1_priTrack_numberOfOutliersOnTrack',
# 'muo1_priTrack_standardDeviationOfChi2OS',
# 'muo1_priTrack_pixeldEdx',
# 'muo1_IDTrack_d0',
# 'muo1_IDTrack_z0',
# 'muo1_IDTrack_d0Sig',
# 'muo1_IDTrack_z0Sig',
# 'muo1_IDTrack_theta',
# 'muo1_IDTrack_qOverP',
# 'muo1_IDTrack_vx',
# 'muo1_IDTrack_vy',
# 'muo1_IDTrack_vz',
# 'muo1_IDTrack_phi',
# 'muo1_IDTrack_chiSquared',
# 'muo1_IDTrack_numberDoF',
# 'muo1_IDTrack_radiusOfFirstHit',
# 'muo1_IDTrack_trackFitter',
# 'muo1_IDTrack_particleHypothesis',
# 'muo1_IDTrack_numberOfUsedHitsdEdx',
# 'muo1_ET_Core',
# 'muo1_ET_EMCore',
# 'muo1_ET_HECCore',
# 'muo1_ET_TileCore',
# 'muo1_FSR_CandidateEnergy',
# 'muo1_InnerDetectorPt',
# 'muo1_MuonSpectrometerPt',
# 'muo1_combinedTrackOutBoundsPrecisionHits',
# 'muo1_coreMuonEnergyCorrection',
# 'muo1_deltaphi_0',
# 'muo1_deltaphi_1',
# 'muo1_deltatheta_0',
# 'muo1_deltatheta_1',
# 'muo1_etconecoreConeEnergyCorrection',
# 'muo1_extendedClosePrecisionHits',
# 'muo1_extendedOutBoundsPrecisionHits',
# 'muo1_innerClosePrecisionHits',
# 'muo1_innerOutBoundsPrecisionHits',
# 'muo1_isEndcapGoodLayers',
# 'muo1_isSmallGoodSectors',
# 'muo1_middleClosePrecisionHits',
# 'muo1_middleOutBoundsPrecisionHits',
# 'muo1_numEnergyLossPerTrack',
# 'muo1_numberOfGoodPrecisionLayers',
# 'muo1_outerClosePrecisionHits',
# 'muo1_outerOutBoundsPrecisionHits',
# 'muo1_sigmadeltaphi_0',
# 'muo1_sigmadeltaphi_1',
# 'muo1_sigmadeltatheta_0',
# 'muo1_sigmadeltatheta_1',
# 'muo1_etconeCorrBitset',
# 'muo1_neflowisol20',
# 'muo1_neflowisol30',
# 'muo1_neflowisol40',
# 'muo1_neflowisolCorrBitset',
# 'muo1_neflowisolcoreConeEnergyCorrection',
# 'muo1_ptconeCorrBitset',
# 'muo1_ptconecoreTrackPtrCorrection',
# 'muo1_topoetconeCorrBitset',
# 'muo1_topoetconecoreConeEnergyCorrection',
# 'muo1_CT_EL_Type',
# 'muo1_CT_ET_Core',
# 'muo1_CT_ET_FSRCandidateEnergy',
# 'muo1_CT_ET_LRLikelihood',
# 'muo1_d0_staco',
# 'muo1_phi0_staco',
# 'muo1_qOverPErr_staco',
# 'muo1_qOverP_staco',
# 'muo1_theta_staco',
# 'muo1_z0_staco',
####
####
# 'muo2_PID_score',
# 'muo2_ISO_score',
'muo2_d0_d0Sig',
# 'muo2_truthPdgId',
# 'muo2_truthType',
# 'muo2_truthOrigin',
# 'muo2_truth_eta',
# 'muo2_truth_phi',
# 'muo2_truth_m',
# 'muo2_truth_px',
# 'muo2_truth_py',
# 'muo2_truth_pz',
# 'muo2_truth_E',
# 'muo2_etcone20',
# 'muo2_etcone30',
# 'muo2_etcone40',
# 'muo2_ptcone20',
# 'muo2_ptcone30',
# 'muo2_ptcone40',
# 'muo2_ptvarcone20',
# 'muo2_ptvarcone30',
# 'muo2_ptvarcone40',
# 'muo2_pt',
# 'muo2_eta',
# 'muo2_phi',
# 'muo2_charge',
# 'muo2_innerSmallHits',
# 'muo2_innerLargeHits',
# 'muo2_middleSmallHits',
# 'muo2_middleLargeHits',
# 'muo2_outerSmallHits',
# 'muo2_outerLargeHits',
# 'muo2_extendedSmallHits',
# 'muo2_extendedLargeHits',
# 'muo2_cscEtaHits',
# 'muo2_cscUnspoiledEtaHits',
# 'muo2_innerSmallHoles',
# 'muo2_innerLargeHoles',
# 'muo2_middleSmallHoles',
# 'muo2_middleLargeHoles',
# 'muo2_outerSmallHoles',
# 'muo2_outerLargeHoles',
# 'muo2_extendedSmallHoles',
# 'muo2_extendedLargeHoles',
# 'muo2_author',
# 'muo2_allAuthors',
# 'muo2_muonType',
# 'muo2_numberOfPrecisionLayers',
# 'muo2_numberOfPrecisionHoleLayers',
# 'muo2_quality',
# 'muo2_energyLossType',
# 'muo2_spectrometerFieldIntegral',
# 'muo2_scatteringCurvatureSignificance',
# 'muo2_scatteringNeighbourSignificance',
# 'muo2_momentumBalanceSignificance',
# 'muo2_segmentDeltaEta',
# 'muo2_CaloLRLikelihood',
# 'muo2_EnergyLoss',
# 'muo2_CaloMuonIDTag',
# 'muo2_DFCommonGoodMuon',
# 'muo2_DFCommonMuonsPreselection',
'muo2_LHLoose',
'muo2_LHMedium',
'muo2_LHTight',
'muo2_priTrack_d0',
'muo2_priTrack_z0',
# 'muo2_priTrack_d0Sig',
# 'muo2_priTrack_z0Sig',
# 'muo2_priTrack_theta',
# 'muo2_priTrack_qOverP',
# 'muo2_priTrack_vx',
# 'muo2_priTrack_vy',
# 'muo2_priTrack_vz',
# 'muo2_priTrack_phi',
# 'muo2_priTrack_chiSquared',
# 'muo2_priTrack_numberDoF',
# 'muo2_priTrack_radiusOfFirstHit',
# 'muo2_priTrack_trackFitter',
# 'muo2_priTrack_particleHypothesis',
# 'muo2_priTrack_numberOfUsedHitsdEdx',
# 'muo2_priTrack_numberOfContribPixelLayers',
# 'muo2_priTrack_numberOfInnermostPixelLayerHits',
# 'muo2_priTrack_expectInnermostPixelLayerHit',
# 'muo2_priTrack_numberOfNextToInnermostPixelLayerHits',
# 'muo2_priTrack_expectNextToInnermostPixelLayerHit',
# 'muo2_priTrack_numberOfPixelHits',
# 'muo2_priTrack_numberOfGangedPixels',
# 'muo2_priTrack_numberOfGangedFlaggedFakes',
# 'muo2_priTrack_numberOfPixelSpoiltHits',
# 'muo2_priTrack_numberOfDBMHits',
# 'muo2_priTrack_numberOfSCTHits',
# 'muo2_priTrack_numberOfTRTHits',
# 'muo2_priTrack_numberOfOutliersOnTrack',
# 'muo2_priTrack_standardDeviationOfChi2OS',
# 'muo2_priTrack_pixeldEdx',
# 'muo2_IDTrack_d0',
# 'muo2_IDTrack_z0',
# 'muo2_IDTrack_d0Sig',
# 'muo2_IDTrack_z0Sig',
# 'muo2_IDTrack_theta',
# 'muo2_IDTrack_qOverP',
# 'muo2_IDTrack_vx',
# 'muo2_IDTrack_vy',
# 'muo2_IDTrack_vz',
# 'muo2_IDTrack_phi',
# 'muo2_IDTrack_chiSquared',
# 'muo2_IDTrack_numberDoF',
# 'muo2_IDTrack_radiusOfFirstHit',
# 'muo2_IDTrack_trackFitter',
# 'muo2_IDTrack_particleHypothesis',
# 'muo2_IDTrack_numberOfUsedHitsdEdx',
# 'muo2_ET_Core',
# 'muo2_ET_EMCore',
# 'muo2_ET_HECCore',
# 'muo2_ET_TileCore',
# 'muo2_FSR_CandidateEnergy',
# 'muo2_InnerDetectorPt',
# 'muo2_MuonSpectrometerPt',
# 'muo2_combinedTrackOutBoundsPrecisionHits',
# 'muo2_coreMuonEnergyCorrection',
# 'muo2_deltaphi_0',
# 'muo2_deltaphi_1',
# 'muo2_deltatheta_0',
# 'muo2_deltatheta_1',
# 'muo2_etconecoreConeEnergyCorrection',
# 'muo2_extendedClosePrecisionHits',
# 'muo2_extendedOutBoundsPrecisionHits',
# 'muo2_innerClosePrecisionHits',
# 'muo2_innerOutBoundsPrecisionHits',
# 'muo2_isEndcapGoodLayers',
# 'muo2_isSmallGoodSectors',
# 'muo2_middleClosePrecisionHits',
# 'muo2_middleOutBoundsPrecisionHits',
# 'muo2_numEnergyLossPerTrack',
# 'muo2_numberOfGoodPrecisionLayers',
# 'muo2_outerClosePrecisionHits',
# 'muo2_outerOutBoundsPrecisionHits',
# 'muo2_sigmadeltaphi_0',
# 'muo2_sigmadeltaphi_1',
# 'muo2_sigmadeltatheta_0',
# 'muo2_sigmadeltatheta_1',
# 'muo2_etconeCorrBitset',
# 'muo2_neflowisol20',
# 'muo2_neflowisol30',
# 'muo2_neflowisol40',
# 'muo2_neflowisolCorrBitset',
# 'muo2_neflowisolcoreConeEnergyCorrection',
# 'muo2_ptconeCorrBitset',
# 'muo2_ptconecoreTrackPtrCorrection',
# 'muo2_topoetconeCorrBitset',
# 'muo2_topoetconecoreConeEnergyCorrection',
# 'muo2_CT_EL_Type',
# 'muo2_CT_ET_Core',
# 'muo2_CT_ET_FSRCandidateEnergy',
# 'muo2_CT_ET_LRLikelihood',
# 'muo2_d0_staco',
# 'muo2_phi0_staco',
# 'muo2_qOverPErr_staco',
# 'muo2_qOverP_staco',
# 'muo2_theta_staco',
# 'muo2_z0_staco'
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


X_trainLH = data_train[training_varLH]
X_validLH = data_valid[training_varLH]


# create LGBM dataset
print("We are using the weights: regWeight_nEst10")
train_dataset = lgb.Dataset(X_train, label=y_train, weight=data_train["regWeight_nEst10"])
valid_dataset = lgb.Dataset(X_valid, label=y_valid, weight=data_valid["regWeight_nEst10"])

train_datasetLH = lgb.Dataset(X_trainLH, label=y_train, weight=data_train["regWeight_nEst10"])
valid_datasetLH = lgb.Dataset(X_validLH, label=y_valid, weight=data_valid["regWeight_nEst10"])

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
bst.save_model(args.outdir + "lgbmZ" + ".txt")

print(f"Training LH model...")

num_round=500
bstLH = lgb.train(params, train_datasetLH, num_round)

print('Saving model...')
bstLH.save_model(args.outdir + "lgbmZ_LHmodel" + ".txt")


#============================================================================
# Predict
#============================================================================

y_pred_train = bst.predict(X_train, num_iteration=bst.best_iteration)
y_pred_valid = bst.predict(X_valid, num_iteration=bst.best_iteration)

data_train["predLGBM"] = y_pred_train
data_valid["predLGBM"] = y_pred_valid

print('AUC score of prediction:')
print(f"        Training:   {roc_auc_score(y_train, y_pred_train):.6f}")
print(f"        Validation: {roc_auc_score(y_valid, y_pred_valid):.6f}")
# print('AUC score of prediction (weighted):')
# print(f"        Training:   {roc_auc_score(y_train, y_pred_train, sample_weight=data_train["weight"]):.6f}")
# print(f"        Validation: {roc_auc_score(y_valid, y_pred_valid, sample_weight=data_valid["weight"]):.6f}")


y_pred_trainLH = bstLH.predict(X_trainLH, num_iteration=bst.best_iteration)
y_pred_validLH = bstLH.predict(X_validLH, num_iteration=bst.best_iteration)

data_train["predLGBM_LH"] = y_pred_trainLH
data_valid["predLGBM_LH"] = y_pred_validLH

print('AUC score of prediction (LH model):')
print(f"        Training:   {roc_auc_score(y_train, y_pred_trainLH):.6f}")
print(f"        Validation: {roc_auc_score(y_valid, y_pred_validLH):.6f}")
# print('AUC score of prediction (weighted):')
# print(f"        Training:   {roc_auc_score(y_train, y_pred_train, sample_weight=data_train["weight"]):.6f}")
# print(f"        Validation: {roc_auc_score(y_valid, y_pred_valid, sample_weight=data_valid["weight"]):.6f}")



#%%############################################################################
#   Signal selection
###############################################################################

# First we get the ROC / AUC Scores
fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_pred_train, sample_weight=data_train["regWeight_nEst10"])
auc_train = auc(fpr_train, tpr_train)

fpr_valid, tpr_valid, thresholds_valid = roc_curve(y_valid, y_pred_valid, sample_weight=data_valid["regWeight_nEst10"])
auc_valid = auc(fpr_valid, tpr_valid)

fpr_ATLAS_train, tpr_ATLAS_train, thresholds_ATLAS_train = roc_curve(y_train, data_train['isATLAS'])#, sample_weight=data_train["revWeight_nEst100"])
auc_ATLAS_train = auc(fpr_ATLAS_train, tpr_ATLAS_train)

fpr_ATLAS_valid, tpr_ATLAS_valid, thresholds_ATLAS_valid = roc_curve(y_valid, data_valid['isATLAS'])#, sample_weight=data_valid["revWeight_nEst100"])
auc_ATLAS_valid = auc(fpr_ATLAS_valid, tpr_ATLAS_valid)

fprCompare = fpr_ATLAS_valid[1]; tprCompare = tpr_ATLAS_valid[1]


print("Running signal selection")
t = time()


# Get the same FPR to compare models

fprMask, idx, fprSel, tprSel, sigSeltrain = getSameFpr(fpr_train, tpr_train, fprCompare, thresholds_train)
print(f"The false positive rate for the training set is {fprSel}")
print(f"Yielding true positive rate {tprSel}")
print(f'This is an increase of {np.round((tprSel-tprCompare)/tprSel,4)*100} %')

print(f"Signal selection: {sigSeltrain}:")
print()


fprMask, idx, fprSel, tprSel, sigSelvalid = getSameFpr(fpr_valid, tpr_valid, fprCompare, thresholds_valid)
print(f"The false positive rate for the validation set is {fprSel}")
print(f"Yielding true positive rate {tprSel}")
print(f'This is an increase of {np.round((tprSel-tprCompare)/tprSel,4)*100} %')
print(f"Signal selection: {sigSelvalid}:")
print("")


### Trying with background instead of FPR
data_train_cut = data_train[(data_train["invM"] > 80) &  (data_train["invM"] < 100)]
data_valid_cut = data_valid[(data_valid["invM"] > 80) &  (data_valid["invM"] < 100)]

print("Now with same background as ATLAS insted of Fpr")
print("")
sel_train = getSameBkg(data_train_cut, mask_LGBM, 0.95, decimals=4)
# print(f"The false positive rate for the training set is {fpr_train[thresholds_train == sel_train]}")
# thres_mask = (np.around(thresholds_train,decimals=6) == np.around(tpr_train,decimals=6))
# print(f"Yielding true positive rate {tpr_train[thres_mask]}")
# print(f'This is an increase of {np.round((tpr_train[thres_mask]-tprCompare)/tpr_train[thres_mask],4)*100} %')
# print(f"Signal selection: {sel_train}:")
# print("")

sel_valid = getSameBkg(data_valid_cut, mask_LGBM, 0.98, decimals=4)
# thres_mask = (np.around(thresholds_valid,decimals=6) == np.around(tpr_valid,decimals=6))
# print(f"The false positive rate for the training set is {fpr_valid[thres_mask]}")
# print(f"Yielding true positive rate {tpr_valid[thres_mask]}")
# print(f'This is an increase of {np.round((tpr_valid[thres_mask]-tprCompare)/tpr_valid[thres_mask],4)*100} %')
# print(f"Signal selection: {sel_valid}:")
# print("")

### Trying with background instead of FPR
print("Now with same background as ATLAS insted of Fpr for LH")
print("")
sel_trainLH = getSameBkg(data_train_cut, mask_LGBM_LH, 0.98, decimals=4)
sel_validLH = getSameBkg(data_valid_cut, mask_LGBM_LH, 0.98, decimals=4)

### Get ROC for LH mode_cutl
fpr_trainLH, tpr_trainLH, thresholds_trainLH = roc_curve(y_train, y_pred_trainLH, sample_weight=data_train["regWeight_nEst10"])
auc_trainLH = auc(fpr_trainLH, tpr_trainLH)

fpr_validLH, tpr_validLH, thresholds_validLH = roc_curve(y_valid, y_pred_validLH, sample_weight=data_valid["regWeight_nEst10"])
auc_validLH = auc(fpr_validLH, tpr_validLH)

#============================================================================
# Finding ML cut
#============================================================================

print("Create ATLAS prediction masks with Pid")
print("")

MLpidSel_train = getMLcut(data_train_cut, GetPIDCut,-2,decimals=1)
MLpidSel_valid = getMLcut(data_valid_cut, GetPIDCut,-20,decimals=4)

data_valid['predATLASmlpid'] = 0
data_train['predATLASmlpid'] = 0
data_valid_cut['predATLASmlpid'] = 0
data_train_cut['predATLASmlpid'] = 0

data_train.loc[GetPIDCut(data_train, MLpidSel_train), ['predATLASmlpid']] = 1
data_valid.loc[GetPIDCut(data_valid, MLpidSel_valid), ['predATLASmlpid']] = 1
data_train_cut.loc[GetPIDCut(data_train_cut, MLpidSel_train), ['predATLASmlpid']] = 1
data_valid_cut.loc[GetPIDCut(data_valid_cut, MLpidSel_valid), ['predATLASmlpid']] = 1

## ML instead of ATLAS
fpr_MLPID_train, tpr_MLPID_train, thresholds_MLPID_train = roc_curve(y_train, data_train['predATLASmlpid'])#, sample_weight=data_train["regWeight_nEst10"])
auc_MLPID_train = auc(fpr_MLPID_train, tpr_MLPID_train)

fpr_MLPID_valid, tpr_MLPID_valid, thresholds_MLPID_valid = roc_curve(y_valid, data_valid['predATLASmlpid'])#, sample_weight=data_valid["regWeight_nEst10"])
auc_MLPID_valid = auc(fpr_MLPID_valid, tpr_MLPID_valid)

print("Create ATLAS prediction masks with ISO")
print("")
# We need to name the ML PID selection for the function to use
MLpidSel = MLpidSel_train
print("For training set")
MLisoSel_train = getMLcut(data_train_cut, GetPIDISOCut, -4, decimals=4)

data_train['predATLASmlisopid'] = 0
data_train.loc[GetPIDISOCut(data_train, MLisoSel_train), ['predATLASmlisopid']] = 1
data_train_cut['predATLASmlisopid'] = 0
data_train_cut.loc[GetPIDISOCut(data_train_cut, MLisoSel_train), ['predATLASmlisopid']] = 1

# And rename for the validation set
MLpidSel = MLpidSel_valid
print("For validation set")
MLisoSel_valid = getMLcut(data_valid_cut, GetPIDISOCut, -4, decimals=4)

data_valid['predATLASmlisopid'] = 0
data_valid.loc[GetPIDISOCut(data_valid, MLisoSel_valid), ['predATLASmlisopid']] = 1
data_valid_cut['predATLASmlisopid'] = 0
data_valid_cut.loc[GetPIDISOCut(data_valid_cut, MLisoSel_valid), ['predATLASmlisopid']] = 1

## ML instead of ATLAS
fpr_MLPIDISO_train, tpr_MLPIDISO_train, thresholds_MLPIDISO_train = roc_curve(y_train, data_train['predATLASmlisopid'])#, sample_weight=data_train["regWeight_nEst10"])
auc_MLPIDISO_train = auc(fpr_MLPIDISO_train, tpr_MLPIDISO_train)

fpr_MLPIDISO_valid, tpr_MLPIDISO_valid, thresholds_MLPIDISO_valid = roc_curve(y_valid, data_valid['predATLASmlisopid'])#, sample_weight=data_valid["regWeight_nEst10"])
auc_MLPIDISO_valid = auc(fpr_MLPIDISO_valid, tpr_MLPIDISO_valid)

######## Give the Z selection LH values instead of ML PID/ISO

#============================================================================
# Plotting ROC curve
#============================================================================
print(f"Plotting ROC curves")
print("")
###
# data_train["fpr_train"] = fpr_train
# data_train["tpr_train"] = tpr_train
# data_train["auc_train"] = auc_train
#
# data_valid["fpr_valid"] = fpr_valid
# data_valid["tpr_valid"] = tpr_valid
# data_valid["auc_valid"] = auc_valid
fprs = [fpr_train, fpr_trainLH]
tprs = [tpr_train, tpr_trainLH]
aucs = [auc_train, auc_trainLH]
names = ["train", "train LH model"]

fig, ax = plt.subplots(1,1, figsize=(8,5))
for fpr, tpr, auc, name in zip(fprs, tprs, aucs, names):
    ax.plot(tpr, fpr, label="LGBM {:s} (area = {:.3f})".format(name, auc))
ax.plot(tpr_ATLAS_train[1], fpr_ATLAS_train[1], 'r*', label="{:s} (area = {:.3f})".format("ATLAS_train", auc_ATLAS_train))
# ax.plot(tpr_ATLAS_valid[1], fpr_ATLAS_valid[1], 'b*', label="{:s} (area = {:.3f})".format("ATLAS_valid", auc_ATLAS_valid))

ax.plot(tpr_MLPID_train[1], fpr_MLPID_train[1], 'g*', label="{:s} (area = {:.3f})".format("ATLAS + ML PID train", auc_MLPID_train))
# ax.plot(tpr_MLPID_valid[1], fpr_MLPID_valid[1], 'm*', label="{:s} (area = {:.3f})".format("ATLAS + ML PID valid", auc_MLPID_valid))

ax.plot(tpr_MLPIDISO_train[1], fpr_MLPIDISO_train[1], 'c*', label="{:s} (area = {:.3f})".format("ATLAS + ML PID + ML ISO train", auc_MLPIDISO_train))
# ax.plot(tpr_MLPIDISO_valid[1], fpr_MLPIDISO_valid[1], 'y*', label="{:s} (area = {:.3f})".format("ATLAS + ML PID + ML ISO valid", auc_MLPIDISO_valid))

ax.set_ylabel('Background efficiency')
ax.set_xlabel('Signal efficiency')
ax.set_xlim([0.7, 1.02])
ax.set_yscale('log', nonposy='clip')
ax.legend(loc='best', prop={'size': 13})
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)
fig.tight_layout()
fig.savefig(args.outdir + "ROC_train" + ".pdf")
plt.close()


fprs = [fpr_valid, fpr_validLH]
tprs = [tpr_valid, tpr_validLH]
aucs = [auc_valid, auc_validLH]
names = ["valid", "valid LH model"]

fig, ax = plt.subplots(1,1, figsize=(8,5))
for fpr, tpr, auc, name in zip(fprs, tprs, aucs, names):
    ax.plot(tpr, fpr, label="LGBM {:s} (area = {:.3f})".format(name, auc))
# ax.plot(tpr_ATLAS_train[1], fpr_ATLAS_train[1], 'r*', label="{:s} (area = {:.3f})".format("ATLAS_train", auc_ATLAS_train))
ax.plot(tpr_ATLAS_valid[1], fpr_ATLAS_valid[1], 'b*', label="{:s} (area = {:.3f})".format("ATLAS_valid", auc_ATLAS_valid))

# ax.plot(tpr_MLPID_train[1], fpr_MLPID_train[1], 'g*', label="{:s} (area = {:.3f})".format("ATLAS + ML PID train", auc_MLPID_train))
ax.plot(tpr_MLPID_valid[1], fpr_MLPID_valid[1], 'm*', label="{:s} (area = {:.3f})".format("ATLAS + ML PID valid", auc_MLPID_valid))

# ax.plot(tpr_MLPIDISO_train[1], fpr_MLPIDISO_train[1], 'c*', label="{:s} (area = {:.3f})".format("ATLAS + ML PID + ML ISO train", auc_MLPIDISO_train))
ax.plot(tpr_MLPIDISO_valid[1], fpr_MLPIDISO_valid[1], 'y*', label="{:s} (area = {:.3f})".format("ATLAS + ML PID + ML ISO valid", auc_MLPIDISO_valid))

ax.set_ylabel('Background efficiency')
ax.set_xlabel('Signal efficiency')
ax.set_xlim([0.7, 1.02])
ax.set_yscale('log', nonposy='clip')
ax.legend(loc='best', prop={'size': 13})
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)
fig.tight_layout()
fig.savefig(args.outdir + "ROC_valid" + ".pdf")
plt.close()


#============================================================================
# Plotting SHAP values
#============================================================================

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


#============================================================================
# Plotting invMass
#============================================================================

print(f"Plotting invariant mass plots")
print("")

#data = [data_train]#, data_valid]
fig, ax = plt.subplots(figsize=(8,5))
data_train["selLGBM"] = 0
data_train.loc[mask_LGBM(data_train, sigSeltrain), ["selLGBM"]] = 1
data_train_cut["selLGBM"] = 0
data_train_cut.loc[mask_LGBM(data_train_cut, sigSeltrain), ["selLGBM"]] = 1

data_train["selLGBM2"] = 0
data_train.loc[mask_LGBM(data_train, sel_train), ["selLGBM2"]] = 1
data_train_cut["selLGBM2"] = 0
data_train_cut.loc[mask_LGBM(data_train_cut, sel_train), ["selLGBM2"]] = 1

data_train["selLGBM_LH"] = 0
data_train.loc[mask_LGBM_LH(data_train, sel_trainLH), ["selLGBM_LH"]] = 1
data_train_cut["selLGBM_LH"] = 0
data_train_cut.loc[mask_LGBM_LH(data_train_cut, sel_trainLH), ["selLGBM_LH"]] = 1


sigData = data_train.loc[(data_train["label"] == 1)]["invM"]
sigDataATLAS = data_train.loc[(data_train["selLGBM"] == 1) & (data_train["label"] == 1)]["invM"]
sigDataATLAS2 = data_train.loc[(data_train["selLGBM2"] == 1) & (data_train["label"] == 1)]["invM"]
sigDataLH = data_train.loc[(data_train["selLGBM_LH"] == 1) & (data_train["label"] == 1)]["invM"]
ATLASData = data_train.loc[(data_train["isATLAS"] == 1) & (data_train["label"] == 1)]["invM"]
PIDData = data_train.loc[(data_train["predATLASmlpid"] == 1) & (data_train["label"] == 1)]["invM"]
ISOData = data_train.loc[(data_train["predATLASmlisopid"] == 1) & (data_train["label"] == 1)]["invM"]

ax.hist(sigData, bins=120, range=(50,100),  histtype='step', color = 'k', label = "True")
ax.hist(sigDataATLAS, bins=120, range=(50,100), histtype='step', color = 'r', linestyle = 'dashed', label = "LGBM (ATLAS fpr cut)")
ax.hist(sigDataATLAS2, bins=120, range=(50,100), histtype='step', color = 'c', linestyle = 'dashed', label = "LGBM (ATLAS bkg cut)")
ax.hist(sigDataLH, bins=120, range=(50,100), histtype='step', color = 'tab:purple', label = "LGBM LH model")
ax.hist(ATLASData, bins=120, range=(50,100),  histtype='step', color = 'b', label = "ATLAS")
ax.hist(PIDData, bins=120, range=(50,100),  histtype='step', color = 'g', label = "ATLAS + ML PID")
ax.hist(ISOData, bins=120, range=(50,100),  histtype='step', color = 'm', label = "ATLAS + ML PID + ML ISO")

ax.set_xlabel("invM [GeV]")
ax.set_ylabel("Frequency")
ax.legend(loc=2, prop={'size': 13})
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)
fig.tight_layout()
fig.savefig(args.outdir + "invMdist_train" + ".pdf")


fig, ax = plt.subplots(figsize=(8,5))

data_valid["selLGBM"] = 0
data_valid.loc[mask_LGBM(data_valid, sigSelvalid), ["selLGBM"]] = 1
data_valid_cut["selLGBM"] = 0
data_valid_cut.loc[mask_LGBM(data_valid_cut, sigSelvalid), ["selLGBM"]] = 1

data_valid["selLGBM2"] = 0
data_valid.loc[mask_LGBM(data_valid, sel_valid), ["selLGBM2"]] = 1
data_valid_cut["selLGBM2"] = 0
data_valid_cut.loc[mask_LGBM(data_valid_cut, sel_valid), ["selLGBM2"]] = 1

data_valid["selLGBM_LH"] = 0
data_valid.loc[mask_LGBM_LH(data_valid, sel_validLH), ["selLGBM_LH"]] = 1
data_valid_cut["selLGBM_LH"] = 0
data_valid_cut.loc[mask_LGBM_LH(data_valid_cut, sel_validLH), ["selLGBM_LH"]] = 1


sigData = data_valid.loc[(data_valid["label"] == 1)]["invM"]
sigDataATLAS = data_valid.loc[(data_valid["selLGBM"] == 1) & (data_valid["label"] == 1)]["invM"]
sigDataATLAS2 = data_valid.loc[(data_valid["selLGBM2"] == 1) & (data_valid["label"] == 1)]["invM"]
sigDataLH = data_valid.loc[(data_valid["selLGBM_LH"] == 1) & (data_valid["label"] == 1)]["invM"]
ATLASData = data_valid.loc[(data_valid["isATLAS"] == 1) & (data_valid["label"] == 1)]["invM"]
PIDData = data_valid.loc[(data_valid["predATLASmlpid"] == 1) & (data_valid["label"] == 1)]["invM"]
ISOData = data_valid.loc[(data_valid["predATLASmlisopid"] == 1) & (data_valid["label"] == 1)]["invM"]


ax.hist(sigData, bins=120, range=(60,120),  histtype='step', color = 'k', label = "True")
# ax.hist(sigDataATLAS, bins=120, range=(60,120), histtype='step', color = 'r', label = "LGBM (ATLAS fpr cut)")
ax.hist(sigDataATLAS2, bins=120, range=(60,120), histtype='step', color = 'c', label = "LGBM (ATLAS bkg cut)")
ax.hist(sigDataLH, bins=120, range=(60,120), histtype='step', color = 'r', label = "LGBM LH model")
ax.hist(ATLASData, bins=120, range=(60,120),  histtype='step', color = 'b', label = "ATLAS")
ax.hist(PIDData, bins=120, range=(60,120),  histtype='step', color = 'g', label = "ATLAS + ML PID")
ax.hist(ISOData, bins=120, range=(60,120),  histtype='step', color = 'm', label = "ATLAS + ML PID + ML ISO")

ax.set_xlabel("invM [GeV]")
ax.set_ylabel("Frequency")
ax.legend(loc=2, prop={'size': 13})
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)
fig.tight_layout()
fig.savefig(args.outdir + "invMdist_valid" + ".pdf")

data_train.to_pickle(args.outdir + "train_data.pkl", protocol=0)
data_valid.to_pickle(args.outdir + "valid_data.pkl", protocol=0)


#%%############################################################################
#   Plot histogram to compare
###############################################################################

print(f"Plotting cutflow diagrams")
print("")

#train
sigDataTrue = data_train_cut.loc[(data_train_cut["label"] == 1)]

# sigDataATLAS = data_train_cut.loc[(data_train_cut["selLGBM"] == 1) & (data_train_cut["label"] == 1)]
# bkgDataATLAS = data_train_cut.loc[((data_train_cut["selLGBM"] == 1) & ~(data_train_cut["label"] == 1))]

sigDataATLAS2 = data_train_cut.loc[(data_train_cut["selLGBM2"] == 1) & (data_train_cut["label"] == 1)]
bkgDataATLAS2 = data_train_cut.loc[((data_train_cut["selLGBM2"] == 1) & ~(data_train_cut["label"] == 1))]

sigDataLH = data_train_cut.loc[(data_train_cut["selLGBM_LH"] == 1) & (data_train_cut["label"] == 1)]
bkgDataLH = data_train_cut.loc[(data_train_cut["selLGBM_LH"] == 1) & ~(data_train_cut["label"] == 1)]

ATLASData = data_train_cut.loc[(data_train_cut["isATLAS"] == 1) & (data_train_cut["label"] == 1)]
bkgATLASData = data_train_cut.loc[((data_train_cut["isATLAS"] == 1) & ~(data_train_cut["label"] == 1))]

PIDData = data_train_cut.loc[(data_train_cut["predATLASmlpid"] == 1) & (data_train_cut["label"] == 1)]
bkgPIDData = data_train_cut.loc[((data_train_cut["predATLASmlpid"] == 1) & ~(data_train_cut["label"] == 1))]

ISOData = data_train_cut.loc[(data_train_cut["predATLASmlisopid"] == 1) & (data_train_cut["label"] == 1)]
bkgISOData = data_train_cut.loc[((data_train_cut["predATLASmlisopid"] == 1) & ~(data_train_cut["label"] == 1))]



fig, ax = plt.subplots(figsize=(10,5))

x = np.arange(6) #How many different do we have?
# x = np.arange(3)
signals =  [ATLASData, PIDData, ISOData, sigDataATLAS2, sigDataLH, sigDataTrue] #sigDataATLAS

bkgs = [len(bkgATLASData), len(bkgPIDData), len(bkgISOData),  len(bkgDataATLAS2), len(bkgDataLH), 0] #len(bkgDataATLAS),
# signals = [len(ATLASData), len(sigDataATLAS),  len(sigDataTrue)]
# bkgs = [len(bkgATLASData), len(bkgDataATLAS),  0]

n_groups = len(x)
index = np.arange(n_groups)

bar_width = 0.4

plt.xticks(x)
# author_type = np.unique(sigDataTrue['muo1_author'], return_counts = False)
# n_all = np.zeros(shape=(len(author_type),len(index)))
#
# for s, ix in zip(signals, range(n_all.shape[0])):
#     muo1_author, n_author = np.unique(s['muo1_author'], return_counts = True)
#     for m_true, iy in zip(author_type, range(n_all.shape[1])):
#         for m, n in zip(muo1_author, n_author):
#             if m == m_true:
#                 n_all[ix, iy] = n
                    #else:
                    #    n_temp[i] = 0


# print(n_all)
# h = np.zeros(shape=(len(index)))
# for iy in range(n_all.shape[1]):
#     print(h)
#     sigs = plt.bar(index - bar_width/2, n_all[:,iy], bar_width, bottom=h)
#     h += n_all[:,iy]
#
#     for i,v in enumerate(h):
#         ax.text(x = index[i] + bar_width/2, y = v-5, s = f"n = {int(v)} ", color = 'black', ha = 'right', va = 'bottom',size = 5)

signals = [len(ATLASData), len(PIDData), len(ISOData),  len(sigDataATLAS2), len(sigDataLH), len(sigDataTrue)] #len(sigDataATLAS),
sigs = plt.bar(index - bar_width/2, signals, bar_width, color = 'r', label = 'Signal')
for i, v in enumerate(signals):
    if i != 0:
        ax.text(x = index[i] - bar_width/2, y = v + 0.5, s = f"{str(v)} (+ {np.round((v-signals[0])/signals[0] * 100, 2)} %)", color='black', ha = 'center', va = 'bottom', size = 9)
    else:
        ax.text(x = index[i] - bar_width/2, y = v + 0.5, s = f"{str(v)}", color='black', ha = 'center', va = 'bottom', size = 9)
bkg = plt.bar(index + bar_width/2, bkgs, bar_width, color = 'b', label = 'Background')

for i, v in enumerate(bkgs):
    ax.text(x = index[i] + bar_width/2, y = v + 0.5, s = str(v), color = 'black', ha = 'center', va = 'bottom',size = 9)
#plt.yscale('log')
plt.ylabel('Count')
plt.legend()

labels = [item.get_text() for item in ax.get_xticklabels()]
labels[0] = 'Cut ATLAS'
labels[1] = f'ATLAS\n + ML PID'
labels[2] = f'ATLAS\n + ML PID\n + ML ISO'
# labels[3] = f'ML Zmm\n fpr cut: {np.round(sigSeltrain,4)}'
labels[3] = f'ML Zmm\n bkg cut: {np.round(sel_train,4)}'
labels[4] = f'ML Zmm w. LH \n bkg cut: {np.round(sel_trainLH,4)}'
labels[5] = 'Zmm truth'
ax.set_xticklabels(labels)

plt.tight_layout()
fig.savefig(args.outdir + "barplot_train" + ".pdf")

print("The authors of the muons are (training set):")

muo1_authorZmm, n_authorZmm = np.unique(sigDataATLAS2['muo1_author'], return_counts = True)
muo1_authorZmmPercentage = np.array(n_authorZmm)/sum(n_authorZmm)*100
print(f"{muo1_authorZmm} with counts {n_authorZmm} for muon 1 w. Zmm ML meaning percentage of {muo1_authorZmmPercentage}")
print("")

muo1_authorATLAS, n_authorATLAS = np.unique(ATLASData['muo1_author'], return_counts = True)
muo1_authorATLASPercentage = np.array(n_authorATLAS)/sum(n_authorATLAS)*100
print(f"{muo1_authorATLAS} with counts {n_authorATLAS} for muon 1 w. ATLAS meaning percentage of {muo1_authorATLASPercentage}")
print("")

muo1_authorLH, n_authorLH = np.unique(sigDataLH['muo1_author'], return_counts = True)
muo1_authorLHPercentage = np.array(n_authorLH)/sum(n_authorLH)*100
print(f"{muo1_authorLH} with counts {n_authorLH} for muon 1 w. LH meaning percentage of {muo1_authorLHPercentage}")
print("")
#muo2_author, n_author = np.unique(data_train['muo2_author'], return_counts = True)
#print(f"{muo2_author} with counts {n_author} for muon 2")


####### valid
sigDataTrue = data_valid_cut.loc[(data_valid_cut["label"] == 1)]

# sigDataATLAS = data_valid_cut.loc[(data_valid_cut["selLGBM"] == 1) & (data_valid_cut["label"] == 1)]
# bkgDataATLAS = data_valid_cut.loc[((data_valid_cut["selLGBM"] == 1) & ~(data_valid_cut["label"] == 1))]

sigDataATLAS2 = data_valid_cut.loc[(data_valid_cut["selLGBM2"] == 1) & (data_valid_cut["label"] == 1)]
bkgDataATLAS2 = data_valid_cut.loc[((data_valid_cut["selLGBM2"] == 1) & ~(data_valid_cut["label"] == 1))]

sigDataLH = data_valid_cut.loc[(data_valid_cut["selLGBM_LH"] == 1) & (data_valid_cut["label"] == 1)]
bkgDataLH = data_valid_cut.loc[(data_valid_cut["selLGBM_LH"] == 1) & ~(data_valid_cut["label"] == 1)]

ATLASData = data_valid_cut.loc[(data_valid_cut["isATLAS"] == 1) & (data_valid_cut["label"] == 1)]
bkgATLASData = data_valid_cut.loc[((data_valid_cut["isATLAS"] == 1) & ~(data_valid_cut["label"] == 1))]

PIDData = data_valid_cut.loc[(data_valid_cut["predATLASmlpid"] == 1) & (data_valid_cut["label"] == 1)]
bkgPIDData = data_valid_cut.loc[((data_valid_cut["predATLASmlpid"] == 1) & ~(data_valid_cut["label"] == 1))]

ISOData = data_valid_cut.loc[(data_valid_cut["predATLASmlisopid"] == 1) & (data_valid_cut["label"] == 1)]
bkgISOData = data_valid_cut.loc[((data_valid_cut["predATLASmlisopid"] == 1) & ~(data_valid_cut["label"] == 1))]

fig, ax = plt.subplots(figsize=(10,5))

x = np.arange(6) #How many different do we have?
# x = np.arange(3)

signals = [ATLASData, PIDData, ISOData, sigDataATLAS2,  sigDataLH, sigDataTrue] #sigDataATLAS
bkgs = [len(bkgATLASData), len(bkgPIDData), len(bkgISOData),  len(bkgDataATLAS2), len(bkgDataLH), 0] #len(bkgDataATLAS),
# signals = [len(ATLASData), len(sigDataATLAS),  len(sigDataTrue)]
# bkgs = [len(bkgATLASData), len(bkgDataATLAS),  0]

n_groups = len(x)
index = np.arange(n_groups)

bar_width = 0.4

plt.xticks(x)
#
# author_type = np.unique(sigDataTrue['muo1_author'], return_counts = False)
# n_all = np.zeros(shape=(len(author_type),len(index)))
#
# for s, ix in zip(signals, range(n_all.shape[0])):
#     muo1_author, n_author = np.unique(s['muo1_author'], return_counts = True)
#     for m_true, iy in zip(author_type, range(n_all.shape[1])):
#         for m, n in zip(muo1_author, n_author):
#             if m == m_true:
#                 n_all[ix, iy] = n
#                     #else:
#                     #    n_temp[i] = 0
#
# print("Validation set")
# print(n_all)
# #h = 0
# h = np.zeros(shape=(len(index)))
# for iy in range(n_all.shape[1]):
#     print(h)
#     sigs = plt.bar(index - bar_width/2, n_all[:,iy], bar_width, bottom=h)
#     h += n_all[:,iy]







signals = [len(ATLASData), len(PIDData), len(ISOData),  len(sigDataATLAS2),  len(sigDataLH), len(sigDataTrue)] #len(sigDataATLAS),
sigs = plt.bar(index - bar_width/2, signals, bar_width, color = 'r', label = 'Signal')

for i, v in enumerate(signals):
    if i != 0:
        ax.text(x = index[i] - bar_width/2, y = v + 0.5, s = f"{str(v)} (+ {np.round((v-signals[0])/signals[0] * 100, 2)} %)", color='black', ha = 'center', va = 'bottom', size = 9)
    else:
        ax.text(x = index[i] - bar_width/2, y = v + 0.5, s = f"{str(v)}", color='black', ha = 'center', va = 'bottom', size = 9)

bkg = plt.bar(index + bar_width/2, bkgs, bar_width, color = 'b', label = 'Background')

for i, v in enumerate(bkgs):
   ax.text(x = index[i] + bar_width/2, y = v + 0.5, s = str(v), color = 'black', ha = 'center', va = 'bottom', size = 9)

plt.ylabel('Count')
plt.legend()

labels = [item.get_text() for item in ax.get_xticklabels()]
labels[0] = 'Cut ATLAS'
labels[1] = f'ATLAS\n + ML PID'
labels[2] = f'ATLAS\n + ML PID\n + ML ISO'
# labels[3] = f'ML Zmm\n fpr cut: {np.round(sigSelvalid,4)}'
labels[3] = f'ML Zmm\n bkg cut: {np.round(sel_valid,4)}'
labels[4] = f'ML Zmm w. LH \n bkg cut: {np.round(sel_validLH,4)}'
labels[5] = 'Zmm truth'
ax.set_xticklabels(labels)

plt.tight_layout()
fig.savefig(args.outdir + "barplot_valid" + ".pdf")

#### PLot bar plot for not cutted data


####### valid
sigDataTrue = data_valid.loc[(data_valid["label"] == 1)]

# sigDataATLAS = data_valid.loc[(data_valid["selLGBM"] == 1) & (data_valid["label"] == 1)]
# bkgDataATLAS = data_valid.loc[((data_valid["selLGBM"] == 1) & ~(data_valid["label"] == 1))]

sigDataATLAS2 = data_valid.loc[(data_valid["selLGBM2"] == 1) & (data_valid["label"] == 1)]
bkgDataATLAS2 = data_valid.loc[((data_valid["selLGBM2"] == 1) & ~(data_valid["label"] == 1))]

sigDataLH = data_valid.loc[(data_valid["selLGBM_LH"] == 1) & (data_valid["label"] == 1)]
bkgDataLH = data_valid.loc[(data_valid["selLGBM_LH"] == 1) & ~(data_valid["label"] == 1)]

ATLASData = data_valid.loc[(data_valid["isATLAS"] == 1) & (data_valid["label"] == 1)]
bkgATLASData = data_valid.loc[((data_valid["isATLAS"] == 1) & ~(data_valid["label"] == 1))]

PIDData = data_valid.loc[(data_valid["predATLASmlpid"] == 1) & (data_valid["label"] == 1)]
bkgPIDData = data_valid.loc[((data_valid["predATLASmlpid"] == 1) & ~(data_valid["label"] == 1))]

ISOData = data_valid.loc[(data_valid["predATLASmlisopid"] == 1) & (data_valid["label"] == 1)]
bkgISOData = data_valid.loc[((data_valid["predATLASmlisopid"] == 1) & ~(data_valid["label"] == 1))]

fig, ax = plt.subplots(figsize=(10,5))

x = np.arange(6) #How many different do we have?
# x = np.arange(3)

signals = [ATLASData, PIDData, ISOData, sigDataATLAS2,  sigDataLH, sigDataTrue] #sigDataATLAS
bkgs = [len(bkgATLASData), len(bkgPIDData), len(bkgISOData),  len(bkgDataATLAS2), len(bkgDataLH), 0] #len(bkgDataATLAS),
# signals = [len(ATLASData), len(sigDataATLAS),  len(sigDataTrue)]
# bkgs = [len(bkgATLASData), len(bkgDataATLAS),  0]

n_groups = len(x)
index = np.arange(n_groups)

bar_width = 0.4

plt.xticks(x)
#
# author_type = np.unique(sigDataTrue['muo1_author'], return_counts = False)
# n_all = np.zeros(shape=(len(author_type),len(index)))
#
# for s, ix in zip(signals, range(n_all.shape[0])):
#     muo1_author, n_author = np.unique(s['muo1_author'], return_counts = True)
#     for m_true, iy in zip(author_type, range(n_all.shape[1])):
#         for m, n in zip(muo1_author, n_author):
#             if m == m_true:
#                 n_all[ix, iy] = n
#                     #else:
#                     #    n_temp[i] = 0
#
# print("Validation set")
# print(n_all)
# #h = 0
# h = np.zeros(shape=(len(index)))
# for iy in range(n_all.shape[1]):
#     print(h)
#     sigs = plt.bar(index - bar_width/2, n_all[:,iy], bar_width, bottom=h)
#     h += n_all[:,iy]







signals = [len(ATLASData), len(PIDData), len(ISOData),  len(sigDataATLAS2),  len(sigDataLH), len(sigDataTrue)] #len(sigDataATLAS),
sigs = plt.bar(index - bar_width/2, signals, bar_width, color = 'r', label = 'Signal')

for i, v in enumerate(signals):
    if i != 0:
        ax.text(x = index[i] - bar_width/2, y = v + 0.5, s = f"{str(v)} (+ {np.round((v-signals[0])/signals[0] * 100, 2)} %)", color='black', ha = 'center', va = 'bottom', size = 9)
    else:
        ax.text(x = index[i] - bar_width/2, y = v + 0.5, s = f"{str(v)}", color='black', ha = 'center', va = 'bottom', size = 9)

bkg = plt.bar(index + bar_width/2, bkgs, bar_width, color = 'b', label = 'Background')

for i, v in enumerate(bkgs):
   ax.text(x = index[i] + bar_width/2, y = v + 0.5, s = str(v), color = 'black', ha = 'center', va = 'bottom', size = 9)

plt.ylabel('Count')
plt.legend()

labels = [item.get_text() for item in ax.get_xticklabels()]
labels[0] = 'Cut ATLAS'
labels[1] = f'ATLAS\n + ML PID'
labels[2] = f'ATLAS\n + ML PID\n + ML ISO'
# labels[3] = f'ML Zmm\n fpr cut: {np.round(sigSelvalid,4)}'
labels[3] = f'ML Zmm\n bkg cut: {np.round(sel_valid,4)}'
labels[4] = f'ML Zmm w. LH \n bkg cut: {np.round(sel_validLH,4)}'
labels[5] = 'Zmm truth'
ax.set_xticklabels(labels)

plt.tight_layout()
fig.savefig(args.outdir + "barplot_valid_nocut" + ".pdf")

###


print("The authors of the muons are (validation set):")

muo1_author, n_author = np.unique(data_valid['muo1_author'], return_counts = True)
print(f"{muo1_author} with counts {n_author} for muon 1")

muo2_author, n_author = np.unique(data_valid['muo2_author'], return_counts = True)
print(f"{muo2_author} with counts {n_author} for muon 2")


#%%############################################################################
#   Correlation plots
###############################################################################

print(f"Plotting correlation plots")
print("")


import scipy.stats

#### Signal ####
correlation_vars = np.copy(training_var)
correlation_vars = np.append(correlation_vars, "invM")
correlation_vars = np.append(correlation_vars, ["muo1_LHLoose", "muo1_LHMedium", "muo1_LHTight"])
correlation_vars = np.append(correlation_vars, ["muo2_LHLoose", "muo2_LHMedium", "muo2_LHTight"])

correlationSig_train = np.zeros(shape=(len(correlation_vars), len(correlation_vars)))
corrdataSig_train = data_train[data_train["label"] == 1][correlation_vars]
#corrdataSig_train = np.column_stack((data_train[data_train["label"] == 1][training_var], data_train[data_train["label"] == 1]["invM"] ))

corrdataSig_train = np.column_stack((data_train[data_train["label"] == 1][training_var], data_train[data_train["label"] == 1]["invM"],
                                    data_train[data_train["label"] == 1]["muo1_LHLoose"], data_train[data_train["label"] == 1]["muo1_LHMedium"],
                                    data_train[data_train["label"] == 1]["muo1_LHTight"], data_train[data_train["label"] == 1]["muo2_LHLoose"],
                                    data_train[data_train["label"] == 1]["muo2_LHMedium"], data_train[data_train["label"] == 1]["muo2_LHTight"] ))


for i in range(correlationSig_train.shape[0]):
    for j in range(correlationSig_train.shape[1]):
        x = corrdataSig_train[:,i]
        y = corrdataSig_train[:,j]
        nas = np.logical_or(np.isnan(x), np.isnan(y))
        infs = np.logical_or(np.isinf(x), np.isinf(y))
        correlationSig_train[i,j] = scipy.stats.pearsonr(x[~nas & ~infs], y[~nas & ~infs])[0]


correlationSig_valid = np.zeros(shape=(len(correlation_vars), len(correlation_vars)))
#corrdataSig_valid = data_valid[data_valid["label"] == 1][correlation_vars]
corrdataSig_valid = np.column_stack((data_valid[data_valid["label"] == 1][training_var], data_valid[data_valid["label"] == 1]["invM"],
                                    data_valid[data_valid["label"] == 1]["muo1_LHLoose"], data_valid[data_valid["label"] == 1]["muo1_LHMedium"],
                                    data_valid[data_valid["label"] == 1]["muo1_LHTight"], data_valid[data_valid["label"] == 1]["muo2_LHLoose"],
                                    data_valid[data_valid["label"] == 1]["muo2_LHMedium"], data_valid[data_valid["label"] == 1]["muo2_LHTight"] ))

for i in range(correlationSig_valid.shape[0]):
    for j in range(correlationSig_valid.shape[1]):
        x = corrdataSig_valid[:,i]
        y = corrdataSig_valid[:,j]
        nas = np.logical_or(np.isnan(x), np.isnan(y))
        infs = np.logical_or(np.isinf(x), np.isinf(y))
        correlationSig_valid[i,j] = scipy.stats.pearsonr(x[~nas & ~infs], y[~nas & ~infs])[0]


ticks = np.arange(0, len(training_var) + 7, 1)

fig, ax = plt.subplots(figsize=(12,12))
fig.suptitle("Correlation, signal (train)")
img = ax.imshow(correlationSig_train, cmap=plt.cm.coolwarm)
ax.set_xticks(ticks)
ax.set_xticklabels(correlation_vars, rotation=90)
ax.set_yticks(ticks)
ax.set_yticklabels(correlation_vars, rotation=0)

for (j,i),label in np.ndenumerate(correlationSig_train):
    ax.text(i,j,np.round(label,3),ha='center',va='center', size=8)

fig.tight_layout()
fig.colorbar(img, ax=ax);

fig.savefig(args.outdir + "Correlation_Sig_train" + ".pdf")


fig, ax = plt.subplots(figsize=(12,12))
fig.suptitle("Correlation, signal (valid)")
img = ax.imshow(correlationSig_valid, cmap=plt.cm.coolwarm)
ax.set_xticks(ticks)
ax.set_xticklabels(correlation_vars, rotation=90)
ax.set_yticks(ticks)
ax.set_yticklabels(correlation_vars, rotation=0)
for (j,i),label in np.ndenumerate(correlationSig_valid):
    ax.text(i,j,np.round(label,3),ha='center',va='center', size=8)

fig.tight_layout()
fig.colorbar(img, ax=ax);

fig.savefig(args.outdir + "Correlation_Sig_valid" + ".pdf")

#### Background ####

correlationBkg_train = np.zeros(shape=(len(correlation_vars), len(correlation_vars)))
#corrdataBkg_train = data_train[data_train["label"] == 0][correlation_vars]

corrdataBkg_train = np.column_stack((data_train[data_train["label"] == 0][training_var], data_train[data_train["label"] == 0]["invM"],
                                    data_train[data_train["label"] == 0]["muo1_LHLoose"], data_train[data_train["label"] == 0]["muo1_LHMedium"],
                                    data_train[data_train["label"] == 0]["muo1_LHTight"], data_train[data_train["label"] == 0]["muo2_LHLoose"],
                                    data_train[data_train["label"] == 0]["muo2_LHMedium"], data_train[data_train["label"] == 0]["muo2_LHTight"] ))

for i in range(correlationBkg_train.shape[0]):
    for j in range(correlationBkg_train.shape[1]):
        x = corrdataBkg_train[:,i]
        y = corrdataBkg_train[:,j]
        nas = np.logical_or(np.isnan(x), np.isnan(y))
        infs = np.logical_or(np.isinf(x), np.isinf(y))
        correlationBkg_train[i,j] = scipy.stats.pearsonr(x[~nas & ~infs], y[~nas & ~infs])[0]


correlationBkg_valid = np.zeros(shape=(len(correlation_vars), len(correlation_vars)))
#corrdataBkg_valid = data_valid[data_valid["label"] == 0][correlation_vars]

corrdataBkg_valid = np.column_stack((data_valid[data_valid["label"] == 0][training_var], data_valid[data_valid["label"] == 0]["invM"],
                                    data_valid[data_valid["label"] == 0]["muo1_LHLoose"], data_valid[data_valid["label"] == 0]["muo1_LHMedium"],
                                    data_valid[data_valid["label"] == 0]["muo1_LHTight"], data_valid[data_valid["label"] == 0]["muo2_LHLoose"],
                                    data_valid[data_valid["label"] == 0]["muo2_LHMedium"], data_valid[data_valid["label"] == 0]["muo2_LHTight"] ))

for i in range(correlationBkg_valid.shape[0]):
    for j in range(correlationBkg_valid.shape[1]):
        x = corrdataBkg_valid[:,i]
        y = corrdataBkg_valid[:,j]
        nas = np.logical_or(np.isnan(x), np.isnan(y))
        infs = np.logical_or(np.isinf(x), np.isinf(y))
        correlationBkg_valid[i,j] = scipy.stats.pearsonr(x[~nas & ~infs], y[~nas & ~infs])[0]

ticks = np.arange(0, len(training_var) + 7, 1)

fig, ax = plt.subplots(figsize=(12,12))
fig.suptitle("Correlation, background (train)")
img = ax.imshow(correlationBkg_train, cmap=plt.cm.coolwarm)
ax.set_xticks(ticks)
ax.set_xticklabels(correlation_vars, rotation=90)
ax.set_yticks(ticks)
ax.set_yticklabels(correlation_vars, rotation=0)
for (j,i),label in np.ndenumerate(correlationBkg_train):
    ax.text(i,j,np.round(label,3),ha='center',va='center', size=8)

fig.tight_layout()
fig.colorbar(img, ax=ax);

fig.savefig(args.outdir + "Correlation_Bkg_train" + ".pdf")


fig, ax = plt.subplots(figsize=(12,12))
fig.suptitle("Correlation, background (valid)")
img = ax.imshow(correlationBkg_valid, cmap=plt.cm.coolwarm)
ax.set_xticks(ticks)
ax.set_xticklabels(correlation_vars, rotation=90)
ax.set_yticks(ticks)
ax.set_yticklabels(correlation_vars, rotation=0)
for (j,i),label in np.ndenumerate(correlationBkg_valid):
    ax.text(i,j,np.round(label,3),ha='center',va='center', size=8)

fig.tight_layout()
fig.colorbar(img, ax=ax);

fig.savefig(args.outdir + "Correlation_Bkg_valid" + ".pdf")


log.info(f"Done! Total time: {timedelta(seconds=time() - t_start)}")
