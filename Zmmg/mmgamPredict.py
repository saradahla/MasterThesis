#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Tuesday 30 June
@author: Sara Dahl Andersen

Running my Zmmg model for muons and photons and outputting the score

nohup python -u mmgamPredict.py --tag 20201029_HZgam --model output/ZModels/20201029_2/lgbmZmmg.txt output/ZmmgReweightFiles/20201029_HZgam/combined_HZgam20201029_train.h5 2>&1 &> output/logHZgamPred.txt & disown
nohup python -u mmgamPredict.py --tag 20201103_ZgamData --data 1 --model output/ZModels/20201103/lgbmZmmg.txt output/ZmmgReweightFiles/20201030_ZgamData/combined_Zmmgam20201030.h5 2>&1 &> output/logZgamDataPred.txt & disown

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
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
import shap
import lightgbm as lgb




# Logging style and level
log.basicConfig(format='[%(levelname)s] %(message)s', level=log.INFO)
log.info("Packages imported")

# Start "timer"
t_start = time()


# Command line options
parser = argparse.ArgumentParser(description="Combine datafiles, reweigh data and add columns.")
parser.add_argument('--outdir', action='store', default="output/mumugamPredictions/", type=str,
                    help='Output directory.')
parser.add_argument('path', type=str, nargs='+',
                    help='HDF5 file(s) to use.')
parser.add_argument('--tag', action='store', type=str, required=False, default="",
                    help='Tag the data category (Zee, Wev, etc.).')
parser.add_argument('--njobs', action='store', default=10, type=int,
                    help='Maximum number of concurrent processes to use.')
parser.add_argument('--model', action='store', type=str,
                    help='Trained LGBM model to predict from')
parser.add_argument('--data', action='store', type=int, default=0,
                    help='Do we have a truth?')


args = parser.parse_args()

# Validate arguments
if not args.path:
    log.error("No HDF5 file was specified.")
    quit()

if not args.model:
    log.error("No trained model specified.")
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
    return ( ( np.sign(data['muo1_charge'])*np.sign(data['muo2_charge']) == -1 ) & #opposite sign
             ( (data['muo1_pt']) > 10 ) &
             ( (data['muo2_pt']) > 10 ) &
             ( (np.abs( data['muo1_eta']) < 2.7)) &
             ( (np.abs( data['muo2_eta']) < 2.7)) &
             ( data['muo1_LHMedium'] * data['muo2_LHMedium'] ) &
             ( abs(data['muo1_d0_d0Sig']) < 3 ) &
             ( abs(data['muo2_d0_d0Sig']) < 3 ) &
             ### photon cuts
             ( abs(data['pho_et'])*1000 > 10 ) &
             ( (np.abs( data['pho_eta'] ) < 1.37) | ((np.abs( data['pho_eta'] ) > 1.52) & (np.abs( data['pho_eta'] ) < 2.37))) &
             ( data['pho_isPhotonEMTight'] )
             )
#
# def GetPIDCut(data, MLpidSel):
#     return ( ( np.sign(data['muo1_truthPdgId'])*np.sign(data['muo2_truthPdgId']) == -1 ) & #opposite sign
#              ( data['muo1_pt']/1000 > 10 ) &
#              ( data['muo2_pt']/1000 > 10 ) &
#              ( (np.abs( data['muo1_eta'])<2.7)) &
#              ( (np.abs( data['muo2_eta'])<2.7)) &
#              ( data['muo1_PID_score'] > MLpidSel ) &
#              ( data['muo2_PID_score'] > MLpidSel ) &
#              ( abs(data['muo1_d0_d0Sig']) < 3 ) &
#              ( abs(data['muo2_d0_d0Sig']) < 3 )
#              )
#
# def GetPIDISOCut(data, MLisoSel):
#     return ( ( np.sign(data['muo1_truthPdgId'])*np.sign(data['muo2_truthPdgId']) == -1 ) & #opposite sign
#              ( data['muo1_pt']/1000 > 10 ) &
#              ( data['muo2_pt']/1000 > 10 ) &
#              ( (np.abs( data['muo1_eta'])<2.7)) &
#              ( (np.abs( data['muo2_eta'])<2.7)) &
#              ( data['muo1_PID_score'] > MLpidSel ) &
#              ( data['muo2_PID_score'] > MLpidSel ) &
#              ( abs(data['muo1_d0_d0Sig']) < 3 ) &
#              ( abs(data['muo2_d0_d0Sig']) < 3 ) &
#              ( data['muo1_ISO_score'] > MLisoSel ) &
#              ( data['muo2_ISO_score'] > MLisoSel )
#              )


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

    #fprSel, tprSel, thresholdsSel = roc_curve(y_valid, predSel[:][0], sample_weight=data_valid["regWeight_nEst30"])
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
        # fprSel, tprSel, thresholdsSel = roc_curve(y_valid, predSel[:][0], sample_weight=data_valid["regWeight_nEst30"])
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
data_pred = data_get

# is the pt for muons in MeV or GeV?
GeV = (np.mean(data_pred["muo1_pt"]) > 1000)
if not GeV:
    data_pred["muo1_pt"] = data_pred["muo1_pt"]/1000

data_pred["isATLAS"] = GetATLASCut(data_pred)


# Check shapes

if args.data == 0:
    shapeAll = np.shape(data_pred)
    shapeSig = np.shape(data_pred[data_pred["label"] == 1])
    shapeBkg = np.shape(data_pred[data_pred["label"] == 0])

    log.info(f"Shape:       {shapeAll}")
    log.info(f"Shape sig: {shapeSig}, bkg: {shapeBkg}")
elif args.data == 1:
    log.info(f"The file type is Data, so we do not have a label")

# =========================
#       Variables
# =========================

truth_var = "label"
training_var = [
# 'correctedScaledAverageMu',
# 'correctedScaledActualMu',
# 'NvtxReco',
# "invM" ,
# "pt" ,
# "eta" ,
# "phi" ,
# "type" ,
'Z_sig',
# "isATLAS" ,
'Z_score',
'pho_PID_score',
'pho_ISO_score',
'pho_isConv',
####
####
# 'muo1_PID_score',
# 'muo1_ISO_score',
# 'muo1_d0_d0Sig',
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
# 'muo1_priTrack_d0',
# 'muo1_priTrack_z0',
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
# 'muo2_d0_d0Sig',
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
# 'muo2_priTrack_d0',
# 'muo2_priTrack_z0',
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


X = data_pred[training_var]
if args.data == 0:
    y = data_pred[truth_var]

# create LGBM dataset
if args.data == 0:
    pred_dataset = lgb.Dataset(X, label=y)

elif args.data == 1:
    pred_dataset = lgb.Dataset(X)

#============================================================================
# Train the model
#============================================================================

print(f"Importing the model...")
bst = lgb.Booster(model_file = args.model)

#============================================================================
# Predict
#============================================================================

y_pred = bst.predict(X, num_iteration=bst.best_iteration)

data_pred["predLGBM"] = y_pred

if args.data == 0:
    print('AUC score of prediction:')
    print(f"{roc_auc_score(y, y_pred):.6f}")
# print('AUC score of prediction (weighted):')
# print(f"        Training:   {roc_auc_score(y_train, y_pred_train, sample_weight=data_train["weight"]):.6f}")
# print(f"        Validation: {roc_auc_score(y_valid, y_pred_valid, sample_weight=data_valid["weight"]):.6f}")


#%%############################################################################
#   Signal selection
###############################################################################
if args.data == 0:
    # First we get the ROC / AUC Scores
    fpr_pred, tpr_pred, thresholds_pred = roc_curve(y, y_pred)
    auc_pred = auc(fpr_pred, tpr_pred)

    fpr_ATLAS, tpr_ATLAS, thresholds_ATLAS = roc_curve(y, data_pred['isATLAS'])#, sample_weight=data["revWeight_nEst100"])
    auc_ATLAS = auc(fpr_ATLAS, tpr_ATLAS)

    fprCompare = fpr_ATLAS[1]; tprCompare = tpr_ATLAS[1]

    print("Running signal selection")
    t = time()


    # Get the same FPR to compare models

    fprMask, idx, fprSel, tprSel, sigSel = getSameFpr(fpr_pred, tpr_pred, fprCompare, thresholds_pred)
    print(f"The false positive rate for the training set is {fprSel}")
    print(f"Yielding true positive rate {tprSel}")
    print(f'This is an increase of {np.round((tprSel-tprCompare)/tprSel,4)*100} %')

    print(f"Signal selection: {sigSel}:")
    print()

    ### Trying with background instead of FPR
    # data_train_cut = data_train[(data_train["invM"] > 80) &  (data_train["invM"] < 100)]
    # data_valid_cut = data_valid[(data_valid["invM"] > 80) &  (data_valid["invM"] < 100)]

    print("Now with same background as ATLAS insted of Fpr")
    print("")
    sel_pred = getSameBkg(data_pred, mask_LGBM, 0.97, decimals=4)


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
    fprs = [fpr_pred]
    tprs = [tpr_pred]
    aucs = [auc_pred]
    names = ["HZgam"]

    fig, ax = plt.subplots(1,1, figsize=(8,5))
    for fpr, tpr, auc, name in zip(fprs, tprs, aucs, names):
        ax.plot(tpr, fpr, label="LGBM {:s} (area = {:.3f})".format(name, auc))
    ax.plot(tpr_ATLAS[1], fpr_ATLAS[1], 'r*', label="{:s} (area = {:.3f})".format("ATLAS_train", auc_ATLAS))

    ax.set_ylabel('Background efficiency')
    ax.set_xlabel('Signal efficiency')
    ax.set_xlim([0, 1.02])
    # ax.set_yscale('log', nonposy='clip')
    ax.legend(loc='best', prop={'size': 13})
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)
    fig.tight_layout()
    fig.savefig(args.outdir + "ROC" + ".pdf")
    plt.close()

elif args.data == 1:
    # sel_pred = 0.5
    sel_pred = 0.96

#============================================================================
# Plotting SHAP values
#============================================================================
#
# shap_values_train = shap.TreeExplainer(bst).shap_values(X)
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
# fig_shap.savefig(args.outdir + 'featureImportance_SHAP' + '.pdf')

#============================================================================
# Plotting invMass
#============================================================================

print(f"Plotting invariant mass plots")
print("")

#data = [data_train]#, data_valid]
fig, ax = plt.subplots(figsize=(8,5))
# data_pred["selLGBM"] = 0
# data_pred.loc[mask_LGBM(data_pred, sigSel), ["selLGBM"]] = 1

data_pred["selLGBM2"] = 0
data_pred.loc[mask_LGBM(data_pred, sel_pred), ["selLGBM2"]] = 1

if args.data == 0:
    sigData = data_pred.loc[(data_pred["label"] == 1)]["invM"]
# sigDataATLAS = data_pred.loc[(data_pred["selLGBM"] == 1) & (data_pred["label"] == 1)]["invM"]
    sigDataATLAS2 = data_pred.loc[(data_pred["selLGBM2"] == 1) & (data_pred["label"] == 1)]["invM"]
    ATLASData = data_pred.loc[(data_pred["isATLAS"] == 1) & (data_pred["label"] == 1)]["invM"]
elif args.data == 1:
    sigDataATLAS2 = data_pred.loc[(data_pred["selLGBM2"] == 1)]["invM"]
    ATLASData = data_pred.loc[(data_pred["isATLAS"] == 1)]["invM"]
# PIDData = data_train.loc[(data_train["predATLASmlpid"] == 1) & (data_train["label"] == 1)]["invM"]
# ISOData = data_train.loc[(data_train["predATLASmlisopid"] == 1) & (data_train["label"] == 1)]["invM"]

if args.data == 0:
    ax.hist(sigData, bins=120, range=(50,200),  histtype='step', color = 'k', label = "True")
# ax.hist(sigDataATLAS, bins=120, range=(50,200), histtype='step', color = 'r', linestyle = 'dashed', label = "LGBM (ATLAS fpr cut)")
ax.hist(sigDataATLAS2, bins=120, range=(50,200), histtype='step', color = 'c', linestyle = 'dashed', label = "LGBM (ATLAS bkg cut)")
# ax.hist(sigDataLH, bins=120, range=(50,200), histtype='step', color = 'tab:purple', label = "LGBM LH model")
ax.hist(ATLASData, bins=120, range=(50,200),  histtype='step', color = 'b', label = "ATLAS")
# ax.hist(PIDData, bins=120, range=(50,200),  histtype='step', color = 'g', label = "ATLAS + ML PID")
# ax.hist(ISOData, bins=120, range=(50,200),  histtype='step', color = 'm', label = "ATLAS + ML PID + ML ISO")

ax.set_xlabel("invM [GeV]")
ax.set_ylabel("Frequency")
ax.legend(loc=2, prop={'size': 13})
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)
fig.tight_layout()
fig.savefig(args.outdir + "invMdist" + ".pdf")

data_pred.to_pickle(args.outdir + "pred_data.pkl", protocol=0)

#%%############################################################################
#   Plot histogram to compare
###############################################################################
#
# print(f"Plotting cutflow diagrams")
# print("")
#
# #train
if args.data == 0:
    sigDataTrue = data_pred.loc[(data_pred["label"] == 1)]

    sigDataATLAS2 = data_pred.loc[(data_pred["selLGBM2"] == 1) & (data_pred["label"] == 1)]
    bkgDataATLAS2 = data_pred.loc[((data_pred["selLGBM2"] == 1) & ~(data_pred["label"] == 1))]
    ATLASData = data_pred.loc[(data_pred["isATLAS"] == 1) & (data_pred["label"] == 1)]
    bkgATLASData = data_pred.loc[((data_pred["isATLAS"] == 1) & ~(data_pred["label"] == 1))]


    fig, ax = plt.subplots(figsize=(10,5))
    #
    x = np.arange(3) #How many different do we have?

    signals =  [ATLASData, sigDataATLAS2, sigDataTrue] #sigDataATLAS
    bkgs = [len(bkgATLASData), len(bkgDataATLAS2),  0]
    #
    n_groups = len(x)
    index = np.arange(n_groups)

    bar_width = 0.4
    #
    plt.xticks(x)
    signals = [len(ATLASData), len(sigDataATLAS2), len(sigDataTrue)] #len(sigDataATLAS),
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
    #
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[0] = 'Cut ATLAS'
    # labels[1] = f'ATLAS\n + ML PID'
    # labels[2] = f'ATLAS\n + ML PID\n + ML ISO'
    # # labels[3] = f'ML Zmm\n fpr cut: {np.round(sigSeltrain,4)}'
    labels[1] = f'ML Zmm\n bkg cut: {np.round(sel_pred,4)}'
    # labels[4] = f'ML Zmm w. LH \n bkg cut: {np.round(sel_trainLH,4)}'
    labels[2] = 'Zmm truth'
    ax.set_xticklabels(labels)
    #
    plt.tight_layout()
    fig.savefig(args.outdir + "barplot" + ".pdf")

log.info(f"Done! Total time: {timedelta(seconds=time() - t_start)}")
