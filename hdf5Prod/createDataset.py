#!/usr/bin/env python
# -*- coding: utf-8 -*-
print("Program running...")

import warnings
warnings.filterwarnings('ignore', 'ROOT .+ is currently active but you ')
warnings.filterwarnings('ignore', 'numpy .+ is currently installed but you ')

import h5py
import numpy as np
import logging as log
import argparse
import os
import matplotlib.pyplot as plt

from utils import mkdir
from itertools import combinations
from skhep.math import vectors
import multiprocessing

from time import time
from datetime import timedelta

import lightgbm as lgb


# Logging style and level
log.basicConfig(format='[%(levelname)s] %(message)s', level=log.INFO)
log.info("Packages imported")

# Start "timer"
t_start = time()

# Command line options
parser = argparse.ArgumentParser(description="Extract data from HDF5 files into flat HDF5 files for training.")
parser.add_argument('--tag', action='store', type=str, required=True,
                    help='Tag the data category (Zee, Wev, etc.).')
parser.add_argument('--outdir', action='store', default="output/MuoPairHdf5/", type=str,
                    help='Output directory.')
#parser.add_argument('--modeldir', action='store', default="output/MuoModels/", type=str,
#                    help='Directory with PID and ISO models.')
parser.add_argument('paths', type=str, nargs='+',
                    help='ROOT file(s) to be converted.')
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

#============================================================================
# Functions
#============================================================================



def signalSelection(hf, event, comb):
    # Do the muons originate from a Z boson?
    isZmm_0 = (hf['muo_truthOrigin'][event][comb[0]] == 13)
    isZmm_1 = (hf['muo_truthOrigin'][event][comb[1]] == 13)
    # Do the muons have opposite sign?
    muo0_PdgId = hf['muo_truthPdgId'][event][comb[0]]
    muo1_PdgId = hf['muo_truthPdgId'][event][comb[1]]
    # Does the PgdId indicate that they are muons? (PgdId = +-13)
    muo0_isMuo = (np.abs(muo0_PdgId) == 13)
    muo1_isMuo = (np.abs(muo1_PdgId) == 13)


    if (isZmm_0*isZmm_1): # Both origin from a Z

        bothMuo = muo0_isMuo*muo1_isMuo
        oppositeSign = (muo0_PdgId*muo1_PdgId < 0)

        if (bothMuo*oppositeSign):
            #nType[0,0] += 1
            return 1 # Signal
        else:
            #nType[0,1] += 1
            return 2 # Trash

    else:
        if (isZmm_0+isZmm_1>0): # One origins from a Z
            #nType[0,1] += 1
            return 2 # Trash
        elif (muo0_isMuo+muo1_isMuo==2): # Both are muons
            #nType[0,2] += 1
            return 3 # Bkg2Muo
        elif (muo0_isMuo+muo1_isMuo==1): # One is an muon
            #nType[0,3] += 1
            return 4 # Bkg1Muo
        else: # No muons
            #nType[0,4] += 1
            return 5 # Bkg0Muo




def invMass(hf, event, comb):
    # Calculate mZee using: https://github.com/scikit-hep/scikit-hep/blob/master/skhep/math/vectors.py?fbclid=IwAR3C0qnNlxKx-RhGjwo1c1FeZEpWbYqFrNmEqMv5iE-ibyPw_xEqmDYgRpc
    # Get variables
    p1 = hf['muo_pt'][event][comb[0]]
    eta1 = hf['muo_eta'][event][comb[0]]
    phi1 = hf['muo_phi'][event][comb[0]]
    p2 = hf['muo_pt'][event][comb[1]]
    eta2 = hf['muo_eta'][event][comb[1]]
    phi2 = hf['muo_phi'][event][comb[1]]

    # make four vector
    vecFour1 = vectors.LorentzVector()
    vecFour2 = vectors.LorentzVector()

    vecFour1.setptetaphim(p1/1000,eta1,phi1,0.105) #Units in GeV for pt and mass
    vecFour2.setptetaphim(p2/1000,eta2,phi2,0.105)

    # calculate invariant mass
    vecFour = vecFour1+vecFour2
    invM = vecFour.mass
    et = vecFour.et
    eta = vecFour.eta
    phi = vecFour.phi()

    return invM, et, eta, phi

def isATLAS(hf, event, muo1, muo2):

    pT0 = hf["muo_pt"][ event ][ muo1 ]
    pT1 = hf["muo_pt"][ event ][ muo2 ]

    eta0 = hf["muo_eta"][ event ][ muo1 ]
    eta1 = hf["muo_eta"][ event ][ muo2 ]

    isMedium0 = hf["muo_LHMedium"][ event ][ muo1 ]
    isMedium1 = hf["muo_LHMedium"][ event ][ muo2 ]

    d0Sigd0_0 = abs(hf["muo_priTrack_d0"][ event ][ muo1 ]) / abs(hf["muo_priTrack_d0Sig"][ event ][ muo1 ])
    d0Sigd0_1 = abs(hf["muo_priTrack_d0"][ event ][ muo2 ]) / abs(hf["muo_priTrack_d0Sig"][ event ][ muo2 ])

    if (pT0/1000 > 10) and (pT1/1000 > 10) and (abs(eta0) < 2.7) and (abs(eta1) < 2.7) and (isMedium0*isMedium1) and (d0Sigd0_0 < 3) and (d0Sigd0_1 < 3):
        return 1
    else:
        return 0

def addMuonVariables(hf, event, data_temp, muoNr, muo):
    """
    Takes variables from file and adds them to a temporary array, that is later
    appended to the returned data.

    Arguments:
        hf: File to get variables from.
        event: Event number.
        data_temp: Numpy array to add variables to.
        eleNr: Used for naming variables. (1=tag, 2=probe)
        ele: Electron index.

    Returns:
        Nothing. Data is set in existing array.

    """

    data_temp[ 0, column_names.index( f'muo{muoNr}_truthPdgId' ) ] = hf[ 'muo_truthPdgId' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_truthType' ) ] = hf[ 'muo_truthType' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_truthOrigin' ) ] = hf[ 'muo_truthOrigin' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo{muoNr}_etcone20' ) ] = hf[ 'muo_etcone20' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_etcone30' ) ] = hf[ 'muo_etcone30' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_etcone40' ) ] = hf[ 'muo_etcone40' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_ptcone20' ) ] = hf[ 'muo_ptcone20' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_ptcone30' ) ] = hf[ 'muo_ptcone30' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_ptcone40' ) ] = hf[ 'muo_ptcone40' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo{muoNr}_ptvarcone20' ) ] = hf[ 'muo_ptvarcone20' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_ptvarcone30' ) ] = hf[ 'muo_ptvarcone30' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_ptvarcone40' ) ] = hf[ 'muo_ptvarcone40' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo{muoNr}_muonType' ) ] = hf[ 'muo_muonType' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_numberOfPrecisionLayers' ) ] = hf[ 'muo_numberOfPrecisionLayers' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_numberOfPrecisionHoleLayers' ) ] = hf[ 'muo_numberOfPrecisionHoleLayers' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_quality' ) ] = hf[ 'muo_quality' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo{muoNr}_LHLoose' ) ] = hf[ 'muo_LHLoose' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_LHMedium' ) ] = hf[ 'muo_LHMedium' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_LHTight' ) ] = hf[ 'muo_LHTight' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo{muoNr}_trigger' ) ] = hf[ 'muo_trigger' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo{muoNr}_pt' ) ] = hf[ 'muo_pt' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_eta' ) ] = hf[ 'muo_eta' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_phi' ) ] = hf[ 'muo_phi' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo{muoNr}_ET_TileCore' ) ] = hf[ 'muo_ET_TileCore' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_MuonSpectrometerPt' ) ] = hf[ 'muo_MuonSpectrometerPt' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo{muoNr}_deltaphi_0' ) ] = hf[ 'muo_deltaphi_0' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_deltaphi_1' ) ] = hf[ 'muo_deltaphi_1' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_deltatheta_0' ) ] = hf[ 'muo_deltatheta_0' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_deltatheta_1' ) ] = hf[ 'muo_deltatheta_1' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo{muoNr}_etconecoreConeEnergyCorrection' ) ] = hf[ 'muo_etconecoreConeEnergyCorrection' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo{muoNr}_sigmadeltaphi_0' ) ] = hf[ 'muo_sigmadeltaphi_0' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_sigmadeltaphi_1' ) ] = hf[ 'muo_sigmadeltaphi_1' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_sigmadeltatheta_0' ) ] = hf[ 'muo_sigmadeltatheta_0' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_sigmadeltatheta_1' ) ] = hf[ 'muo_sigmadeltatheta_1' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo{muoNr}_neflowisolcoreConeEnergyCorrection' ) ] = hf[ 'muo_neflowisolcoreConeEnergyCorrection' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_ptconecoreTrackPtrCorrection' ) ] = hf[ 'muo_ptconecoreTrackPtrCorrection' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_topoetconecoreConeEnergyCorrection' ) ] = hf[ 'muo_topoetconecoreConeEnergyCorrection' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo{muoNr}_author' ) ] = hf[ 'muo_author' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_allAuthors' ) ] = hf[ 'muo_allAuthors' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_priTrack_d0' ) ] = hf[ 'muo_priTrack_d0' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_priTrack_z0' ) ] = hf[ 'muo_priTrack_z0' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_priTrack_d0Sig' ) ] = hf[ 'muo_priTrack_d0Sig' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_priTrack_z0Sig' ) ] = hf[ 'muo_priTrack_z0Sig' ][ event ][ muo ]



def MakeFiles(arguments):
    """
    Extracts files and determine sig/bkg events.
    Arguments:
        tree: the root tree
        start: event index in file to start at
        stop: event index in file to stop at

    Returns:
        Data of muon pairs in array

    """
    # Unpack arguments
    process, counter, path, start, stop = arguments

    log.info("[{}]  Importing data from {}".format(process,path))
    hf = h5py.File(path, "r")

    data = np.empty((0,len(column_names)), float)

    # Total number of events in batch
    n_events = stop-start

    for i, event in enumerate(np.arange(start,stop)):
        # Print information on progress
        if i%100==0:
            log.info("[{}]  {} of {} events examined".format(process,i,n_events))

        # Number of muons in event
        nMuo = np.shape(hf[ 'muo_truthType' ][ event ])[0]

        if (nMuo >= 2):
            # Create all pairs of muons
            muo_index = np.arange(0, nMuo,1)
            comb = np.asarray(list(combinations(muo_index, 2)))

            # Shuffle random indexes
            n_range = len(comb)                             # Range to shuffle in, eg. 1 to 5
            n_shuffle = np.random.randint(len(comb))        # Number of combinations to shuffle
            idx = np.random.choice(n_range,n_shuffle)       # Index'
            comb[idx] = comb[idx][:,::-1]
            data_temp = np.zeros((1,len(column_names)))

            for c in comb:
                #try:
                selection = signalSelection(hf, event, c)

                if selection == 2:
                    continue #trash Muon

                invM, et, eta, phi = invMass(hf, event, c)

                # Add event variables to array
                data_temp[ 0, column_names.index( 'NvtxReco' ) ] = np.int(hf['NvtxReco'][event])
                data_temp[ 0, column_names.index( 'correctedScaledAverageMu' ) ] = hf[ 'correctedScaledAverageMu' ][ event ]
                data_temp[ 0, column_names.index( 'invM' ) ] = invM
                data_temp[ 0, column_names.index( 'et' ) ] = et
                data_temp[ 0, column_names.index( 'eta' ) ] = eta
                data_temp[ 0, column_names.index( 'phi' ) ] = phi
                data_temp[ 0, column_names.index( 'type' ) ] = selection
                # is it passing the ATLAS cut?
                data_temp[ 0, column_names.index( 'isATLAS' ) ] = isATLAS(hf, event, c[0], c[1])
                    # Add muon variables to array
                addMuonVariables(hf, event, data_temp, 1, c[0])
                addMuonVariables(hf, event, data_temp, 2, c[1])

                data = np.append(data, data_temp, axis=0)

                #except:
                #    continue
    return data
                    #print(f"Found signal with combination {c}, mass = {mass}")



def saveToFile(fname, data, column_names, column_dtype):
    """
    Simply saves data to fname.

    Arguments:
        fname: filename (directory is taken from script's args).
        data: numpy array.
        column_names: names of each column.
        column_dtype: dtype of column.

    Returns:
        Nothing.
    """
    log.info("Saving to {}".format(args.outdir + fname))
    with h5py.File(args.outdir + fname, 'w') as hf:
        for var in column_names:
            hf.create_dataset( f'{var}',
                              data=data[:,column_names.index(f'{var}')],
                              dtype=column_dtype[f'{var}'],
                              chunks=True,
                              maxshape= (None,),
                              compression='lzf')

def appendToFile(fname, data, column_names, column_dtype):
    """
    Simply appends data to fname.

    Arguments:
        fname: filename (directory is taken from script's args).
        data: numpy array.
        column_names: names of each column.
        column_dtype: dtype of column.

    Returns:
        Nothing.
    """
    log.info("Appending to {}".format(args.outdir + fname))
    with h5py.File(args.outdir + fname, 'a') as hf:
        for var in column_names:

            array = data[:,column_names.index(f'{var}')]
            hf[f'{var}'].resize((hf[f'{var}'].shape[0] + array.shape[0]), axis = 0)
            hf[f'{var}'][-array.shape[0]:] = array.astype(column_dtype[f'{var}'])


#============================================================================
# Define column names and dtypes
#============================================================================

selection_types = {1 : "Sig",
                   2 : "Trash",
                   3 : "Bkg2Ele",
                   4 : "Bkg1Ele",
                   5 : "Bkg0Ele"}

column_dtype = {
'correctedScaledAverageMu': float,
'NvtxReco': float,
"invM" : float,
"et" : float,
"eta" : float,
"phi" : float,
"type" : int,
"isATLAS" : int,
####
####
'muo1_truthPdgId': int,
'muo1_truthType': int,
'muo1_truthOrigin': int,
# 'muo1_truth_eta': float,
# 'muo1_truth_phi': float,
# 'muo1_truth_m': float,
# 'muo1_truth_px': float,
# 'muo1_truth_py': float,
# 'muo1_truth_pz': float,
# 'muo1_truth_E': float,
'muo1_etcone20': float,
'muo1_etcone30': float,
'muo1_etcone40': float,
'muo1_ptcone20': float,
'muo1_ptcone30': float,
'muo1_ptcone40': float,
'muo1_ptvarcone20': float,
'muo1_ptvarcone30': float,
'muo1_ptvarcone40': float,
'muo1_pt': float,
'muo1_eta': float,
'muo1_phi': float,
# 'muo1_charge': int,
# 'muo1_innerSmallHits': int,
# 'muo1_innerLargeHits': int,
# 'muo1_middleSmallHits': int,
# 'muo1_middleLargeHits': int,
# 'muo1_outerSmallHits': int,
# 'muo1_outerLargeHits': int,
# 'muo1_extendedSmallHits': int,
# 'muo1_extendedLargeHits': int,
# 'muo1_cscEtaHits': int,
# 'muo1_cscUnspoiledEtaHits': int,
# 'muo1_innerSmallHoles': int,
# 'muo1_innerLargeHoles': int,
# 'muo1_middleSmallHoles': int,
# 'muo1_middleLargeHoles': int,
# 'muo1_outerSmallHoles': int,
# 'muo1_outerLargeHoles': int,
# 'muo1_extendedSmallHoles': int,
# 'muo1_extendedLargeHoles': int,
'muo1_author': int,
'muo1_allAuthors': int,
'muo1_muonType': int,
'muo1_numberOfPrecisionLayers': int,
'muo1_numberOfPrecisionHoleLayers': int,
'muo1_quality': int,
# 'muo1_energyLossType': int,
# 'muo1_spectrometerFieldIntegral': float,
# 'muo1_scatteringCurvatureSignificance': float,
# 'muo1_scatteringNeighbourSignificance': float,
# 'muo1_momentumBalanceSignificance': float,
# 'muo1_segmentDeltaEta': float,
# 'muo1_CaloLRLikelihood': float,
# 'muo1_EnergyLoss': float,
# 'muo1_CaloMuonIDTag': float,
# 'muo1_DFCommonGoodMuon': float,
# 'muo1_DFCommonMuonsPreselection': float,
'muo1_LHLoose': int,
'muo1_LHMedium': int,
'muo1_LHTight': int,
'muo1_trigger':int,
'muo1_priTrack_d0': float,
'muo1_priTrack_z0': float,
'muo1_priTrack_d0Sig': float,
'muo1_priTrack_z0Sig': float,
# 'muo1_priTrack_theta': float,
# 'muo1_priTrack_qOverP': float,
# 'muo1_priTrack_vx': float,
# 'muo1_priTrack_vy': float,
# 'muo1_priTrack_vz': float,
# 'muo1_priTrack_phi': float,
# 'muo1_priTrack_chiSquared': float,
# 'muo1_priTrack_numberDoF': float,
# 'muo1_priTrack_radiusOfFirstHit': float,
# 'muo1_priTrack_trackFitter': float,
# 'muo1_priTrack_particleHypothesis': float,
# 'muo1_priTrack_numberOfUsedHitsdEdx': float,
# 'muo1_priTrack_numberOfContribPixelLayers': float,
# 'muo1_priTrack_numberOfInnermostPixelLayerHits': float,
# 'muo1_priTrack_expectInnermostPixelLayerHit': float,
# 'muo1_priTrack_numberOfNextToInnermostPixelLayerHits': float,
# 'muo1_priTrack_expectNextToInnermostPixelLayerHit': float,
# 'muo1_priTrack_numberOfPixelHits': float,
# 'muo1_priTrack_numberOfGangedPixels': float,
# 'muo1_priTrack_numberOfGangedFlaggedFakes': float,
# 'muo1_priTrack_numberOfPixelSpoiltHits': float,
# 'muo1_priTrack_numberOfDBMHits': float,
# 'muo1_priTrack_numberOfSCTHits': float,
# 'muo1_priTrack_numberOfTRTHits': float,
# 'muo1_priTrack_numberOfOutliersOnTrack': float,
# 'muo1_priTrack_standardDeviationOfChi2OS': float,
# 'muo1_priTrack_pixeldEdx': float,
# 'muo1_IDTrack_d0': float,
# 'muo1_IDTrack_z0': float,
# 'muo1_IDTrack_d0Sig': float,
# 'muo1_IDTrack_z0Sig': float,
# 'muo1_IDTrack_theta': float,
# 'muo1_IDTrack_qOverP': float,
# 'muo1_IDTrack_vx': float,
# 'muo1_IDTrack_vy': float,
# 'muo1_IDTrack_vz': float,
# 'muo1_IDTrack_phi': float,
# 'muo1_IDTrack_chiSquared': float,
# 'muo1_IDTrack_numberDoF': float,
# 'muo1_IDTrack_radiusOfFirstHit': float,
# 'muo1_IDTrack_trackFitter': float,
# 'muo1_IDTrack_particleHypothesis': float,
# 'muo1_IDTrack_numberOfUsedHitsdEdx': float,
# 'muo1_ET_Core': float,
# 'muo1_ET_EMCore': float,
# 'muo1_ET_HECCore': float,
'muo1_ET_TileCore': float,
# 'muo1_FSR_CandidateEnergy': float,
# 'muo1_InnerDetectorPt': float,
'muo1_MuonSpectrometerPt': float,
# 'muo1_combinedTrackOutBoundsPrecisionHits': float,
# 'muo1_coreMuonEnergyCorrection': float,
'muo1_deltaphi_0': float,
'muo1_deltaphi_1': float,
'muo1_deltatheta_0': float,
'muo1_deltatheta_1': float,
'muo1_etconecoreConeEnergyCorrection': float,
# 'muo1_extendedClosePrecisionHits': int,
# 'muo1_extendedOutBoundsPrecisionHits': int,
# 'muo1_innerClosePrecisionHits': int,
# 'muo1_innerOutBoundsPrecisionHits': int,
# 'muo1_isEndcapGoodLayers': int,
# 'muo1_isSmallGoodSectors': int,
# 'muo1_middleClosePrecisionHits': int,
# 'muo1_middleOutBoundsPrecisionHits': int,
# 'muo1_numEnergyLossPerTrack': int,
# 'muo1_numberOfGoodPrecisionLayers': int,
# 'muo1_outerClosePrecisionHits': int,
# 'muo1_outerOutBoundsPrecisionHits': int,
'muo1_sigmadeltaphi_0': float,
'muo1_sigmadeltaphi_1': float,
'muo1_sigmadeltatheta_0': float,
'muo1_sigmadeltatheta_1': float,
# 'muo1_etconeCorrBitset': float,
# 'muo1_neflowisol20': float,
# 'muo1_neflowisol30': float,
# 'muo1_neflowisol40': float,
# 'muo1_neflowisolCorrBitset': float,
'muo1_neflowisolcoreConeEnergyCorrection': float,
# 'muo1_ptconeCorrBitset': float,
'muo1_ptconecoreTrackPtrCorrection': float,
# 'muo1_topoetconeCorrBitset': float,
'muo1_topoetconecoreConeEnergyCorrection': float,
# 'muo1_CT_EL_Type': float,
# 'muo1_CT_ET_Core': float,
# 'muo1_CT_ET_FSRCandidateEnergy': float,
# 'muo1_CT_ET_LRLikelihood': float,
# 'muo1_d0_staco': float,
# 'muo1_phi0_staco': float,
# 'muo1_qOverPErr_staco': float,
# 'muo1_qOverP_staco': float,
# 'muo1_theta_staco': float,
# 'muo1_z0_staco': float,
####
####
'muo2_truthPdgId': int,
'muo2_truthType': int,
'muo2_truthOrigin': int,
# 'muo2_truth_eta': float,
# 'muo2_truth_phi': float,
# 'muo2_truth_m': float,
# 'muo2_truth_px': float,
# 'muo2_truth_py': float,
# 'muo2_truth_pz': float,
# 'muo2_truth_E': float,
'muo2_etcone20': float,
'muo2_etcone30': float,
'muo2_etcone40': float,
'muo2_ptcone20': float,
'muo2_ptcone30': float,
'muo2_ptcone40': float,
'muo2_ptvarcone20': float,
'muo2_ptvarcone30': float,
'muo2_ptvarcone40': float,
'muo2_pt': float,
'muo2_eta': float,
'muo2_phi': float,
# 'muo2_charge': int,
# 'muo2_innerSmallHits': int,
# 'muo2_innerLargeHits': int,
# 'muo2_middleSmallHits': int,
# 'muo2_middleLargeHits': int,
# 'muo2_outerSmallHits': int,
# 'muo2_outerLargeHits': int,
# 'muo2_extendedSmallHits': int,
# 'muo2_extendedLargeHits': int,
# 'muo2_cscEtaHits': int,
# 'muo2_cscUnspoiledEtaHits': int,
# 'muo2_innerSmallHoles': int,
# 'muo2_innerLargeHoles': int,
# 'muo2_middleSmallHoles': int,
# 'muo2_middleLargeHoles': int,
# 'muo2_outerSmallHoles': int,
# 'muo2_outerLargeHoles': int,
# 'muo2_extendedSmallHoles': int,
# 'muo2_extendedLargeHoles': int,
'muo2_author': int,
'muo2_allAuthors': int,
'muo2_muonType': int,
'muo2_numberOfPrecisionLayers': int,
'muo2_numberOfPrecisionHoleLayers': int,
'muo2_quality': int,
# 'muo2_energyLossType': int,
# 'muo2_spectrometerFieldIntegral': float,
# 'muo2_scatteringCurvatureSignificance': float,
# 'muo2_scatteringNeighbourSignificance': float,
# 'muo2_momentumBalanceSignificance': float,
# 'muo2_segmentDeltaEta': float,
# 'muo2_CaloLRLikelihood': float,
# 'muo2_EnergyLoss': float,
# 'muo2_CaloMuonIDTag': float,
# 'muo2_DFCommonGoodMuon': float,
# 'muo2_DFCommonMuonsPreselection': float,
'muo2_LHLoose': int,
'muo2_LHMedium': int,
'muo2_LHTight': int,
'muo2_trigger':int,
'muo2_priTrack_d0': float,
'muo2_priTrack_z0': float,
'muo2_priTrack_d0Sig': float,
'muo2_priTrack_z0Sig': float,
# 'muo2_priTrack_theta': float,
# 'muo2_priTrack_qOverP': float,
# 'muo2_priTrack_vx': float,
# 'muo2_priTrack_vy': float,
# 'muo2_priTrack_vz': float,
# 'muo2_priTrack_phi': float,
# 'muo2_priTrack_chiSquared': float,
# 'muo2_priTrack_numberDoF': float,
# 'muo2_priTrack_radiusOfFirstHit': float,
# 'muo2_priTrack_trackFitter': float,
# 'muo2_priTrack_particleHypothesis': float,
# 'muo2_priTrack_numberOfUsedHitsdEdx': float,
# 'muo2_priTrack_numberOfContribPixelLayers': float,
# 'muo2_priTrack_numberOfInnermostPixelLayerHits': float,
# 'muo2_priTrack_expectInnermostPixelLayerHit': float,
# 'muo2_priTrack_numberOfNextToInnermostPixelLayerHits': float,
# 'muo2_priTrack_expectNextToInnermostPixelLayerHit': float,
# 'muo2_priTrack_numberOfPixelHits': float,
# 'muo2_priTrack_numberOfGangedPixels': float,
# 'muo2_priTrack_numberOfGangedFlaggedFakes': float,
# 'muo2_priTrack_numberOfPixelSpoiltHits': float,
# 'muo2_priTrack_numberOfDBMHits': float,
# 'muo2_priTrack_numberOfSCTHits': float,
# 'muo2_priTrack_numberOfTRTHits': float,
# 'muo2_priTrack_numberOfOutliersOnTrack': float,
# 'muo2_priTrack_standardDeviationOfChi2OS': float,
# 'muo2_priTrack_pixeldEdx': float,
# 'muo2_IDTrack_d0': float,
# 'muo2_IDTrack_z0': float,
# 'muo2_IDTrack_d0Sig': float,
# 'muo2_IDTrack_z0Sig': float,
# 'muo2_IDTrack_theta': float,
# 'muo2_IDTrack_qOverP': float,
# 'muo2_IDTrack_vx': float,
# 'muo2_IDTrack_vy': float,
# 'muo2_IDTrack_vz': float,
# 'muo2_IDTrack_phi': float,
# 'muo2_IDTrack_chiSquared': float,
# 'muo2_IDTrack_numberDoF': float,
# 'muo2_IDTrack_radiusOfFirstHit': float,
# 'muo2_IDTrack_trackFitter': float,
# 'muo2_IDTrack_particleHypothesis': float,
# 'muo2_IDTrack_numberOfUsedHitsdEdx': float,
# 'muo2_ET_Core': float,
# 'muo2_ET_EMCore': float,
# 'muo2_ET_HECCore': float,
'muo2_ET_TileCore': float,
# 'muo2_FSR_CandidateEnergy': float,
# 'muo2_InnerDetectorPt': float,
'muo2_MuonSpectrometerPt': float,
# 'muo2_combinedTrackOutBoundsPrecisionHits': float,
# 'muo2_coreMuonEnergyCorrection': float,
'muo2_deltaphi_0': float,
'muo2_deltaphi_1': float,
'muo2_deltatheta_0': float,
'muo2_deltatheta_1': float,
'muo2_etconecoreConeEnergyCorrection': float,
# 'muo2_extendedClosePrecisionHits': int,
# 'muo2_extendedOutBoundsPrecisionHits': int,
# 'muo2_innerClosePrecisionHits': int,
# 'muo2_innerOutBoundsPrecisionHits': int,
# 'muo2_isEndcapGoodLayers': int,
# 'muo2_isSmallGoodSectors': int,
# 'muo2_middleClosePrecisionHits': int,
# 'muo2_middleOutBoundsPrecisionHits': int,
# 'muo2_numEnergyLossPerTrack': int,
# 'muo2_numberOfGoodPrecisionLayers': int,
# 'muo2_outerClosePrecisionHits': int,
# 'muo2_outerOutBoundsPrecisionHits': int,
'muo2_sigmadeltaphi_0': float,
'muo2_sigmadeltaphi_1': float,
'muo2_sigmadeltatheta_0': float,
'muo2_sigmadeltatheta_1': float,
# 'muo2_etconeCorrBitset': float,
# 'muo2_neflowisol20': float,
# 'muo2_neflowisol30': float,
# 'muo2_neflowisol40': float,
# 'muo2_neflowisolCorrBitset': float,
'muo2_neflowisolcoreConeEnergyCorrection': float,
# 'muo2_ptconeCorrBitset': float,
'muo2_ptconecoreTrackPtrCorrection': float,
# 'muo2_topoetconeCorrBitset': float,
'muo2_topoetconecoreConeEnergyCorrection': float,
# 'muo2_CT_EL_Type': float,
# 'muo2_CT_ET_Core': float,
# 'muo2_CT_ET_FSRCandidateEnergy': float,
# 'muo2_CT_ET_LRLikelihood': float,
# 'muo2_d0_staco': float,
# 'muo2_phi0_staco': float,
# 'muo2_qOverPErr_staco': float,
# 'muo2_qOverP_staco': float,
# 'muo2_theta_staco': float,
# 'muo2_z0_staco': float
}

column_names = list(column_dtype.keys())



#============================================================================
# Main
#============================================================================

# create file name and check if the file already exists
filename = '{:s}.h5'.format(args.tag)
if os.path.isfile(args.outdir + filename):
    log.error(f"Output file already exists - please remove yourself. Output: {args.outdir + filename}")
    quit()

# Make a pool of processes (this must come after the functions needed to run over since it apparently imports __main__ here)
pool = multiprocessing.Pool(processes=args.max_processes)

for path in args.paths:
    # Count which file we have made it to
    counter += 1

    # Read hdf5 data to get number of events
    hf_read = h5py.File(path, "r")

    print(hf_read.keys())

    N = hf_read['NvtxReco'].shape[0]

    print("N = ", N)

    # Split indices into equally-sized batches
    index_edges = list(map(int, np.linspace(0, N, args.max_processes + 1, endpoint=True)))
    index_ranges = zip(index_edges[:-1], index_edges[1:])

    results = pool.map(MakeFiles, [(i, counter, path, start, stop) for i, (start, stop) in enumerate(index_ranges)])
    results_np = np.array(results)

    # Concatenate resulting data from the multiple converters
    data = np.concatenate(results_np)

    # Print the total event count in the file
    log.info("Data shape: {}".format(data.shape))

    # Save output to a file
    if counter == 0:
        saveToFile(filename, data, column_names, column_dtype)
    else:
        appendToFile(filename, data, column_names, column_dtype)


sec = timedelta(seconds=time() - t_start)
log.info(f"Extraction finished. Time spent: {str(sec)}")
