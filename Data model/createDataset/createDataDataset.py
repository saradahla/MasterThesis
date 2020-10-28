#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
nohup python -u createDataDataset.py --tag 010920 ../hdf5Prod/output/root2hdf5/Data010920/Data010920_0000.h5 ../hdf5Prod/output/root2hdf5/Data010920/Data010920_0001.h5 ../hdf5Prod/output/root2hdf5/Data010920/Data010920_0002.h5 ../hdf5Prod/output/root2hdf5/Data010920/Data010920_0003.h5 2>&1 &> output/logDataDataset.txt & disown
nohup python -u createDataDataset.py --tag 020920_MC ../hdf5Prod/output/root2hdf5/Zmm310820/Zmm310820_0000.h5 ../hdf5Prod/output/root2hdf5/Zmm310820/Zmm310820_0001.h5 ../hdf5Prod/output/root2hdf5/Zmm310820/Zmm310820_0002.h5 ../hdf5Prod/output/root2hdf5/Zmm310820/Zmm310820_0003.h5 ../hdf5Prod/output/root2hdf5/bb310820/bb310820_0000.h5 ../hdf5Prod/output/root2hdf5/bb310820/bb310820_0001.h5 ../hdf5Prod/output/root2hdf5/bb310820/bb310820_0002.h5 ../hdf5Prod/output/root2hdf5/bb310820/bb310820_0003.h5 ../hdf5Prod/output/root2hdf5/Wmn310820/Wmn310820_0000.h5 ../hdf5Prod/output/root2hdf5/Wmn310820/Wmn310820_0001.h5 ../hdf5Prod/output/root2hdf5/Wmn310820/Wmn310820_0002.h5 ../hdf5Prod/output/root2hdf5/Wmn310820/Wmn310820_0003.h5 2>&1 &> output/logDataDataset.txt & disown
"""

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


# Logging style and level
log.basicConfig(format='[%(levelname)s] %(message)s', level=log.INFO)
log.info("Packages imported")

# Start "timer"
t_start = time()

# Command line options
parser = argparse.ArgumentParser(description="Extract data from HDF5 files into flat HDF5 files for training.")
parser.add_argument('--tag', action='store', type=str, required=True,
                    help='Tag the data category (Zee, Wev, etc.).')
parser.add_argument('--outdir', action='store', default="output/MuoSingleHdf5/", type=str,
                    help='Output directory.')
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

#%##########################
#         Functions
#%##########################

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

def invMass(hf, event, tag, probe):
    # Calculate mZee using: https://github.com/scikit-hep/scikit-hep/blob/master/skhep/math/vectors.py?fbclid=IwAR3C0qnNlxKx-RhGjwo1c1FeZEpWbYqFrNmEqMv5iE-ibyPw_xEqmDYgRpc
    # Get variables
    p1 = hf[ 'muo_pt' ][ event ][ tag ]
    eta1 = hf[ 'muo_eta' ][ event ][ tag ]
    phi1 = hf[ 'muo_phi' ][ event ][ tag ]
    p2 = hf[ 'muo_pt' ][ event ][ probe ]
    eta2 = hf[ 'muo_eta' ][ event ][ probe ]
    phi2 = hf[ 'muo_phi' ][ event ][ probe ]

    # make four vector
    vecFour1 = vectors.LorentzVector()
    vecFour2 = vectors.LorentzVector()

    vecFour1.setptetaphim(p1/1000, eta1, phi1, 0.105) #Units in GeV for pt and mass
    vecFour2.setptetaphim(p2/1000, eta2, phi2, 0.105)

    # calculate invariant mass
    vecFour = vecFour1+vecFour2
    invM = vecFour.mass
    pt = vecFour.pt
    eta = vecFour.eta
    phi = vecFour.phi()

    return invM, pt, eta, phi

def getTagsAndProbes(hf, event, nMuo):
    MuoTag = []
    MuoProbe = []
    for muo in np.arange(nMuo):
        if (hf[ "muo_trigger" ][ event ][ muo ] & hf[ "muo_LHTight" ][ event ][ muo ] & (hf[ "muo_pt" ][ event ][ muo ] > 26000)):
            MuoTag.append(muo)
        else:
            MuoProbe.append(muo)
    return MuoTag, MuoProbe


def signalSelection(hf, event, tag, probe, invM, cutZ = 5):
    Z_mass = 91.187 #GeV

    muo0_charge = hf["muo_charge"][ event ][ tag ]
    muo1_charge = hf["muo_charge"][ event ][ probe ]

    if ((muo0_charge*muo1_charge) < 0):
        if ((Z_mass - cutZ) < invM) & ((Z_mass + cutZ) > invM): # Invariant mass in range [85, 95] GeV
            muoType = 1

        else: #Opposite sign but not in Z mass range
            muoType = 2
    else: # sameSign
        muoType = 0
    return muoType



def addMuonVariables(hf, event, data_temp, muo, path):
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

    # data_temp[ 0, column_names.index( f'muo_truthPdgId' ) ] = hf[ 'muo_truthPdgId' ][ event ][ muo ]
    # data_temp[ 0, column_names.index( f'muo_truthType' ) ] = hf[ 'muo_truthType' ][ event ][ muo ]
    # data_temp[ 0, column_names.index( f'muo_truthOrigin' ) ] = hf[ 'muo_truthOrigin' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo_etcone20' ) ] = hf[ 'muo_etcone20' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_etcone30' ) ] = hf[ 'muo_etcone30' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_etcone40' ) ] = hf[ 'muo_etcone40' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo_ptcone20' ) ] = hf[ 'muo_ptcone20' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_ptcone30' ) ] = hf[ 'muo_ptcone30' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_ptcone40' ) ] = hf[ 'muo_ptcone40' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo_ptvarcone20' ) ] = hf[ 'muo_ptvarcone20' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_ptvarcone30' ) ] = hf[ 'muo_ptvarcone30' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_ptvarcone40' ) ] = hf[ 'muo_ptvarcone40' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo_pt' ) ] = hf[ 'muo_pt' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_eta' ) ] = hf[ 'muo_eta' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_phi' ) ] = hf[ 'muo_phi' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo_muonType' ) ] = hf[ 'muo_muonType' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo_numberOfPrecisionLayers' ) ] = hf[ 'muo_numberOfPrecisionLayers' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_numberOfPrecisionHoleLayers' ) ] = hf[ 'muo_numberOfPrecisionHoleLayers' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo_quality' ) ] = hf[ 'muo_quality' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo_LHLoose' ) ] = hf[ 'muo_LHLoose' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_LHMedium' ) ] = hf[ 'muo_LHMedium' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_LHTight' ) ] = hf[ 'muo_LHTight' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo_trigger' ) ] = hf[ 'muo_trigger' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_charge' ) ] = hf[ 'muo_charge' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo_innerSmallHits' ) ] = hf[ 'muo_innerSmallHits' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_innerLargeHits' ) ] = hf[ 'muo_innerLargeHits' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_middleSmallHits' ) ] = hf[ 'muo_middleSmallHits' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_middleLargeHits' ) ] = hf[ 'muo_middleLargeHits' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_outerSmallHits' ) ] = hf[ 'muo_outerSmallHits' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_outerLargeHits' ) ] = hf[ 'muo_outerLargeHits' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo_CaloLRLikelihood' ) ] = hf[ 'muo_CaloLRLikelihood' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo_priTrack_d0' ) ] = hf[ 'muo_priTrack_d0' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_priTrack_z0' ) ] = hf[ 'muo_priTrack_z0' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_priTrack_d0Sig' ) ] = hf[ 'muo_priTrack_d0Sig' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_priTrack_z0Sig' ) ] = hf[ 'muo_priTrack_z0Sig' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo_priTrack_chiSquared' ) ] = hf[ 'muo_priTrack_chiSquared' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_priTrack_numberDoF' ) ] = hf[ 'muo_priTrack_numberDoF' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo_priTrack_numberOfInnermostPixelLayerHits' ) ] = hf[ 'muo_priTrack_numberOfInnermostPixelLayerHits' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_priTrack_expectInnermostPixelLayerHit' ) ] = hf[ 'muo_priTrack_expectInnermostPixelLayerHit' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_priTrack_numberOfPixelHits' ) ] = hf[ 'muo_priTrack_numberOfPixelHits' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_priTrack_numberOfSCTHits' ) ] = hf[ 'muo_priTrack_numberOfSCTHits' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_priTrack_numberOfTRTHits' ) ] = hf[ 'muo_priTrack_numberOfTRTHits' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo_IDTrack_chiSquared' ) ] = hf[ 'muo_IDTrack_chiSquared' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_IDTrack_numberDoF' ) ] = hf[ 'muo_IDTrack_numberDoF' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo_neflowisol20' ) ] = hf[ 'muo_neflowisol20' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_neflowisol30' ) ] = hf[ 'muo_neflowisol30' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_neflowisol40' ) ] = hf[ 'muo_neflowisol40' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo_ET_TileCore' ) ] = hf[ 'muo_ET_TileCore' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_MuonSpectrometerPt' ) ] = hf[ 'muo_MuonSpectrometerPt' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo_deltaphi_0' ) ] = hf[ 'muo_deltaphi_0' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_deltaphi_1' ) ] = hf[ 'muo_deltaphi_1' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_deltatheta_0' ) ] = hf[ 'muo_deltatheta_0' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_deltatheta_1' ) ] = hf[ 'muo_deltatheta_1' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo_sigmadeltaphi_0' ) ] = hf[ 'muo_sigmadeltaphi_0' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_sigmadeltaphi_1' ) ] = hf[ 'muo_sigmadeltaphi_1' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_sigmadeltatheta_0' ) ] = hf[ 'muo_sigmadeltatheta_0' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_sigmadeltatheta_1' ) ] = hf[ 'muo_sigmadeltatheta_1' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo_etconecoreConeEnergyCorrection' ) ] = hf[ 'muo_etconecoreConeEnergyCorrection' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_neflowisolcoreConeEnergyCorrection' ) ] = hf[ 'muo_neflowisolcoreConeEnergyCorrection' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_ptconecoreTrackPtrCorrection' ) ] = hf[ 'muo_ptconecoreTrackPtrCorrection' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_topoetconecoreConeEnergyCorrection' ) ] = hf[ 'muo_topoetconecoreConeEnergyCorrection' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo_InnerDetectorPt' ) ] = hf[ 'muo_InnerDetectorPt' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_DFCommonMuonsPreselection' ) ] = hf[ 'muo_DFCommonMuonsPreselection' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_author' ) ] = hf[ 'muo_author' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo_scatteringCurvatureSignificance' ) ] = hf[ 'muo_scatteringCurvatureSignificance' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_scatteringNeighbourSignificance' ) ] = hf[ 'muo_scatteringNeighbourSignificance' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_momentumBalanceSignificance' ) ] = hf[ 'muo_momentumBalanceSignificance' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_EnergyLoss' ) ] = hf[ 'muo_EnergyLoss' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo_energyLossType' ) ] = hf[ 'muo_energyLossType' ][ event ][ muo ]




### Importing data
#
# hf = h5ToDf("/Users/sda/hep/work/hdf5Prod/output/root2hdf5/Data310820/Data310820_0000.h5")
# hf2 = h5ToDf("/Users/sda/hep/work/hdf5Prod/output/root2hdf5/Data310820/Data310820_0001.h5")
#
# hf = hf.append(hf2, ignore_index = True)

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

        nMuo = len(hf["muo_pt"][event])
        MuoTag, MuoProbe = getTagsAndProbes(hf, event, nMuo)

        if (nMuo >= 2) & (len(MuoTag) > 0):
            # Loop over tags
            for iTag, tag in enumerate(MuoTag):
                muoProbes = MuoProbe.copy()

                for muo in (MuoTag[(iTag+1):]): #Other tag muons are also probes
                    muoProbes.append(muo)
                # Loop over probes:
                for iProbe, probe in enumerate(muoProbes):
                    invM, pt, eta, phi = invMass(hf, event, tag, probe)

                    muoType = signalSelection(hf, event, tag, probe, invM)

                    data_temp = np.zeros((1,len(column_names)))

                    # Add event variables to array
                    data_temp[ 0, column_names.index( 'NvtxReco' ) ] = np.int(hf[ 'NvtxReco' ][ event ])
                    data_temp[ 0, column_names.index( 'correctedScaledAverageMu' ) ] = hf[ 'correctedScaledAverageMu' ][ event ]
                    data_temp[ 0, column_names.index( 'Type' ) ] = muoType
                    data_temp[ 0, column_names.index( 'invM' ) ] = invM
                    # Add muon variables to array
                    invariantMass.append(invM)
                    types.append(muoType)
                    pt_arr.append(hf[ 'muo_pt' ][ event ][ probe ])

                    addMuonVariables(hf, event, data_temp, probe, path) #only adding the probe
                    data = np.append(data, data_temp, axis=0)

        #else:
        #    noTag += 1
    return data, invariantMass, types, pt_arr


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





column_dtype = {
'correctedScaledAverageMu': float,
'NvtxReco': float,
'Type': int,
'invM': float,
####
####
'muo_truthPdgId': int,
'muo_truthType': int,
'muo_truthOrigin': int,
# 'muo_truth_eta': float,
# 'muo_truth_phi': float,
# 'muo_truth_m': float,
# 'muo_truth_px': float,
# 'muo_truth_py': float,
# 'muo_truth_pz': float,
# 'muo_truth_E': float,
'muo_etcone20': float,
'muo_etcone30': float,
'muo_etcone40': float,
'muo_ptcone20': float,
'muo_ptcone30': float,
'muo_ptcone40': float,
'muo_ptvarcone20': float,
'muo_ptvarcone30': float,
'muo_ptvarcone40': float,
'muo_pt': float,
'muo_eta': float,
'muo_phi': float,
'muo_charge': int,
'muo_innerSmallHits': int,
'muo_innerLargeHits': int,
'muo_middleSmallHits': int,
'muo_middleLargeHits': int,
'muo_outerSmallHits': int,
'muo_outerLargeHits': int,
# 'muo_extendedSmallHits': int,
# 'muo_extendedLargeHits': int,
# 'muo_cscEtaHits': int,
# 'muo_cscUnspoiledEtaHits': int,
# 'muo_innerSmallHoles': int,
# 'muo_innerLargeHoles': int,
# 'muo_middleSmallHoles': int,
# 'muo_middleLargeHoles': int,
# 'muo_outerSmallHoles': int,
# 'muo_outerLargeHoles': int,
# 'muo_extendedSmallHoles': int,
# 'muo_extendedLargeHoles': int,
'muo_author': int,
# 'muo_allAuthors': int,
'muo_muonType': int,
'muo_numberOfPrecisionLayers': int,
'muo_numberOfPrecisionHoleLayers': int,
'muo_quality': int,
'muo_energyLossType': int,
# 'muo_spectrometerFieldIntegral': float,
'muo_scatteringCurvatureSignificance': float,
'muo_scatteringNeighbourSignificance': float,
'muo_momentumBalanceSignificance': float,
# 'muo_segmentDeltaEta': float,
'muo_CaloLRLikelihood': float,
'muo_EnergyLoss': float,
# 'muo_CaloMuonIDTag': float,
# 'muo_DFCommonGoodMuon': float,
'muo_DFCommonMuonsPreselection': float,
'muo_LHLoose': int,
'muo_LHMedium': int,
'muo_LHTight': int,
'muo_trigger': int,
'muo_priTrack_d0': float,
'muo_priTrack_z0': float,
'muo_priTrack_d0Sig': float,
'muo_priTrack_z0Sig': float,
# 'muo_priTrack_theta': float,
# 'muo_priTrack_qOverP': float,
# 'muo_priTrack_vx': float,
# 'muo_priTrack_vy': float,
# 'muo_priTrack_vz': float,
# 'muo_priTrack_phi': float,
'muo_priTrack_chiSquared': float,
'muo_priTrack_numberDoF': float,
# 'muo_priTrack_radiusOfFirstHit': float,
# 'muo_priTrack_trackFitter': float,
# 'muo_priTrack_particleHypothesis': float,
# 'muo_priTrack_numberOfUsedHitsdEdx': float,
# 'muo_priTrack_numberOfContribPixelLayers': float,
'muo_priTrack_numberOfInnermostPixelLayerHits': float,
'muo_priTrack_expectInnermostPixelLayerHit': float,
# 'muo_priTrack_numberOfNextToInnermostPixelLayerHits': float,
# 'muo_priTrack_expectNextToInnermostPixelLayerHit': float,
'muo_priTrack_numberOfPixelHits': float,
# 'muo_priTrack_numberOfGangedPixels': float,
# 'muo_priTrack_numberOfGangedFlaggedFakes': float,
# 'muo_priTrack_numberOfPixelSpoiltHits': float,
# 'muo_priTrack_numberOfDBMHits': float,
'muo_priTrack_numberOfSCTHits': float,
'muo_priTrack_numberOfTRTHits': float,
# 'muo_priTrack_numberOfOutliersOnTrack': float,
# 'muo_priTrack_standardDeviationOfChi2OS': float,
# 'muo_priTrack_pixeldEdx': float,
# 'muo_IDTrack_d0': float,
# 'muo_IDTrack_z0': float,
# 'muo_IDTrack_d0Sig': float,
# 'muo_IDTrack_z0Sig': float,
# 'muo_IDTrack_theta': float,
# 'muo_IDTrack_qOverP': float,
# 'muo_IDTrack_vx': float,
# 'muo_IDTrack_vy': float,
# 'muo_IDTrack_vz': float,
# 'muo_IDTrack_phi': float,
'muo_IDTrack_chiSquared': float,
'muo_IDTrack_numberDoF': float,
# 'muo_IDTrack_radiusOfFirstHit': float,
# 'muo_IDTrack_trackFitter': float,
# 'muo_IDTrack_particleHypothesis': float,
# 'muo_IDTrack_numberOfUsedHitsdEdx': float,
# 'muo_ET_Core': float,
# 'muo_ET_EMCore': float,
# 'muo_ET_HECCore': float,
'muo_ET_TileCore': float,
# 'muo_FSR_CandidateEnergy': float,
'muo_InnerDetectorPt': float,
'muo_MuonSpectrometerPt': float,
# 'muo_combinedTrackOutBoundsPrecisionHits': float,
# 'muo_coreMuonEnergyCorrection': float,
'muo_deltaphi_0': float,
'muo_deltaphi_1': float,
'muo_deltatheta_0': float,
'muo_deltatheta_1': float,
'muo_etconecoreConeEnergyCorrection': float,
# 'muo_extendedClosePrecisionHits': int,
# 'muo_extendedOutBoundsPrecisionHits': int,
# 'muo_innerClosePrecisionHits': int,
# 'muo_innerOutBoundsPrecisionHits': int,
# 'muo_isEndcapGoodLayers': int,
# 'muo_isSmallGoodSectors': int,
# 'muo_middleClosePrecisionHits': int,
# 'muo_middleOutBoundsPrecisionHits': int,
# 'muo_numEnergyLossPerTrack': int,
# 'muo_numberOfGoodPrecisionLayers': int,
# 'muo_outerClosePrecisionHits': int,
# 'muo_outerOutBoundsPrecisionHits': int,
'muo_sigmadeltaphi_0': float,
'muo_sigmadeltaphi_1': float,
'muo_sigmadeltatheta_0': float,
'muo_sigmadeltatheta_1': float,
# 'muo_etconeCorrBitset': float,
'muo_neflowisol20': float,
'muo_neflowisol30': float,
'muo_neflowisol40': float,
# 'muo_neflowisolCorrBitset': float,
'muo_neflowisolcoreConeEnergyCorrection': float,
# 'muo_ptconeCorrBitset': float,
'muo_ptconecoreTrackPtrCorrection': float,
# 'muo_topoetconeCorrBitset': float,
'muo_topoetconecoreConeEnergyCorrection': float,
# 'muo_CT_EL_Type': float,
# 'muo_CT_ET_Core': float,
# 'muo_CT_ET_FSRCandidateEnergy': float,
# 'muo_CT_ET_LRLikelihood': float,
# 'muo_d0_staco': float,
# 'muo_phi0_staco': float,
# 'muo_qOverPErr_staco': float,
# 'muo_qOverP_staco': float,
# 'muo_theta_staco': float,
# 'muo_z0_staco': float,

}

column_names = list(column_dtype.keys())


#============================================================================
# Main
#============================================================================
# Total counters for signal selection diagram
invariantMass = []
types = []
pt_arr = []

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
    data = np.concatenate(results_np[:,0])

    # Concatenate data and add to total
    invariantMass = np.concatenate(results_np[:,1], axis=0)
    types = np.concatenate(results_np[:,2], axis=0)
    pt_arr = np.concatenate(results_np[:,3], axis=0)
    #histNtype_total = histNtype_total + np.sum(histNtype, axis=0)

    # Concatenate data and add to total
    # histTrigPass = np.concatenate(results_np[:,2], axis=0)
    # histTrigPass_total = histTrigPass_total + np.sum(histTrigPass, axis=0)

    # Print the total event count in the file
    log.info("Data shape: {}".format(data.shape))

    # Save output to a file
    if counter == 0:
        saveToFile(filename, data, column_names, column_dtype)
    else:
        appendToFile(filename, data, column_names, column_dtype)

#### PLOTS #####

#Variables
invariantMass = np.array(invariantMass)
types = np.array(types)
pt_arr = np.array(pt_arr)/1000


# Invariant mass
fig, ax = plt.subplots(figsize=(5,5))
ax.hist(invariantMass, bins = 100, range = (0,120), color = 'b', histtype = "step")
ax.set(xlabel = r"$m_{\mu\mu}$", ylabel = "Frequency")
fig.tight_layout()
fig.savefig(args.outdir + "invariantMass_all" + ".pdf")

fig, ax = plt.subplots(figsize=(5,5))
ax.hist(invariantMass[types == 0], bins = 100, range = (0,120), color = 'b', histtype = "step", label = "0: Background")
ax.hist(invariantMass[types == 1], bins = 100, range = (0,120), color = 'r', histtype = "step", label = "1: Signal")
ax.hist(invariantMass[types == 2], bins = 100, range = (0,120), color = 'g', histtype = "step",  label = "2: Trash")
ax.set(xlabel = r"$m_{\mu\mu}$", ylabel = "Frequency")
ax.legend()
fig.tight_layout()
fig.savefig(args.outdir + "invariantMass_all_separate" + ".pdf")

fig, ax = plt.subplots(figsize=(5,5))
ax.hist(invariantMass[types == 0], bins = 100, range = (0,120), color = 'b', histtype = "step", label = "0: Background")
ax.hist(invariantMass[types == 1], bins = 100, range = (0,120), color = 'r', histtype = "step", label = "1: Signal")
ax.set(xlabel = r"$m_{\mu\mu}$", ylabel = "Frequency")
ax.legend()
fig.tight_layout()
fig.savefig(args.outdir + "invariantMass_sigbkg" + ".pdf")

# Pt cut on invariant mass (is not being done, just to make the histogram!)
pt_min = 10
fig, ax = plt.subplots(2,2, figsize = (10,10))
ax = ax.flatten()
ax[0].set(title = r"$m_{\mu\mu}$" + r"for p$_T^{Probe} < $" + f"{pt_min} GeV", xlabel = r"$m_{\mu\mu}$", ylabel = "Frequency")
ax[0].hist(invariantMass[types != 2][pt_arr[types != 2] < pt_min], bins = 120, range = (0, 120), histtype = "step", color = "b");
ax[0].axvline(40, color = 'k', linestyle = 'dashed')

ax[1].set(title = r"$m_{\mu\mu}$" + r"for p$_T^{Probe} > $" + f"{pt_min} GeV", xlabel = r"$m_{\mu\mu}$", ylabel = "Frequency")
ax[1].hist(invariantMass[types != 2][pt_arr[types != 2] > pt_min], bins = 120, range = (0, 120), histtype = "step", color = "b");
ax[1].axvline(40, color = 'k', linestyle = 'dashed')

pt_min = 15
ax[2].set(title = r"$m_{\mu\mu}$" + r"for p$_T^{Probe} < $" + f"{pt_min} GeV", xlabel = r"$m_{\mu\mu}$", ylabel = "Frequency")
ax[2].hist(invariantMass[types != 2][pt_arr[types != 2] < pt_min], bins = 120, range = (0, 120), histtype = "step", color = "b");
ax[2].axvline(40, color = 'k', linestyle = 'dashed')

ax[3].set(title = r"$m_{\mu\mu}$" + r"for p$_T^{Probe} > $" + f"{pt_min} GeV", xlabel = r"$m_{\mu\mu}$", ylabel = "Frequency")
ax[3].hist(invariantMass[types != 2][pt_arr[types != 2] > pt_min], bins = 120, range = (0, 120), histtype = "step", color = "b");
ax[3].axvline(40, color = 'k', linestyle = 'dashed')

fig.tight_layout()
fig.savefig(args.outdir + "ptCutMmm" + ".pdf")


### Other cuts (also not being performed, just plotted)
fig, ax = plt.subplots(2,2, figsize = (10,10))
ax = ax.flatten()
ax[0].set(title = "No cut", xlabel = "pt", ylabel = "Frequency")
ax[0].hist(pt_arr[types == 0], bins = 100, range = (0, 80), color = "b", histtype = "step", label = "Background");
ax[0].hist(pt_arr[types == 1], bins = 100, range = (0, 80), color = "r", histtype = "step", label = f"Signal, size {np.round((len(pt_arr[types == 1])/len(pt_arr[types == 0])),2)} of background");
ax[0].legend()

ax[1].set(title = "Cut p$_T > 5$ GeV", xlabel = "pt", ylabel = "Frequency")
ax[1].hist(pt_arr[types == 0][pt_arr[types == 0] > 5], bins = 100, range = (0, 80), color = "b", histtype = "step", label = "Background");
ax[1].hist(pt_arr[types == 1][pt_arr[types == 1] > 5], bins = 100, range = (0, 80), color = "r", histtype = "step", label = f"Signal, size {np.round((len(pt_arr[types == 1][pt_arr[types == 1] > 5])/len(pt_arr[types == 0][pt_arr[types == 0] > 5])),2)} of background");
ax[1].legend()

ax[2].set(title = r"Cut $m_{\mu\mu}$ > 40 GeV", xlabel = "pt", ylabel = "Frequency")
ax[2].hist(pt_arr[types == 0][invariantMass[types == 0] > 40], bins = 100, range = (0, 80), color = "b", histtype = "step", label = "Background");
ax[2].hist(pt_arr[types == 1][invariantMass[types == 1] > 40], bins = 100, range = (0, 80), color = "r", histtype = "step", label = f"Signal, size {np.round((len(pt_arr[types == 1][invariantMass[types == 1] > 40])/len(pt_arr[types == 0][invariantMass[types == 0] > 40])),2)} of background");
ax[2].legend()

ax[3].set(title = r"Cut p$_T > 5$ GeV and $m_{\mu\mu}$ > 40 GeV", xlabel = "pt", ylabel = "Frequency")
cut_bkg = (invariantMass > 40) & (pt_arr > 5) & (type == 0)
cut_sig = (invariantMass > 40) & (pt_arr > 5) & (type == 1)
ax[3].hist(pt_arr[cut_bkg], bins = 100, range = (0, 80), color = "b", histtype = "step", label = "Background");
ax[3].hist(pt_arr[cut_sig], bins = 100, range = (0, 80), color = "r", histtype = "step", label = f"Signal");#, size {np.round((len(pt_arr[cut_sig])/len(pt_arr[cut_bkg])),2)} of background");
ax[3].legend()
# ax[4].set(title = r"Cut background is non-isolated", xlabel = "pt", ylabel = "Frequency")
# ax[4].hist(pt[type == 0][ISO_prime < 0.5], bins = 100, range = (0, 80), color = "b", histtype = "step", label = "Background");
# ax[4].hist(pt[type == 1], bins = 100, range = (0, 80), color = "r", histtype = "step", label = f"Signal, size {np.round((len(pt[type == 1])/len(pt[type == 0][ISO_prime < 0.5])),2)} of background");
# ax[4].legend()
#
# ax[5].set(title = r"Cut p$_T > 5$ GeV and $m_{\mu\mu}$ > 40 GeV and non-isolated background", xlabel = "pt", ylabel = "Frequency")
# ax[5].hist(pt[type == 0][pt > 5][invM > 40][ISO_prime < 0.5], bins = 100, range = (0, 80), color = "b", histtype = "step", label = "Background");
# ax[5].hist(pt[type == 1][pt > 5][invM > 40], bins = 100, range = (0, 80), color = "r", histtype = "step", label = f"Signal, size {np.round((len(pt[type == 1][pt > 5][invM > 40])/len(pt[type == 0][pt > 5][invM > 40][ISO_prime < 0.5])),2)} of background");
# ax[5].legend()

fig.tight_layout()
fig.savefig(args.outdir + "ptZoom" + ".pdf")



sec = timedelta(seconds=time() - t_start)
log.info(f"Extraction finished. Time spent: {str(sec)}")
