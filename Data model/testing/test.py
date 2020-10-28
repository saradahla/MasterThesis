#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
nohup python -u createDataDataset.py --tag 310820
../../hdf5Prod/output/root2hdf5/Data310820/Data310820_0000.h5
../../hdf5Prod/output/root2hdf5/Data310820/Data310820_0001.h5
../../hdf5Prod/output/root2hdf5/Data310820/Data310820_0002.h5 
../../hdf5Prod/output/root2hdf5/Data310820/Data310820_0003.h5
2>&1 &> output/logDataDataset.txt & disown
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
    p1 = hf['muo_pt'][event][tag]
    eta1 = hf['muo_eta'][event][tag]
    phi1 = hf['muo_phi'][event][tag]
    p2 = hf['muo_pt'][event][probe]
    eta2 = hf['muo_eta'][event][probe]
    phi2 = hf['muo_phi'][event][probe]

    # make four vector
    vecFour1 = vectors.LorentzVector()
    vecFour2 = vectors.LorentzVector()

    vecFour1.setptetaphim(p1/1000,eta1,phi1,0.105) #Units in GeV for pt and mass
    vecFour2.setptetaphim(p2/1000,eta2,phi2,0.105)

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


def signalSelection(hf, event, tag, probe, invM):
    Z_mass = 90 #GeV

    muo0_charge = hf["muo_charge"][ event ][ tag ]
    muo1_charge = hf["muo_charge"][ event ][ probe ]

    if (muo0_charge*muo1_charge) < 0:
        if ((Z_mass - 7) < invM) & ((Z_mass + 7) > invM): # Invariant mass in range [85, 95] GeV
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




### Importing data

hf = h5ToDf("/Users/sda/hep/work/hdf5Prod/output/root2hdf5/Data310820/Data310820_0000.h5")
hf2 = h5ToDf("/Users/sda/hep/work/hdf5Prod/output/root2hdf5/Data310820/Data310820_0001.h5")

hf = hf.append(hf2, ignore_index = True)





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
                    data_temp[ 0, column_names.index( 'NvtxReco' ) ] = np.int(hf['NvtxReco'][event])
                    data_temp[ 0, column_names.index( 'Type' ) ] = muoType
                    data_temp[ 0, column_names.index( 'correctedScaledAverageMu' ) ] = hf[ 'correctedScaledAverageMu' ][ event ]
                    # Add muon variables to array

                    addMuonVariables(hf, event, data_temp, probe, path) #only adding the probe
                    data = np.append(data, data_temp, axis=0)

        #else:
        #    noTag += 1
    return data


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
####
####
# 'muo_truthPdgId': int,
# 'muo_truthType': int,
# 'muo_truthOrigin': int,
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
# 'muo_charge': int,
# 'muo_innerSmallHits': int,
# 'muo_innerLargeHits': int,
# 'muo_middleSmallHits': int,
# 'muo_middleLargeHits': int,
# 'muo_outerSmallHits': int,
# 'muo_outerLargeHits': int,
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
# 'muo_author': int,
# 'muo_allAuthors': int,
'muo_muonType': int,
'muo_numberOfPrecisionLayers': int,
'muo_numberOfPrecisionHoleLayers': int,
'muo_quality': int,
# 'muo_energyLossType': int,
# 'muo_spectrometerFieldIntegral': float,
# 'muo_scatteringCurvatureSignificance': float,
# 'muo_scatteringNeighbourSignificance': float,
# 'muo_momentumBalanceSignificance': float,
# 'muo_segmentDeltaEta': float,
# 'muo_CaloLRLikelihood': float,
# 'muo_EnergyLoss': float,
# 'muo_CaloMuonIDTag': float,
# 'muo_DFCommonGoodMuon': float,
# 'muo_DFCommonMuonsPreselection': float,
'muo_LHLoose': int,
'muo_LHMedium': int,
'muo_LHTight': int,
'muo_trigger': int,
# 'muo_priTrack_d0': float,
# 'muo_priTrack_z0': float,
# 'muo_priTrack_d0Sig': float,
# 'muo_priTrack_z0Sig': float,
# 'muo_priTrack_theta': float,
# 'muo_priTrack_qOverP': float,
# 'muo_priTrack_vx': float,
# 'muo_priTrack_vy': float,
# 'muo_priTrack_vz': float,
# 'muo_priTrack_phi': float,
# 'muo_priTrack_chiSquared': float,
# 'muo_priTrack_numberDoF': float,
# 'muo_priTrack_radiusOfFirstHit': float,
# 'muo_priTrack_trackFitter': float,
# 'muo_priTrack_particleHypothesis': float,
# 'muo_priTrack_numberOfUsedHitsdEdx': float,
# 'muo_priTrack_numberOfContribPixelLayers': float,
# 'muo_priTrack_numberOfInnermostPixelLayerHits': float,
# 'muo_priTrack_expectInnermostPixelLayerHit': float,
# 'muo_priTrack_numberOfNextToInnermostPixelLayerHits': float,
# 'muo_priTrack_expectNextToInnermostPixelLayerHit': float,
# 'muo_priTrack_numberOfPixelHits': float,
# 'muo_priTrack_numberOfGangedPixels': float,
# 'muo_priTrack_numberOfGangedFlaggedFakes': float,
# 'muo_priTrack_numberOfPixelSpoiltHits': float,
# 'muo_priTrack_numberOfDBMHits': float,
# 'muo_priTrack_numberOfSCTHits': float,
# 'muo_priTrack_numberOfTRTHits': float,
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
# 'muo_IDTrack_chiSquared': float,
# 'muo_IDTrack_numberDoF': float,
# 'muo_IDTrack_radiusOfFirstHit': float,
# 'muo_IDTrack_trackFitter': float,
# 'muo_IDTrack_particleHypothesis': float,
# 'muo_IDTrack_numberOfUsedHitsdEdx': float,
# 'muo_ET_Core': float,
# 'muo_ET_EMCore': float,
# 'muo_ET_HECCore': float,
'muo_ET_TileCore': float,
# 'muo_FSR_CandidateEnergy': float,
# 'muo_InnerDetectorPt': float,
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
# 'muo_neflowisol20': float,
# 'muo_neflowisol30': float,
# 'muo_neflowisol40': float,
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
# histNtype_total = np.zeros((1,5))
# histTrigPass_total = np.zeros((1,2))

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

    results = pool.map(multiExtract, [(i, counter, path, start, stop) for i, (start, stop) in enumerate(index_ranges)])
    results_np = np.array(results)

    # Concatenate resulting data from the multiple converters
    data = np.concatenate(results_np[:,0])

    # Concatenate data and add to total
    histNtype = np.concatenate(results_np[:,1], axis=0)
    histNtype_total = histNtype_total + np.sum(histNtype, axis=0)

    # Concatenate data and add to total
    histTrigPass = np.concatenate(results_np[:,2], axis=0)
    histTrigPass_total = histTrigPass_total + np.sum(histTrigPass, axis=0)

    # Print the total event count in the file
    log.info("Data shape: {}".format(data.shape))

    # Save output to a file
    if counter == 0:
        saveToFile(filename, data, column_names, column_dtype)
    else:
        appendToFile(filename, data, column_names, column_dtype)


sec = timedelta(seconds=time() - t_start)
log.info(f"Extraction finished. Time spent: {str(sec)}")
