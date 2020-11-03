#!/usr/bin/env python
# -*- coding: utf-8 -*-
print("Program running...")
'''
nohup python -u Zmmg_dataset.py --tag 20201028 ../hdf5Prod/output/root2hdf5/Zmmgam20201022/Zmmgam20201022_0004.h5 ../hdf5Prod/output/root2hdf5/Zmmgam20201022/Zmmgam20201022_0005.h5 ../hdf5Prod/output/root2hdf5/Zmmgam20201022/Zmmgam20201022_0006.h5 ../hdf5Prod/output/root2hdf5/Zmmgam20201022/Zmmgam20201022_0007.h5 ../hdf5Prod/output/root2hdf5/Zmmgam20201022/Zmmgam20201022_0008.h5 ../hdf5Prod/output/root2hdf5/Zmmgam20201022/Zmmgam20201022_0009.h5 2>&1 &> output/logZmmgDataset.txt & disown

    nohup python -u Zmmg_dataset.py --tag Zmmgam20201027_PdgID_Tight  ../hdf5Prod/output/root2hdf5/Zmmgam20201022/Zmmgam20201022_0000.h5   2>&1 &> output/logZmmgDataset.txt & disown
nohup python -u createDataset.py --tag 20201029  ../hdf5Prod/output/root2hdf5/Zmmgam20201029_data/Zmmgam20201029_data_0000.h5   2>&1 &> output/logZmmgDataset.txt & disown

'''

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
parser.add_argument('--outdir', action='store', default="output/MuoPairGammaDataset/", type=str,
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


args.outdir = args.outdir+f"{args.tag}/"
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
def signalSelection(hf, event, invM, pProbe, histNtype, i, process):
    """
    Selects a type for the given electron pair and photon based on a flowchart.

    Arguments:
        hf: File to get variables from.
        event: Event number.
        eTag: Index of tag electron.
        eProbe: Index of probe electron.
        pProbe: Index of probe photon.
        phoOrigin: Demanded origin type of photon.

    Returns:
        The type of the pair:
        0 = background
        1 = signal
    """
    # Origin of photon is correct
    # origPho = (hf['pho_truthOrigin'][event][pProbe] == 3)
    # typeIsoPho = (hf['pho_truthType'][event][pProbe] == 14)
    # print(hf['pho_truthOrigin'][event][pProbe])
    # PdgId of photon is correct
    # pdgId22 = (np.abs(hf['pho_truthPdgId_atlas'][event][pProbe]) == 22)
    Tight = (hf['pho_isPhotonEMTight'][event][pProbe])

    # Return type
    if (Tight):
    # if ((invM/1000 > 80) and (invM/1000 < 100)):
        histNtype[0,0] += 1
        return 1 # Signal
    # elif not pdgId22:
    #     histNtype[0,2] += 1
    #     return 2 # Trash
    else:
        histNtype[0,1] += 1
        return 0 # Bkg
    #

def combinedVariables(hf, event, muoTag, muoProbe, phoProbe):
    """
    Calculate variables of the electron pair and photon.

    Arguments:
        hf: File to get variables from.
        event: Event number.
        eleTag: Index of tag electron.
        eleProbe: Index of probe electron.
        phoProbe: Index of probe photon.

    Returns:
        invM: Invariant mass of combined four-vector.
        et: Transverse energy of combined four-vector.
        eta: Eta of combined four-vector.
        phi: Phi of combined four-vector.
    """
    # Calculate invM etc. of eegamma using: https://github.com/scikit-hep/scikit-hep/blob/master/skhep/math/vectors.py?fbclid=IwAR3C0qnNlxKx-RhGjwo1c1FeZEpWbYqFrNmEqMv5iE-ibyPw_xEqmDYgRpc
    # Get variables
    p1 = hf['muo_pt'][event][muoTag]
    eta1 = hf['muo_eta'][event][muoTag]
    phi1 = hf['muo_phi'][event][muoTag]
    p2 = hf[f'muo_pt'][event][muoProbe]
    eta2 = hf[f'muo_eta'][event][muoProbe]
    phi2 = hf[f'muo_phi'][event][muoProbe]
    p3 = hf[f'pho_et'][event][phoProbe]
    eta3 = hf[f'pho_eta'][event][phoProbe]
    phi3 = hf[f'pho_phi'][event][phoProbe]

    # make four vector
    vecFour1 = vectors.LorentzVector()
    vecFour2 = vectors.LorentzVector()
    vecFour3 = vectors.LorentzVector()

    vecFour1.setptetaphim(p1,eta1,phi1,0)
    vecFour2.setptetaphim(p2,eta2,phi2,0)
    vecFour3.setptetaphim(p3,eta3,phi3,0)

    # calculate invariant mass
    vecFour = vecFour1+vecFour2+vecFour3
    invM = vecFour.mass
    pt = vecFour.pt
    eta = vecFour.eta
    phi = vecFour.phi()

    return invM, pt, eta, phi

def getTagsAndProbes(hf, event, i, process):
    """
    Gets tag and probe indices.

    Arguments:
        hf: File to get variables from.
        event: Event number.

    Returns:
        mTag: Array of tag muons.
        mProbe: Array of probe muons.
        pProbe: Array of probe photons.
    """
    mTag = []
    mProbe = []
    pProbe = []

    # Get ele tags and probes
    for muo in range(len(hf['muo_pt'][event])):
        # ptMuo = hf['muo_pt'][event][muo]/1000
        # origZ = (hf['muo_truthOrigin'][event][muo] == 13)
        # pdgId13 = (np.abs(hf['muo_truthPdgId'][event][muo]) == 13)
        # trigger = hf['muo_trigger'][event][muo]
        # LHTight = hf['muo_LHTight'][event][muo]

        if (hf[ "muo_trigger" ][ event ][ muo ] & hf[ "muo_LHTight" ][ event ][ muo ] & (hf[ "muo_pt" ][ event ][ muo ] > 26000)):
            # print("triggered")
            mTag.append(muo)
        else:
            mProbe.append(muo)
    # ptPho = hf['pho_et'][event][muo]
    # Get pho probes
    pProbe = np.arange(0,len(hf['pho_et'][event]),1)


    return mTag, mProbe, pProbe

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

    # data_temp[ 0, column_names.index( f'muo{muoNr}_truthPdgId' ) ] = hf[ 'muo_truthPdgId' ][ event ][ muo ]
    # data_temp[ 0, column_names.index( f'muo{muoNr}_truthType' ) ] = hf[ 'muo_truthType' ][ event ][ muo ]
    # data_temp[ 0, column_names.index( f'muo{muoNr}_truthOrigin' ) ] = hf[ 'muo_truthOrigin' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo{muoNr}_etcone20' ) ] = hf[ 'muo_etcone20' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_etcone30' ) ] = hf[ 'muo_etcone30' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_etcone40' ) ] = hf[ 'muo_etcone40' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo{muoNr}_ptcone20' ) ] = hf[ 'muo_ptcone20' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_ptcone30' ) ] = hf[ 'muo_ptcone30' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_ptcone40' ) ] = hf[ 'muo_ptcone40' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo{muoNr}_ptvarcone20' ) ] = hf[ 'muo_ptvarcone20' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_ptvarcone30' ) ] = hf[ 'muo_ptvarcone30' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_ptvarcone40' ) ] = hf[ 'muo_ptvarcone40' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo{muoNr}_pt' ) ] = hf[ 'muo_pt' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_eta' ) ] = hf[ 'muo_eta' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_phi' ) ] = hf[ 'muo_phi' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo{muoNr}_muonType' ) ] = hf[ 'muo_muonType' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo{muoNr}_numberOfPrecisionLayers' ) ] = hf[ 'muo_numberOfPrecisionLayers' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_numberOfPrecisionHoleLayers' ) ] = hf[ 'muo_numberOfPrecisionHoleLayers' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo{muoNr}_quality' ) ] = hf[ 'muo_quality' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo{muoNr}_LHLoose' ) ] = hf[ 'muo_LHLoose' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_LHMedium' ) ] = hf[ 'muo_LHMedium' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_LHTight' ) ] = hf[ 'muo_LHTight' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo{muoNr}_trigger' ) ] = hf[ 'muo_LHTight' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_charge' ) ] = hf[ 'muo_charge' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo{muoNr}_innerSmallHits' ) ] = hf[ 'muo_innerSmallHits' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_innerLargeHits' ) ] = hf[ 'muo_innerLargeHits' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_middleSmallHits' ) ] = hf[ 'muo_middleSmallHits' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_middleLargeHits' ) ] = hf[ 'muo_middleLargeHits' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_outerSmallHits' ) ] = hf[ 'muo_outerSmallHits' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_outerLargeHits' ) ] = hf[ 'muo_outerLargeHits' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo{muoNr}_CaloLRLikelihood' ) ] = hf[ 'muo_CaloLRLikelihood' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo{muoNr}_priTrack_d0' ) ] = hf[ 'muo_priTrack_d0' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_priTrack_z0' ) ] = hf[ 'muo_priTrack_z0' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_priTrack_d0Sig' ) ] = hf[ 'muo_priTrack_d0Sig' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_priTrack_z0Sig' ) ] = hf[ 'muo_priTrack_z0Sig' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo{muoNr}_priTrack_chiSquared' ) ] = hf[ 'muo_priTrack_chiSquared' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_priTrack_numberDoF' ) ] = hf[ 'muo_priTrack_numberDoF' ][ event ][ muo ]
    # data_temp[ 0, column_names.index( f'muo{muoNr}_IDTrack_chiSquared' ) ] = hf[ 'muo_IDTrack_chiSquared' ][ event ][ muo ]
    # data_temp[ 0, column_names.index( f'muo{muoNr}_IDTrack_numberDoF' ) ] = hf[ 'muo_IDTrack_numberDoF' ][ event ][ muo ]

    # data_temp[ 0, column_names.index( f'muo{muoNr}_priTrack_numberOfInnermostPixelLayerHits' ) ] = hf[ 'muo_priTrack_numberOfInnermostPixelLayerHits' ][ event ][ muo ]
    # data_temp[ 0, column_names.index( f'muo{muoNr}_priTrack_expectInnermostPixelLayerHit' ) ] = hf[ 'muo_priTrack_expectInnermostPixelLayerHit' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_priTrack_numberOfPixelHits' ) ] = hf[ 'muo_priTrack_numberOfPixelHits' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_priTrack_numberOfSCTHits' ) ] = hf[ 'muo_priTrack_numberOfSCTHits' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_priTrack_numberOfTRTHits' ) ] = hf[ 'muo_priTrack_numberOfTRTHits' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo{muoNr}_neflowisol20' ) ] = hf[ 'muo_neflowisol20' ][ event ][ muo ]
    # data_temp[ 0, column_names.index( f'muo{muoNr}_neflowisol30' ) ] = hf[ 'muo_neflowisol30' ][ event ][ muo ]
    # data_temp[ 0, column_names.index( f'muo{muoNr}_neflowisol40' ) ] = hf[ 'muo_neflowisol40' ][ event ][ muo ]

    # data_temp[ 0, column_names.index( f'muo{muoNr}_ET_TileCore' ) ] = hf[ 'muo_ET_TileCore' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_MuonSpectrometerPt' ) ] = hf[ 'muo_MuonSpectrometerPt' ][ event ][ muo ]

    # data_temp[ 0, column_names.index( f'muo{muoNr}_deltaphi_0' ) ] = hf[ 'muo_deltaphi_0' ][ event ][ muo ]
    # data_temp[ 0, column_names.index( f'muo{muoNr}_deltaphi_1' ) ] = hf[ 'muo_deltaphi_1' ][ event ][ muo ]
    # data_temp[ 0, column_names.index( f'muo{muoNr}_deltatheta_0' ) ] = hf[ 'muo_deltatheta_0' ][ event ][ muo ]
    # data_temp[ 0, column_names.index( f'muo{muoNr}_deltatheta_1' ) ] = hf[ 'muo_deltatheta_1' ][ event ][ muo ]
    #
    # data_temp[ 0, column_names.index( f'muo{muoNr}_sigmadeltaphi_0' ) ] = hf[ 'muo_sigmadeltaphi_0' ][ event ][ muo ]
    # data_temp[ 0, column_names.index( f'muo{muoNr}_sigmadeltaphi_1' ) ] = hf[ 'muo_sigmadeltaphi_1' ][ event ][ muo ]
    # data_temp[ 0, column_names.index( f'muo{muoNr}_sigmadeltatheta_0' ) ] = hf[ 'muo_sigmadeltatheta_0' ][ event ][ muo ]
    # data_temp[ 0, column_names.index( f'muo{muoNr}_sigmadeltatheta_1' ) ] = hf[ 'muo_sigmadeltatheta_1' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo{muoNr}_etconecoreConeEnergyCorrection' ) ] = hf[ 'muo_etconecoreConeEnergyCorrection' ][ event ][ muo ]
    # data_temp[ 0, column_names.index( f'muo{muoNr}_neflowisolcoreConeEnergyCorrection' ) ] = hf[ 'muo_neflowisolcoreConeEnergyCorrection' ][ event ][ muo ]
    # data_temp[ 0, column_names.index( f'muo{muoNr}_ptconecoreTrackPtrCorrection' ) ] = hf[ 'muo_ptconecoreTrackPtrCorrection' ][ event ][ muo ]
    # data_temp[ 0, column_names.index( f'muo{muoNr}_topoetconecoreConeEnergyCorrection' ) ] = hf[ 'muo_topoetconecoreConeEnergyCorrection' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo{muoNr}_InnerDetectorPt' ) ] = hf[ 'muo_InnerDetectorPt' ][ event ][ muo ]
    # data_temp[ 0, column_names.index( f'muo{muoNr}_DFCommonMuonsPreselection' ) ] = hf[ 'muo_DFCommonMuonsPreselection' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_author' ) ] = hf[ 'muo_author' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_allAuthors' ) ] = hf[ 'muo_allAuthors' ][ event ][ muo ]

    data_temp[ 0, column_names.index( f'muo{muoNr}_scatteringCurvatureSignificance' ) ] = hf[ 'muo_scatteringCurvatureSignificance' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_scatteringNeighbourSignificance' ) ] = hf[ 'muo_scatteringNeighbourSignificance' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_momentumBalanceSignificance' ) ] = hf[ 'muo_momentumBalanceSignificance' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_EnergyLoss' ) ] = hf[ 'muo_EnergyLoss' ][ event ][ muo ]
    data_temp[ 0, column_names.index( f'muo{muoNr}_energyLossType' ) ] = hf[ 'muo_energyLossType' ][ event ][ muo ]



def addPhotonVariables(hf, event, data_temp, pho):
    """
    Takes variables from file and adds them to a temporary array, that is later
    appended to the returned data.

    Arguments:
        hf: File to get variables from.
        event: Event number.
        data_temp: Numpy array to add variables to.
        pho: Photon index.

    Returns:
        Nothing. Data is set in existing array.
    """
    # data_temp[ 0, column_names.index( 'pho_truthPdgId_egam') ] = hf[ 'pho_truthPdgId_egam' ][ event][ pho ]
    # data_temp[ 0, column_names.index( 'pho_truthPdgId_atlas') ] = hf[ 'pho_truthPdgId_atlas' ][ event][ pho ]
    # data_temp[ 0, column_names.index( 'pho_egamTruthParticle') ] = hf[ 'pho_egamTruthParticle' ][ event][ pho ]
    # data_temp[ 0, column_names.index( 'pho_truthType') ] = hf[ 'pho_truthType' ][ event][ pho ]
    # data_temp[ 0, column_names.index( 'pho_truthOrigin') ] = hf[ 'pho_truthOrigin' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_isPhotonEMLoose') ] = hf[ 'pho_isPhotonEMLoose' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_isPhotonEMTight') ] = hf[ 'pho_isPhotonEMTight' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_e') ] = hf[ 'pho_e' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_eta') ] = hf[ 'pho_eta' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_phi') ] = hf[ 'pho_phi' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_et') ] = hf[ 'pho_et' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_Rhad1') ] = hf[ 'pho_Rhad1' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_Rhad') ] = hf[ 'pho_Rhad' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_weta2') ] = hf[ 'pho_weta2' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_Rphi') ] = hf[ 'pho_Rphi' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_Reta') ] = hf[ 'pho_Reta' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_Eratio') ] = hf[ 'pho_Eratio' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_f1') ] = hf[ 'pho_f1' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_wtots1') ] = hf[ 'pho_wtots1' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_DeltaE') ] = hf[ 'pho_DeltaE' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_weta1') ] = hf[ 'pho_weta1' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_fracs1') ] = hf[ 'pho_fracs1' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_ConversionType') ] = hf[ 'pho_ConversionType' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_ConversionRadius') ] = hf[ 'pho_ConversionRadius' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_VertexConvEtOverPt') ] = hf[ 'pho_VertexConvEtOverPt' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_VertexConvPtRatio') ] = hf[ 'pho_VertexConvPtRatio' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_topoetcone20') ] = hf[ 'pho_topoetcone20' ][ event][ pho ]
    # data_temp[ 0, column_names.index( 'pho_topoetcone30') ] = hf[ 'pho_topoetcone30' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_topoetcone40') ] = hf[ 'pho_topoetcone40' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_ptvarcone20') ] = hf[ 'pho_ptvarcone20' ][ event][ pho ]
    # data_temp[ 0, column_names.index( 'pho_ptvarcone30') ] = hf[ 'pho_ptvarcone30' ][ event][ pho ]
    # data_temp[ 0, column_names.index( 'pho_ptvarcone40') ] = hf[ 'pho_ptvarcone40' ][ event][ pho ]


def multiExtract(arguments):
    """
    Extracts the variables of the hdf5 file produced with root2hdf5.
    Uses simple tag and probe:
        tag = electron that has triggered
        probe = any other electron in container
    Selects signal type based on pdgid, truthOrigin and truthParticle

    Arguments:
        process: process number
        counter: file counter
        path: path to file
        start: event index in file to start at
        stop: event index in file to stop at

    Returns:
        Data of electron pairs in numpy array
        Counters for histogram over signal selection and cut flow
    """
    # Unpack arguments
    process, counter, path, start, stop = arguments

    # Counters for histograms
    histNtype = np.zeros((1,3))
    # histTrigPass = np.zeros((1,2))

    # Read ROOT data from file
    log.info("[{}]  Importing data from {}".format(process,path))
    hf = h5py.File(path, "r")

    # Numpy array to return
    data = np.empty((0,len(column_names)), float)

    # Total number of events in batch
    n_events = stop-start

    # Run over all events in the start stop range
    for i, event in enumerate(np.arange(start,stop)):
        # Print information on progress
        if i%100==0:
            log.info("[{}]  {} of {} events examined".format(process,i,n_events))

        # Get tags and probes
        mTag, mProbe, pProbe = getTagsAndProbes(hf, event, i, process)
        # print(mTag, mProbe, pProbe)
        nTag = len(mTag)

        # If there is at least one tag, 2 muons and 1 photon -> get values
        if ( len(mTag)>0 & (len(mProbe)+len(mTag)>1) & (len(pProbe)>0) ):
            # Event passed trigger cut - add to histogram
            # histTrigPass[0,1] += 1

            for iMuoTag, muoTag in enumerate(mTag):
                # Get probes
                muoProbes = mProbe.copy()
                phoProbes = pProbe.copy()
                # Append subsecuent electron tags to electron probes (applies if there is more than one tag electron)
                for muo in (mTag[(iMuoTag+1):]):
                    muoProbes.append(muo)

                for iMuoProbe, muoProbe in enumerate(muoProbes):
                    if (hf['muo_charge'][event][muoTag] * hf['muo_charge'][event][muoProbe] < 0 ): # opposite sign
                        for iPhoProbe, phoProbe in enumerate(phoProbes):
                            # Create array for saving data
                            data_temp = np.zeros((1,len(column_names)))
                            # Calculate variables of the electron pair
                            invM, pt, eta, phi = combinedVariables(hf, event, muoTag, muoProbe, phoProbe)

                            # Get type
                            selection = signalSelection(hf, event, invM, phoProbe, histNtype, i, process)
                            if selection == 2:
                                continue

                            # Add event variables to array
                            data_temp[ 0, column_names.index( 'NvtxReco' ) ] = np.int(hf['NvtxReco'][event])
                            data_temp[ 0, column_names.index( 'correctedScaledAverageMu' ) ] = hf[ 'correctedScaledAverageMu' ][ event ]
                            data_temp[ 0, column_names.index( 'correctedScaledActualMu' ) ] = hf[ 'correctedScaledActualMu' ][ event ]
                            data_temp[ 0, column_names.index( 'invM' ) ] = invM/1000
                            data_temp[ 0, column_names.index( 'pt' ) ] = pt/1000
                            data_temp[ 0, column_names.index( 'eta' ) ] = eta
                            data_temp[ 0, column_names.index( 'phi' ) ] = phi
                            data_temp[ 0, column_names.index( 'type' ) ] = selection

                            # Add electron variables to array
                            addMuonVariables(hf, event, data_temp, 1, muoTag)
                            addMuonVariables(hf, event, data_temp, 2, muoProbe)
                            addPhotonVariables(hf, event, data_temp, phoProbe)

                            # Append data to array that is returned
                            data = np.append(data, data_temp, axis=0)


    return data, histNtype

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

def plotHistogram(histVal, fname, names, title, xlabel):
    """
    Simply plots histogram and saves it.

    Arguments:
        histVal: Values on y axis.
        fname: filename (directory is taken from script's args).
        names: Names on x axis.
        title: Title of histogram.
        xlabel: xlabel of histogram.

    Returns:
        Nothing.
    """
    log.info("Plot and save histogram: {}".format(args.outdir + fname))
    # Length, position and values of data
    n = len(histVal)
    x = np.arange(n)
    val = histVal

    # Plot histogram
    fig, ax = plt.subplots()
    ax.bar(x, height=val)           # Create bar plot
    plt.xticks(x, names)            # Rename x ticks
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")

    # Add values to histogram
    shift = np.max(val)*0.01
    for i in range(n):
        ax.text(x[i], val[i]+shift, f"{int(val[i])}", horizontalalignment='center')

    # Save histogram
    plt.tight_layout()
    fig.savefig(args.outdir + fname)

#============================================================================
# Define column names and dtypes (Used when saving)
#============================================================================
column_dtype = {'correctedScaledAverageMu': float,
                'correctedScaledActualMu': float,
                'NvtxReco': float,
                'eventWeight':float,
                "invM" : float,
                "pt" : float,
                "eta" : float,
                "phi" : float,
                "type" : int,
                "isATLAS" : int,
                ####
                ####
                # 'muo1_truthPdgId': int,
                # 'muo1_truthType': int,
                # 'muo1_truthOrigin': int,
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
                'muo1_charge': int,
                'muo1_innerSmallHits': int,
                'muo1_innerLargeHits': int,
                'muo1_middleSmallHits': int,
                'muo1_middleLargeHits': int,
                'muo1_outerSmallHits': int,
                'muo1_outerLargeHits': int,
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
                'muo1_energyLossType': int,
                # 'muo1_spectrometerFieldIntegral': float,
                'muo1_scatteringCurvatureSignificance': float,
                'muo1_scatteringNeighbourSignificance': float,
                'muo1_momentumBalanceSignificance': float,
                # 'muo1_segmentDeltaEta': float,
                'muo1_CaloLRLikelihood': float,
                'muo1_EnergyLoss': float,
                # 'muo1_CaloMuonIDTag': float,
                # 'muo1_DFCommonGoodMuon': float,
                # 'muo1_DFCommonMuonsPreselection': float,
                'muo1_LHLoose': int,
                'muo1_LHMedium': int,
                'muo1_LHTight': int,
                'muo1_trigger': int,
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
                'muo1_priTrack_chiSquared': float,
                'muo1_priTrack_numberDoF': float,
                # 'muo1_priTrack_radiusOfFirstHit': float,
                # 'muo1_priTrack_trackFitter': float,
                # 'muo1_priTrack_particleHypothesis': float,
                # 'muo1_priTrack_numberOfUsedHitsdEdx': float,
                # 'muo1_priTrack_numberOfContribPixelLayers': float,
                # 'muo1_priTrack_numberOfInnermostPixelLayerHits': float,
                # 'muo1_priTrack_expectInnermostPixelLayerHit': float,
                # 'muo1_priTrack_numberOfNextToInnermostPixelLayerHits': float,
                # 'muo1_priTrack_expectNextToInnermostPixelLayerHit': float,
                'muo1_priTrack_numberOfPixelHits': float,
                # 'muo1_priTrack_numberOfGangedPixels': float,
                # 'muo1_priTrack_numberOfGangedFlaggedFakes': float,
                # 'muo1_priTrack_numberOfPixelSpoiltHits': float,
                # 'muo1_priTrack_numberOfDBMHits': float,
                'muo1_priTrack_numberOfSCTHits': float,
                'muo1_priTrack_numberOfTRTHits': float,
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
                # 'muo1_ET_TileCore': float,
                # 'muo1_FSR_CandidateEnergy': float,
                'muo1_InnerDetectorPt': float,
                'muo1_MuonSpectrometerPt': float,
                # 'muo1_combinedTrackOutBoundsPrecisionHits': float,
                # 'muo1_coreMuonEnergyCorrection': float,
                # 'muo1_deltaphi_0': float,
                # 'muo1_deltaphi_1': float,
                # 'muo1_deltatheta_0': float,
                # 'muo1_deltatheta_1': float,
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
                # 'muo1_sigmadeltaphi_0': float,
                # 'muo1_sigmadeltaphi_1': float,
                # 'muo1_sigmadeltatheta_0': float,
                # 'muo1_sigmadeltatheta_1': float,
                'muo1_etconeCorrBitset': float,
                'muo1_neflowisol20': float,
                # 'muo1_neflowisol30': float,
                # 'muo1_neflowisol40': float,
                # 'muo1_neflowisolCorrBitset': float,
                # 'muo1_neflowisolcoreConeEnergyCorrection': float,
                # 'muo1_ptconeCorrBitset': float,
                # 'muo1_ptconecoreTrackPtrCorrection': float,
                # 'muo1_topoetconeCorrBitset': float,
                # 'muo1_topoetconecoreConeEnergyCorrection': float,
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
                ######
                ######
                # 'muo2_truthPdgId': int,
                # 'muo2_truthType': int,
                # 'muo2_truthOrigin': int,
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
                'muo2_charge': int,
                'muo2_innerSmallHits': int,
                'muo2_innerLargeHits': int,
                'muo2_middleSmallHits': int,
                'muo2_middleLargeHits': int,
                'muo2_outerSmallHits': int,
                'muo2_outerLargeHits': int,
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
                'muo2_energyLossType': int,
                # 'muo2_spectrometerFieldIntegral': float,
                'muo2_scatteringCurvatureSignificance': float,
                'muo2_scatteringNeighbourSignificance': float,
                'muo2_momentumBalanceSignificance': float,
                # 'muo2_segmentDeltaEta': float,
                'muo2_CaloLRLikelihood': float,
                'muo2_EnergyLoss': float,
                # 'muo2_CaloMuonIDTag': float,
                # 'muo2_DFCommonGoodMuon': float,
                # 'muo2_DFCommonMuonsPreselection': float,
                'muo2_LHLoose': int,
                'muo2_LHMedium': int,
                'muo2_LHTight': int,
                'muo2_trigger': int,
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
                'muo2_priTrack_chiSquared': float,
                'muo2_priTrack_numberDoF': float,
                # 'muo2_priTrack_radiusOfFirstHit': float,
                # 'muo2_priTrack_trackFitter': float,
                # 'muo2_priTrack_particleHypothesis': float,
                # 'muo2_priTrack_numberOfUsedHitsdEdx': float,
                # 'muo2_priTrack_numberOfContribPixelLayers': float,
                # 'muo2_priTrack_numberOfInnermostPixelLayerHits': float,
                # 'muo2_priTrack_expectInnermostPixelLayerHit': float,
                # 'muo2_priTrack_numberOfNextToInnermostPixelLayerHits': float,
                # 'muo2_priTrack_expectNextToInnermostPixelLayerHit': float,
                'muo2_priTrack_numberOfPixelHits': float,
                # 'muo2_priTrack_numberOfGangedPixels': float,
                # 'muo2_priTrack_numberOfGangedFlaggedFakes': float,
                # 'muo2_priTrack_numberOfPixelSpoiltHits': float,
                # 'muo2_priTrack_numberOfDBMHits': float,
                'muo2_priTrack_numberOfSCTHits': float,
                'muo2_priTrack_numberOfTRTHits': float,
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
                # 'muo2_ET_TileCore': float,
                # 'muo2_FSR_CandidateEnergy': float,
                'muo2_InnerDetectorPt': float,
                'muo2_MuonSpectrometerPt': float,
                # 'muo2_combinedTrackOutBoundsPrecisionHits': float,
                # 'muo2_coreMuonEnergyCorrection': float,
                # 'muo2_deltaphi_0': float,
                # 'muo2_deltaphi_1': float,
                # 'muo2_deltatheta_0': float,
                # 'muo2_deltatheta_1': float,
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
                # 'muo2_sigmadeltaphi_0': float,
                # 'muo2_sigmadeltaphi_1': float,
                # 'muo2_sigmadeltatheta_0': float,
                # 'muo2_sigmadeltatheta_1': float,
                'muo2_etconeCorrBitset': float,
                'muo2_neflowisol20': float,
                # 'muo2_neflowisol30': float,
                # 'muo2_neflowisol40': float,
                # 'muo2_neflowisolCorrBitset': float,
                # 'muo2_neflowisolcoreConeEnergyCorrection': float,
                # 'muo2_ptconeCorrBitset': float,
                # 'muo2_ptconecoreTrackPtrCorrection': float,
                # 'muo2_topoetconeCorrBitset': float,
                # 'muo2_topoetconecoreConeEnergyCorrection': float,
                # 'muo2_CT_EL_Type': float,
                # 'muo2_CT_ET_Core': float,
                # 'muo2_CT_ET_FSRCandidateEnergy': float,
                # 'muo2_CT_ET_LRLikelihood': float,
                # 'muo2_d0_staco': float,
                # 'muo2_phi0_staco': float,
                # 'muo2_qOverPErr_staco': float,
                # 'muo2_qOverP_staco': float,
                # 'muo2_theta_staco': float,
                # 'muo2_z0_staco': float,
                #####
                #####
                # "pho_truthPdgId_egam" : int,
                # "pho_truthPdgId_atlas" : int,
                # "pho_egamTruthParticle" : int,
                # "pho_truthType" : int,
                # "pho_truthOrigin" : int,
                "pho_isPhotonEMLoose" : int,
                "pho_isPhotonEMTight" : int,
                "pho_e" : float,
                "pho_eta" : float,
                "pho_phi" : float,
                "pho_et" : float,
                "pho_Rhad1" : float,
                "pho_Rhad" : float,
                "pho_weta2" : float,
                "pho_Rphi" : float,
                "pho_Reta" : float,
                "pho_Eratio" : float,
                "pho_f1" : float,
                "pho_wtots1" : float,
                "pho_DeltaE" : float,
                "pho_weta1" : float,
                "pho_fracs1" : float,
                "pho_ConversionType" : float,
                "pho_ConversionRadius" : float,
                "pho_VertexConvEtOverPt" : float,
                "pho_VertexConvPtRatio" : float,
                'pho_maxEcell_time': float,
                'pho_maxEcell_energy': float,
                'pho_core57cellsEnergyCorrection': float,
                'pho_r33over37allcalo': float,
                'pho_GradientIso': float,
                "pho_topoetcone20": float,
                # "pho_topoetcone30": float,
                "pho_topoetcone40": float,
                "pho_ptvarcone20": float
                # "pho_ptvarcone30": float,
                # "pho_ptvarcone40": float
                }

column_names = list(column_dtype.keys())


#============================================================================
# Main
#============================================================================
# Total counters for signal selection diagram
histNtype_total = np.zeros((1,3))
# histTrigPass_total = np.zeros((1,2))

# countPho13 = 0
# countPho14 = 0

# create file name and check if the file already exists
filename = f"{args.tag}.h5"
if os.path.isfile(args.outdir + filename):
    log.error(f"Output file algready exists - please remove yourself. Output: {args.outdir + filename}")
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
    # histTrigPass = np.concatenate(results_np[:,2], axis=0)
    # histTrigPass_total = histTrigPass_total + np.sum(histTrigPass, axis=0)

    # Print the total event count in the file
    log.info("Data shape: {}".format(data.shape))

    # Save output to a file
    if counter == 0:
        saveToFile(filename, data, column_names, column_dtype)
    else:
        appendToFile(filename, data, column_names, column_dtype)

# Create and save figure of signal selection
plotHistogram(histVal=histNtype_total[0],
              fname=args.tag+"_sigSelDiag.png",
              names=["Signal", "Background", "Trash"],
              title = f"Signal selection ({args.tag})",
              xlabel = "Selection types")
# plotHistogram(histVal=histTrigPass_total[0],
#               fname=args.tag+"_trigPassDiag.png",
#               names=["No trigger in event", "At least one trigger in event"],
#               title = f"Events that passes trigger ({args.tag})",
#               xlabel = "")


sec = timedelta(seconds=time() - t_start)
log.info(f"Extraction finished. Time spent: {str(sec)}")
