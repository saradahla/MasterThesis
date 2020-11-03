#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' nohup python -u .py --tag fil0  ../file1.h5 ../file2.h5 2>&1 &> logNavn.txt & disown

nohup python -u gamma_Dataset.py --tag 20201027 ../hdf5Prod/output/root2hdf5/Zmmgam20201022/Zmmgam20201022_0000.h5 2>&1 &> output/logPhoDataset.txt & disown
nohup python -u gamma_Dataset.py --tag 20201027_2 ../hdf5Prod/output/root2hdf5/Zmmgam20201022/Zmmgam20201022_0000.h5 ../hdf5Prod/output/root2hdf5/Zmmgam20201022/Zmmgam20201022_0001.h5 ../hdf5Prod/output/root2hdf5/Zmmgam20201022/Zmmgam20201022_0002.h5 2>&1 &> output/logPhoDataset.txt & disown

nohup python -u gamma_Dataset.py --tag 20201028 ../hdf5Prod/output/root2hdf5/Zmmgam20201022/Zmmgam20201022_0000.h5 ../hdf5Prod/output/root2hdf5/Zmmgam20201022/Zmmgam20201022_0001.h5 ../hdf5Prod/output/root2hdf5/Zmmgam20201022/Zmmgam20201022_0002.h5 ../hdf5Prod/output/root2hdf5/Zmmgam20201022/Zmmgam20201022_0003.h5 2>&1 &> output/logPhoDataset.txt & disown


'''
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
parser = argparse.ArgumentParser(description="Extract data from HDF5 files into flat HDF5 files for PID and ISO.")
parser.add_argument('--tag', action='store', type=str, required=True,
                    help='Tag the data category (Zee, Wev, etc.).')
parser.add_argument('--outdir', action='store', default="output/pho_Dataset/", type=str,
                    help='Output directory.')
parser.add_argument('paths', type=str, nargs='+',
                    help='Hdf5 file(s) to be converted.')
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
    data_temp[ 0, column_names.index( 'pho_truthPdgId_atlas') ] = hf[ 'pho_truthPdgId_atlas' ][ event][ pho ]
    # data_temp[ 0, column_names.index( 'pho_egamTruthParticle') ] = hf[ 'pho_egamTruthParticle' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_truthType') ] = hf[ 'pho_truthType' ][ event][ pho ]
    data_temp[ 0, column_names.index( 'pho_truthOrigin') ] = hf[ 'pho_truthOrigin' ][ event][ pho ]
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
        nPho = np.shape(hf[ 'pho_truthType' ][ event ])[0]

        for pho in range(nPho):
            #log.info("[{}] Number of muons is {} ".format(process,nMuo))

            data_temp = np.zeros((1,len(column_names)))


            # Add event variables to array
            data_temp[ 0, column_names.index( 'NvtxReco' ) ] = np.int(hf['NvtxReco'][event])
            data_temp[ 0, column_names.index( 'correctedScaledAverageMu' ) ] = hf[ 'correctedScaledAverageMu' ][ event ]
            data_temp[ 0, column_names.index( 'correctedScaledActualMu' ) ] = hf[ 'correctedScaledActualMu' ][ event ]
            # Add muon variables to array

            addPhotonVariables(hf, event, data_temp, pho)

            data = np.append(data, data_temp, axis=0)


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


#============================================================================
# Define column names and dtypes
#============================================================================
# pid_vars = ["muo_deltatheta_1", "muo_quality", "muo_numberOfPrecisionLayers",
#             "muo_MuonSpectrometerPt", "muo_ET_TileCore"]
#
# iso_vars = ["muo_ptconecoreTrackPtrCorrection", "muo_etconecoreConeEnergyCorrection", "muo_pt",
#             "muo_topoetconecoreConeEnergyCorrection", "muo_neflowisolcoreConeEnergyCorrection",
#             "muo_ptcone20", "muo_etcone20"]

column_dtype = {
'correctedScaledAverageMu': float,
'correctedScaledActualMu': float,
'NvtxReco': float,
'eventWeight':float,
####
####
# "pho_truthPdgId_egam" : int,
"pho_truthPdgId_atlas" : int,
# "pho_egamTruthParticle" : int,
"pho_truthType" : int,
"pho_truthOrigin" : int,
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
