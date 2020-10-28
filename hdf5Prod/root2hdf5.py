#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
    How to run, write in terminal: python root2hdf5.py --tag *navn* --tree analysis *path navn*

    exempel:
    nohup python -u root2hdf5.py --tag Data010920 --tree analysis --datatype Data ../../storage/dataAnalysis/data18_13TeV.00349693.physics_Main.deriv.DAOD_MUON1.f933_m1960_p3553.root 2>&1 &> output/logDataToHdf5.txt & disown

    nohup python -u root2hdf5.py --tag Wmn310820 --tree analysis --datatype MC ../../storage/dataAnalysis/mc16_13TeV.361101.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Wplusmunu.deriv.DAOD_MUON1.e3601_e5984_s3126_r10724_r10726_p3629_wTrigger.root 2>&1 &> output/logWmnToHdf5.txt & disown

    nohup python -u root2hdf5.py --tag bb310820 --tree analysis --datatype MC ../../storage/dataAnalysis/mc16_13TeV.361250.Pythia8B_A14_NNPDF23LO_bbTomu15.deriv.DAOD_MUON1.e3878_e5984_s3126_r10724_r10726_p3629_wTrigger.root 2>&1 &> output/logbbToHdf5.txt & disown

    nohup python -u root2hdf5.py --tag Zmm310820 --tree analysis --datatype MC ../../storage/dataAnalysis/mc16_13TeV.361107.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Zmumu.deriv.DAOD_MUON1.e3601_e5984_s3126_r10201_r10210_p3629_wTrigger.root 2>&1 &> output/logZmmToHdf5.txt & disown
    nohup python -u root2hdf5.py --tag Zmumugam --tree analysis --datatype MC ../../storage/dataAnalysis/mc16_13TeV.366145.Sh_224_NN30NNLO_mumugamma_LO_pty_7_15.deriv.DAOD_MUON5.e7006_e5984_s3126_r10201_r10210_p3980.root 2>&1 &> output/logZmmg.txt & disown
    nohup python -u root2hdf5.py --tag Zmm151020 --tree analysis --datatype MC ../../storage/dataAnalysis/151020/mc16_13TeV.361107.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Zmumu.deriv.DAOD_MUON1.e3601_s3126_r10724_p3629.root 2>&1 &> output/logZmmToHdf5_151020.txt & disown
    nohup python -u root2hdf5.py --tag Wmn151020 --tree analysis --datatype MC ../../storage/dataAnalysis/151020/mc16_13TeV.361101.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Wplusmunu.deriv.DAOD_MUON1.e3601_e5984_s3126_r10724_r10726_p3629.root 2>&1 &> output/logWmnToHdf5_151020.txt & disown
    nohup python -u root2hdf5.py --tag Zmmgam20201022 --tree analysis --datatype MC ../../storage/dataAnalysis/mc16_13TeV.301536.Sherpa_CT10_mumugammaPt10_35.deriv.DAOD_EGAM4.e3952_s3126_r10201_r10210_p3956.root 2>&1 &> output/logZmmgToHdf5_221020.txt & disown

'''


print("Program running...")

import warnings
warnings.filterwarnings('ignore', 'ROOT .+ is currently active but you ')
warnings.filterwarnings('ignore', 'numpy .+ is currently installed but you ')

import re
import h5py
import json
import numpy as np
import logging as log
import multiprocessing
from subprocess import call
import ROOT
import argparse
import gc
import os
from time import time
from datetime import timedelta

from utils import mkdir
from variableLists import remove_branches, remove_branchesData, remove_branches_pho

# Logging style and levelo
log.basicConfig(format='[%(levelname)s] %(message)s', level=log.INFO)
log.info("Packages imported")

# Start "timer"
t_start = time()

# Command line options
parser = argparse.ArgumentParser(description="Convert ROOT files with nested structure into flat HDF5 files.")
parser.add_argument('--tag', action='store', type=str, required=True,
                    help='Tag the data category (Zee, Wev, etc.).')
parser.add_argument('--stop', action='store', default=None, type=int,
                    help='Maximum number of events to read.')
parser.add_argument('--max-processes', action='store', default=10, type=int,
                    help='Maximum number of concurrent processes to use.')
#parser.add_argument('--outdir', action='store', default="output/full", type=str,
parser.add_argument('--outdir', action='store', default="output/root2hdf5/", type=str,
                    help='Output directory.')
parser.add_argument('--max-input-files', action='store', default=0, type=int,
                    help='Use at most X files. Default is all.')
parser.add_argument('paths', type=str, nargs='+',
                    help='ROOT file(s) to be converted.')
parser.add_argument('--selection', action='store', default=None, type=str,
                    help='selection string on tree')
parser.add_argument('--datatype', action='store', default='MC', type=str,
                    choices=['MC', 'Data'], help='Real Data or MC')
parser.add_argument('--tree', action='store', required=True, type=str,
                    help='Name of the tree to save!')

args = parser.parse_args()

# Validate arguments
if not args.paths:
    log.error("No ROOT files were specified.")
    quit()

if args.max_processes > 20:
    log.error("The requested number of processes ({}) is excessive (>20). Exiting.".format(args.max_processes))
    quit()

# Maximum number of events
if args.stop is not None:
    args.stop = int(args.stop)
else:
    args.stop = 100_000_000

# Standard selection is no two tags, and tag must be electron (this doesn't work for background or dumps which have no tags)
if args.selection == None:
    args.selection = ""

# Make and set the output directory to tag, if it doesn't already exist
# Will stop if the output already exists since re-running is either not needed or unwanted
# If it's wanted, delete the output first yourself
args.outdir = args.outdir+args.tag+"/"
if os.path.exists(args.outdir):
    log.error("Output already exists - please remove yourself.")
    quit()
else:
    mkdir(args.outdir)

# Sort paths, because it's nice
args.paths = sorted(args.paths)

# File number counter (incremented first in loop)
counter = -1


#============================================================================
# Functions
#============================================================================

def converter(arguments):
    """
    Process converting standard-format ROOT file to HDF5 file with cell
    content.

    Arguments:
        path: Path to the ROOT file to be converted.
        args: Namespace containing command-line arguments, to configure the
            reading and writing of files.

    Returns:
        Converted data in numpy array format
    """
    global args
    # Unpack arguments
    index, counter, path, start, stop = arguments

    # Suppress warnings like these when loading the files: TClass::Init:0: RuntimeWarning: no dictionary for class [bla] is available
    ROOT.gErrorIgnoreLevel = ROOT.kError

    # Split indexes into 10 sets.
    index_edges = list(map(int, np.linspace(start, stop, 10, endpoint=True)))
    index_ranges = zip(index_edges[:-1], index_edges[1:])

    import root_numpy
    # Read-in data from ROOT.TTree
    all_branches = root_numpy.list_branches(path, args.tree)

    # Any branches that needs to be removed is defined in "variableLists.py"
    # remove = remove_branches()
    remove = remove_branches_pho()
    # remove = remove_branchesData()

    keep_branches = sorted(list(set(all_branches)-set(remove)))

    one_evts_array = []
    for i, (loop_start, loop_stop) in enumerate(index_ranges):
        array = root_numpy.root2array(path, args.tree, start=loop_start, stop=loop_stop, selection=args.selection, branches = keep_branches, warn_missing_tree=True)

        ROOT.gErrorIgnoreLevel = ROOT.kInfo

        n_evts = len(array)
        # If NO events survived, it's probably the selection
        if n_evts == 0:
            print("n_evts = 0")
            return
        # If only one event survives ( can happen with small files) the tracks can't be saved properly???, for now add all of these and save them later
        if n_evts == 1:
            one_evts_array.append(array)
            continue
        # Convert to HDF5-ready format.
        data = convert_to_hdf5(array)

        if (args.tree == 'el_tree') and args.datatype == 'MC':
            scale = scale_eventWeight(data['mcChannelNumber'][0])
            data['event_totalWeight'] *= scale

        # Save output of every subprocess to a file
        filename = '{:s}_{:04d}.h5'.format(args.tag, index)
        if counter == 0 and i == 0:
            saveToFile(filename, data)
        else:
            appendToFile(filename, data)

        del data, array
        gc.collect()

    # Add all arrays with only one event and save them to the output file
    if len(one_evts_array) > 1:
        one_evts_array = np.concatenate(one_evts_array)
        one_evts_data = convert_to_hdf5(one_evts_array)
        filename = '{:s}_{:04d}.h5'.format(args.tag, index)
        appendToFile(filename, one_evts_data)


def scale_eventWeight(MCChannelNumber):
    root_file = ROOT.TFile(f"/groups/hep/ehrke/storage/NtuplefromGrid/Hists/v06Combined/hist.mc.combined.{MCChannelNumber}.root")
    hist = root_file.Get("SumOfWeights")

    scale = ( hist.GetBinContent(4) / hist.GetBinContent(1) ) / hist.GetBinContent(2)
    root_file.Close()
    del root_file
    return scale


def convert_types_to_hdf5(x):
    """
    Variable-length containers are converted to `h5py.special_dtype` with the appropriate `vlen` type.

    Arguments:
        x: Numpy-type variable to be type-checked.

    Returns:
        Same numpy type for scalars and h5py dtype for variable-length containers.
    """
    if 'ndarray' in str(type(x)):
        try:
            return h5py.special_dtype(vlen=convert_types_to_hdf5(x[0]))
        except:
            return h5py.special_dtype(vlen=np.float32)
    elif isinstance(x, str):
        return h5py.special_dtype(vlen=str)
    else:
        return x.dtype



def convert_to_hdf5(data):
    """
    Method to convert standard array to suitable format for classifier.

    Arguments:
        data: numpy array returned by root_numpy.tree2array, to be formatted.

    Returns:
        Flattened numpy recarray prepared for saving to HDF5 file.
    """

    # Format output as numpy structured arrays.
    formats = [convert_types_to_hdf5(data[0][var]) for var in data.dtype.names]
    for pair in zip(data.dtype.names, formats):
        try:
            np.dtype([pair])
        except:
            print("Problem for {}".format(pair))
    dtype  = np.dtype(list(zip(data.dtype.names, formats)))
    output = np.array([tuple([d[var] for var in data.dtype.names]) for d in data], dtype=dtype)

    return output


def imagesToArray(data):
    """
    Converts the images retrieved from the ROOT tree into the shape that is
    needed to feed them to keras
    """
    out = []
    for i in range(data.shape[0]):
        out.append(np.array([a.tolist() for a in data[i]]))

    return np.array(out)


def saveToFile(fname, data):
    """
    Simply saves data to fname.

    Arguments:
        fname: filename (directory is taken from script's args).
        data: numpy array returned by convert_to_hdf5(), or parts hereof.

    Returns:
        Nothing.
    """
    log.info("  Saving to {}".format(args.outdir + fname))
    with h5py.File(args.outdir + fname, 'w') as hf:
        for var in data.dtype.names:

            # Images need to be treated slightly different
            if var in ['p1_em_calo0', 'p1_em_calo1', 'p1_em_calo2', 'p1_em_calo3', 'p1_em_calo4', 'p1_em_calo5', 'p1_em_calo6', 'p1_em_calo7',
                        'p1_h_calo0', 'p1_h_calo1', 'p1_h_calo2', 'p1_h_calo3', 'p1_h_calo4', 'p1_h_calo5', 'p1_h_calo6', 'p1_h_calo7',
                        'p2_em_calo0', 'p2_em_calo1', 'p2_em_calo2', 'p2_em_calo3', 'p2_em_calo4', 'p2_em_calo5', 'p2_em_calo6', 'p2_em_calo7',
                        'p2_h_calo0', 'p2_h_calo1', 'p2_h_calo2', 'p2_h_calo3', 'p2_h_calo4', 'p2_h_calo5', 'p2_h_calo6', 'p2_h_calo7']:
                array = imagesToArray(data[f'{var}'])
                hf.create_dataset( f'{var}', data=array, chunks=True, maxshape= (None, None) , compression='lzf' )
            # Do not save the variable 'BC_bunchIntensities' or cell information
            elif var == 'BC_bunchIntensities' or "p_cell_" in var:
                continue
            # Create the dataset and save the array
            else:
                hf.create_dataset( f'{var}', data=data[f'{var}'], chunks=True, maxshape= (None,) , compression='lzf' )



def appendToFile(fname, data):
    """
    Simply appends data to fname.

    Arguments:
        fname: filename (directory is taken from script's args).
        data: numpy array returned by convert_to_hdf5(), or parts hereof.

    Returns:
        Nothing.
    """
    log.info("  Appending to {}".format(args.outdir + fname))
    with h5py.File(args.outdir + fname, 'a') as hf:
        for var in data.dtype.names:
            print(var)

            array = data[f'{var}']
            # Images need to be treated slightly different
            if var in ['p1_em_calo0', 'p1_em_calo1', 'p1_em_calo2', 'p1_em_calo3', 'p1_em_calo4', 'p1_em_calo5', 'p1_em_calo6', 'p1_em_calo7',
                        'p1_h_calo0', 'p1_h_calo1', 'p1_h_calo2', 'p1_h_calo3', 'p1_h_calo4', 'p1_h_calo5', 'p1_h_calo6', 'p1_h_calo7',
                        'p2_em_calo0', 'p2_em_calo1', 'p2_em_calo2', 'p2_em_calo3', 'p2_em_calo4', 'p2_em_calo5', 'p2_em_calo6', 'p2_em_calo7',
                        'p2_h_calo0', 'p2_h_calo1', 'p2_h_calo2', 'p2_h_calo3', 'p2_h_calo4', 'p2_h_calo5', 'p2_h_calo6', 'p2_h_calo7']:
                array = imagesToArray(data[f'{var}'])

            # Do not save the variable 'BC_bunchIntensities' or cell information
            if var == 'BC_bunchIntensities' or "p_cell_" in var:
                continue
            # Resize the existing dataset and save the array at the end
            else:
                hf[f'{var}'].resize((hf[f'{var}'].shape[0] + array.shape[0]), axis = 0)
                hf[f'{var}'][-array.shape[0]:] = array



#============================================================================
# Main
#============================================================================

# Make a pool of processes (this must come after the functions needed to run over since it apparently imports __main__ here)
pool = multiprocessing.Pool(processes=args.max_processes)

for path in args.paths:
    # Count which file we have made it to
    counter += 1

    # Stop if we've reached the maximum number of files
    if args.max_input_files > 0 and counter >= args.max_input_files:
        break

    # Suppress warnings like these when loading the files: TClass::Init:0: RuntimeWarning: no dictionary for class [bla] is available
    ROOT.gErrorIgnoreLevel = ROOT.kError

    # Read ROOT data from file
    log.info("Read data from file: {}".format(path))
    f = ROOT.TFile(path, 'READ')

    tree = f.Get(args.tree)

    if tree.GetEntries() == 0: continue

    ROOT.gErrorIgnoreLevel = ROOT.kInfo

    # Set the number maximum number of entries to get to args.stop, but only if it's higher than the actual amount of events
    N = min(args.stop, tree.GetEntries())

    print("N = ",N)

    # Split indices into equally-sized batches
    index_edges = list(map(int, np.linspace(0, N, args.max_processes + 1, endpoint=True)))
    index_ranges = zip(index_edges[:-1], index_edges[1:])

    # Start conversion process(es) of ROOT data into numpy arrays and save them
    results = pool.map(converter, [(i, counter, path, start, stop) for i, (start, stop) in enumerate(index_ranges)])

sec = timedelta(seconds=time() - t_start)
log.info(f"Program finished. Time spent: {str(sec)}")
