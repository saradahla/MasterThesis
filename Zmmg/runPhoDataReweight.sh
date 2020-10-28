#!/bin/bash
python -u gamma_Dataset.py --tag 20201028 ../hdf5Prod/output/root2hdf5/Zmmgam20201022/Zmmgam20201022_0000.h5 ../hdf5Prod/output/root2hdf5/Zmmgam20201022/Zmmgam20201022_0001.h5 ../hdf5Prod/output/root2hdf5/Zmmgam20201022/Zmmgam20201022_0002.h5 ../hdf5Prod/output/root2hdf5/Zmmgam20201022/Zmmgam20201022_0003.h5 2>&1 &> output/logPhoDataset.txt
python -u phoReweight.py --tag 20201028  output/pho_Dataset/20201028/20201028.h5 2>&1 &> output/logPhoReweight.txt
