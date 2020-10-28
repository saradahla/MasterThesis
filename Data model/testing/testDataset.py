
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
import pandas as pd

import lightgbm as lgb
from scipy.special import logit
import klib
import seaborn as sns

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
    score = gbm.predict(data[training_var], n_jobs=1)
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

    score = gbm.predict(data[training_var], n_jobs=1)
    return logit(score)

def GetPIDscore2(gbm, data):
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

    score = gbm.predict(data[training_var], n_jobs=1)
    return logit(score)



hf = h5ToDf("/Users/sda/hep/work/Data model/output/MuoSingleHdf5/010920_3/010920_3.h5")

modelISO = "/Users/sda/hep/work/Zmm model/PID_ISO_models/output/ISOModels/110820_ZbbW/lgbmISO.txt"
modelPID = "/Users/sda/hep/work/Zmm model/PID_ISO_models/output/PIDModels/010920_ZbbW/lgbmPID.txt"
modelPID2 = "/Users/sda/hep/work/Zmm model/PID_ISO_models/output/PIDModels/010920_ZbbW_only6/lgbmPID.txt"

PIDmod = lgb.Booster(model_file = modelPID)
PIDmod2 = lgb.Booster(model_file = modelPID2)
ISOmod = lgb.Booster(model_file = modelISO)
hf['muo_ISO_score'] = GetISOscore(ISOmod,hf)
hf['muo_PID_score'] = GetPIDscore(PIDmod,hf)
hf['muo_PID2_score'] = GetPIDscore2(PIDmod2,hf)

hf['muo_ISO_score'] = hf['muo_ISO_score'][(hf['muo_ISO_score'] < 4) & (hf['muo_ISO_score'] > -4) ]
hf['muo_PID_score'] = hf['muo_PID_score'][(hf['muo_PID_score'] < 20) & (hf['muo_PID_score'] > -20) ]
hf['muo_PID2_score'] = hf['muo_PID2_score'][(hf['muo_PID2_score'] < 20) & (hf['muo_PID2_score'] > -20) ]

import scipy.stats
type = hf['Type']

from matplotlib.ticker import NullFormatter, MaxNLocator

xlims = [ -20, 20]
ylims = [ -4, 4]

# Define the locations for the axes
left, width = 0.05, 0.5
bottom, height = 0.05, 0.5
bottom_h = left_h = left+width+0.02

# Set up the geometry of the three plots
rect_temperature = [left, bottom, width, height] # dimensions of temp plot
rect_histx = [left, bottom_h, width, 0.1] # dimensions of x-histogram
rect_histy = [left_h, bottom, 0.1, height] # dimensions of y-histogram

# Set up the size of the figure
fig = plt.figure(1, figsize=(9.5,9))

# Make the three plots
axTemperature = plt.axes(rect_temperature) # temperature plot
axHistx = plt.axes(rect_histx) # x histogram
axHisty = plt.axes(rect_histy) # y histogram

# Remove the inner axes numbers of the histograms
nullfmt = NullFormatter()
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)


# Find the min/max of the data
xmin = min(xlims)
xmax = max(xlims)
ymin = min(ylims)
ymax = max(ylims)

# Make the 'main' temperature plot
# Define the number of bins
nxbins = 50
nybins = 50
nbins = 100

xbins = np.linspace(start = xmin, stop = xmax, num = nxbins)
ybins = np.linspace(start = ymin, stop = ymax, num = nybins)
xcenter = (xbins[0:-1]+xbins[1:])/2.0
ycenter = (ybins[0:-1]+ybins[1:])/2.0
aspectratio = 1.0*(xmax - 0)/(1.0*ymax - 0)

H, xedges,yedges = np.histogram2d(hf['muo_ISO_score'],hf['muo_PID_score'],bins=(ybins,xbins))
X = xcenter
Y = ycenter
Z = H

# Plot the temperature data
cax = (axTemperature.imshow(H, extent=[xmin, xmax, ymin, ymax],# vmin = -5, vmax = 50,
       interpolation='nearest', origin='lower',aspect=aspectratio))#, cmap = plt.get_cmap('jet_r')))

#Set up the plot limits
axTemperature.set_xlim(xlims)
axTemperature.set_ylim(ylims)

#Set up the histogram bins
xbins = np.arange(xmin, xmax, (xmax-xmin)/nbins)
ybins = np.arange(ymin, ymax, (ymax-ymin)/nbins)

#Plot the histograms
axHistx.hist(hf['muo_PID_score'][type == 0], bins=xbins, color = 'blue')
axHistx.hist(hf['muo_PID_score'][type == 1], bins=xbins, color = 'red')
axHisty.hist(hf['muo_ISO_score'][type == 0], bins=ybins, orientation='horizontal', color = 'blue')
axHisty.hist(hf['muo_ISO_score'][type == 1], bins=ybins, orientation='horizontal', color = 'red')

#Set up the histogram limits
axHistx.set_xlim( -20, 20 )
axHisty.set_ylim( -4, 4 )

#Show the plot
plt.draw()

cmap = sns.cubehelix_palette(light=1, as_cmap=True)

sns.jointplot(x=hf['muo_PID_score'][type == 0], y=hf['muo_ISO_score'][type == 0], cmap=plt.cm.viridis, xlim = (-20,20), ylim = (-4,4), kind ='kde')
sns.jointplot(x=hf['muo_PID_score'][type == 1], y=hf['muo_ISO_score'][type == 1], cmap=plt.cm.viridis, xlim = (-20,20), ylim = (-4,4), kind ='kde')

#subplots migration
f = plt.figure()
for J in [bkg, sig]:
    for A in J.fig.axes:
        f._axstack.add(f._make_key(A), A)


plt.hist(hf['muo_PID_score'][type==0], range = (-30,30), bins = 100, histtype = "step");
plt.hist(hf['muo_PID_score'][type==1], range = (-30,30), bins = 100, histtype = "step");


plt.hist(hf['muo_PID2_score'][type==0], range = (-20,20), bins = 100, histtype = "step");
plt.hist(hf['muo_PID2_score'][type==1], range = (-20,20), bins = 100, histtype = "step");

fig, ax = plt.subplots(figsize=(5,5))
ax.set_title("For 11 variables")
ax.plot(hf['muo_PID_score'][type==0],hf['muo_ISO_score'][type==0],'.', alpha = 0.5)#, bins = 50, cmax = 30);
ax.plot(hf['muo_PID_score'][type==1],hf['muo_ISO_score'][type==1],'.', alpha = 0.5)#, bins = 50, cmax = 30);
fig.savefig("11vars_PID.pdf")

fig, ax = plt.subplots(figsize=(5,5))
ax.set_title("For 8 variables (ATLAS)")
plt.plot(hf['muo_PID2_score'][type==0],hf['muo_ISO_score'][type==0],'.', alpha = 0.5)#, bins = 50, cmax = 30);
plt.plot(hf['muo_PID2_score'][type==1],hf['muo_ISO_score'][type==1],'.', alpha = 0.5)#, bins = 50, cmax = 30);
fig.savefig("8vars_PID.pdf")



nas = np.logical_or(np.isnan(ISO_prime), np.isnan(hf['muo_PID_score']))
infs = np.logical_or(np.isinf(ISO_prime), np.isinf(hf['muo_PID_score']))

scipy.stats.pearsonr(ISO_prime[~nas & ~infs], hf['muo_PID_score'][~nas & ~infs])[0]


# fig, ax = plt.subplots(figsize=(7,5))
#
# n = len(nType)
# x = np.arange(n)
# val = nType
#
# ax.bar(x, height=val, align = 'center')
#
# shift = np.max(val)*0.01
# for i in range(n):
#     ax.text(x[i], val[i]+shift, f"{int(val[i])}", color = 'black', ha = 'center', va = 'bottom')
#
# plt.xticks(range(n));
# labels = [item.get_text() for item in ax.get_xticklabels()]
# labels[0] = 'Background'
# labels[1] = 'Signal'
# labels[2] = 'Trash'
# ax.set_xticklabels(labels)
# ax.set_ylabel("Frequency")
#
# fig.tight_layout()
# fig.show()
# fig.savefig("DataTypes.pdf")
#
#
