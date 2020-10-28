import h5py
import numpy as np
import logging as log
import argparse
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy import interpolate


from utils import mkdir
from itertools import combinations
from skhep.math import vectors
import multiprocessing

import pandas as pd


from scipy.special import logit

data = pd.read_pickle("/Users/sda/hep/work/Data model/Z model/output/ZModels/230920/train_data.pkl")
dataMC = pd.read_pickle("/Users/sda/hep/work/Zmm model/Z_model/output/ZModels/250920/train_data.pkl")


data["predLGBM"] = logit(data["predLGBM"])
dataMC["predLGBM"] = logit(dataMC["predLGBM"])

dataSS = data[data["type"] != 1]
dataOS = data[data["type"] == 1]

# ------------------------- #
#        FOR Data           #
# ------------------------- #
# nSig = 41231 * (1-0.9333)#74747 * (1-0.4427)#41054
# nBkg = 41231 - nSig
#
# nSigTight = 2745 * (1-0.122)#29964
# nBkgTight = 2745 - nSigTight
# nSigMedium = 3176 * (1-0.148) #44777 * (1-0.0553)# 34529
# nBkgMedium = 3176 - nSigMedium
# nSigLoose = 3347 * (1-0.175)#45097 * (1-0.0699)#35548
# nBkgLoose = 3347 - nSigLoose
# # # ### Test Tight cut
# nSigMLTight = 2607 * (1-0.07)
# nBkgMLTight = 2607 - nSigMLTight
# # # ### Test Medium cut
# nSigMLMedium = 3325*(1-0.183)
# nBkgMLMedium = 3325-nSigMLMedium # cut is -0.706
# # # ### Test Loose cut
# nSigMLLoose = 3315*(1-0.168) # cut is -0.51
# nBkgMLLoose = 3315-nSigMLLoose

# ------------------------- #
#        FOR Data OS        #
# ------------------------- #
nSig = 22276 * (1-0.853)#74747 * (1-0.4427)#41054
nBkg = 22276 * 0.853 #pm 149, pm 0.012
nBkg_err = (149/22276 + 0.012/0.853)*nBkg

# nSig = 3458 * (1-0.129)#74747 * (1-0.4427)#41054
# nBkg = 3458 - nSig
#


nSigTight = 2678 * (1-0.07)#29964
nBkgTight = 2678 * 0.07 #pm 52, pm 0.015
nBkgTight_err = (52/2678 + 0.015/0.07)*nBkgTight

nSigMedium = 3082 * (1-0.073) #44777 * (1-0.0553)# 34529
nBkgMedium = 3082 * 0.073 # pm 56, pm 0.015
nBkgMedium_err = (56/3082 + 0.015/0.073)*nBkgMedium

nSigLoose = 3205 * (1-0.082)#45097 * (1-0.0699)#35548
nBkgLoose = 3205 * 0.082 # pm 57, pm 0.017
nBkgLoose_err = (57/3205 + 0.017/0.082)*nBkgLoose

# # # ### Test Tight cut
nSigMLTight = 2544 * (1-0.033)
nBkgMLTight = 2544 * 0.033 #pm 50, pm 0.016
nBkgMLTight_err = (50/2544 + 0.016/0.033)*nBkgMLTight

# # # ### Test Medium cut
nSigMLMedium = 2949*(1-0.04)
nBkgMLMedium = 2949 * 0.04 # pm 54, pm 0.015
nBkgMLMedium_err = (54/2949 + 0.015/0.04)*nBkgMLMedium

# # # ### Test Loose cut
nSigMLLoose = 3105*(1-0.061) # cut is -0.51
nBkgMLLoose = 3105 * 0.061 # pm 56, pm 0.015
nBkgMLLoose_err = (56/3105 + 0.015/0.061)*nBkgMLLoose


# ------------------------- #
#        FOR Data SS        #
# ------------------------- #
nSigSS = 0
nBkgSS = len(dataSS[(dataSS["invM"] > 60) & (dataSS["invM"] < 130)])
#
nBkgTightSS = len(dataSS[((dataSS["muo1_LHTight"] == 1) & (dataSS["muo2_LHTight"] == 1) & (dataSS["invM"] > 60) & (dataSS["invM"] < 130))])
nBkgMediumSS = len(dataSS[((dataSS["muo1_LHMedium"] == 1) & (dataSS["muo2_LHMedium"] == 1) & (dataSS["invM"] > 60) & (dataSS["invM"] < 130))])
nBkgLooseSS = len(dataSS[((dataSS["muo1_LHLoose"] == 1) & (dataSS["muo2_LHLoose"] == 1) & (dataSS["invM"] > 60) & (dataSS["invM"] < 130))])

# # # ### Test Tight cut
#nSigMLTightSS = 3 * (1-1)
nBkgMLTightSS = 3
# # # ### Test Medium cut
#nSigMLMediumSS = 19*(1-0.984)
nBkgMLMediumSS = 19 # cut is 4.88
# # # ### Test Loose cut
#nSigMLLooseSS = 45 * (1-0.906)
nBkgMLLooseSS = 45

fig, ax = plt.subplots(figsize=(6,5))
LHLooseSig, LHLooseBkg = nSigLoose/nSig, nBkgLoose/nBkg
LHLooseBkg_err =((nBkgLoose_err/nBkgLoose) + (nBkg_err/nBkg))*LHLooseBkg

LHMediumSig, LHMediumBkg = nSigMedium/nSig, nBkgMedium/nBkg
LHMediumBkg_err =((nBkgMedium_err/nBkgMedium) + (nBkg_err/nBkg))*LHMediumBkg

LHTightSig, LHTightBkg = nSigTight/nSig, nBkgTight/nBkg
LHTightBkg_err =((nBkgTight_err/nBkgTight) + (nBkg_err/nBkg))*LHTightBkg

ax.errorbar(LHLooseSig, LHLooseBkg, LHLooseBkg_err, fmt = 'o', color = 'b', capsize = 3)
ax.errorbar(LHMediumSig, LHMediumBkg,LHMediumBkg_err,  fmt = 'o', color = 'r', capsize = 3)
ax.errorbar(LHTightSig, LHTightBkg,LHTightBkg_err, fmt = 'o', color = 'g', capsize = 3)

MLLooseBkg = nBkgMLLoose/nBkg
MLLooseBkg_err =((nBkgMLLoose_err/nBkgMLLoose) + (nBkg_err/nBkg))*MLLooseBkg

MLMediumBkg = nBkgMLMedium/nBkg
MLMediumBkg_err =((nBkgMLMedium_err/nBkgMLMedium) + (nBkg_err/nBkg))*MLMediumBkg

MLTightBkg = nBkgMLTight/nBkg
MLTightBkg_err =((nBkgMLTight_err/nBkgMLTight) + (nBkg_err/nBkg))*MLTightBkg

ax.errorbar(LHLooseSig, MLLooseBkg, MLLooseBkg_err, fmt='*', color = 'b', capsize = 3)
ax.errorbar(LHMediumSig, MLMediumBkg,MLMediumBkg_err, fmt='*', color = 'r', capsize = 3)
ax.errorbar(LHTightSig, MLTightBkg, MLTightBkg_err, fmt='*', color = 'g', capsize = 3)

LHLooseBkgSS = nBkgLooseSS/nBkgSS
LHMediumBkgSS = nBkgMediumSS/nBkgSS
LHTightBkgSS = nBkgTightSS/nBkgSS
ax.scatter(LHLooseSig, LHLooseBkgSS, edgecolor = 'b', facecolors='none')
ax.scatter(LHMediumSig, LHMediumBkgSS, edgecolor = 'r', facecolors='none')
ax.scatter(LHTightSig, LHTightBkgSS, edgecolor = 'g', facecolors='none')

MLLooseBkgSS = nBkgMLLooseSS/nBkgSS
MLMediumBkgSS = nBkgMLMediumSS/nBkgSS
MLTightBkgSS = nBkgMLTightSS/nBkgSS
ax.scatter(LHLooseSig, MLLooseBkgSS, edgecolor = 'b', marker = "*", facecolors='none')
ax.scatter(LHMediumSig, MLMediumBkgSS, edgecolor = 'r',  marker = "*",facecolors='none')
ax.scatter(LHTightSig, MLTightBkgSS, edgecolor = 'g',  marker = "*", facecolors='none')

ax.set(xlim = (0.7,1), ylim = (0,0.04), xlabel = "Signal Efficiency", ylabel = "Background Efficiency")

#Making the legend in only one color
ax.plot([8], [8], 'ko', label = "Likelihood cuts, opposite-sign")
ax.plot([8], [8], 'k*', label = "ML cuts, opposite-sign")
ax.scatter([8], [8], edgecolor = 'k', marker = "o", facecolors='none', label = "Likelihood cuts, same-sign")
ax.scatter([8], [8], edgecolor = 'k', marker = "*", facecolors='none', label = "ML cuts, same-sign")
ax.legend(loc=2, prop={'size': 12}, frameon = False)

from matplotlib.lines import Line2D
from matplotlib.legend import Legend
custom_lines = [Line2D([0], [0], color='g', lw=4),
                Line2D([0], [0], color='r', lw=4),
                Line2D([0], [0], color='b', lw=4)]
labels = ["Tight", "Medium", "Loose"]
leg = Legend(ax, handles = custom_lines, labels = labels, frameon = False, prop={'size': 12})
ax.add_artist(leg);


divider = make_axes_locatable(ax)
ax2 = divider.append_axes("bottom", size="25%", pad=0)
ax.figure.add_axes(ax2)

ax2.plot(LHLooseSig, LHLooseBkg/MLLooseBkg, marker = 's', color = 'b')
ax2.text(LHLooseSig+0.008, (LHLooseBkg/MLLooseBkg)-0.3, f"{np.round(LHLooseBkg/MLLooseBkg,2)}", color = 'b')
ax2.scatter(LHLooseSig, (LHLooseBkgSS/MLLooseBkgSS), marker = 's', color = 'b', facecolor='none')
ax2.text(LHLooseSig+0.008, (LHLooseBkgSS/MLLooseBkgSS)-0.3, f"{np.round(LHLooseBkgSS/MLLooseBkgSS,2)}", color = 'b')

ax2.plot(LHMediumSig, LHMediumBkg/MLMediumBkg, marker = 's', color = 'r')
ax2.text(LHMediumSig+0.008, (LHMediumBkg/MLMediumBkg)-0.3, f"{np.round(LHMediumBkg/MLMediumBkg,2)}", color = 'r')
ax2.scatter(LHMediumSig, (LHMediumBkgSS/MLMediumBkgSS), marker = 's', color = 'r', facecolor='none')
ax2.text(LHMediumSig+0.008, (LHMediumBkgSS/MLMediumBkgSS)-0.3, f"{np.round(LHMediumBkgSS/MLMediumBkgSS,2)}", color = 'r')

ax2.plot(LHTightSig, LHTightBkg/MLTightBkg, marker = 's', color = 'g')
ax2.text(LHTightSig+0.008, (LHTightBkg/MLTightBkg)-0.3, f"{np.round(LHTightBkg/MLTightBkg,2)}", color = 'g')
ax2.scatter(LHTightSig, [5], marker = 's', color = 'g', facecolor='none') #LHTightBkgSS/MLTightBkgSS
ax2.arrow(LHTightSig-0.008, 4.6, 0, 1, head_width=0.005, head_length=0.3, fc='g', ec='g')
ax2.text(LHTightSig+0.008, 4.7, f"{np.round(LHTightBkgSS/MLTightBkgSS,2)}", color = 'g')

ax2.set(xlim = (0.7,1), ylim = (0,6.2), xlabel = "Signal Efficiency", ylabel = "Ratio LH/ML") #ylim = (-0.09,1)
ax2.yaxis.set_label_coords(-0.15,0.5)
ax2.set_yticks([0,2.5,5])

# ax2.set_yticklabels([0, 2.5, 5])
for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
             ax2.get_xticklabels() + ax2.get_yticklabels()):
    item.set_fontsize(14)
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(14)
ax.set_xticks([])
fig.tight_layout()
fig.savefig("ROCcutsData_Data.pdf")


# ------------------------- #
#        FOR ML OS          #
# ------------------------- #
nSig = 46547 * (1-0.0350)#74747 * (1-0.4427)#41054
nBkg = 46547 - nSig
# # nSig
nSigTight = 37250 * (1-0)#29964
nBkgTight = 37250 - nSigTight
# # #nSigMedium
nSigMedium = 43463 * (1-0.028) #44777 * (1-0.0553)# 34529
nBkgMedium = 43463 - nSigMedium
nSigLoose = 44725 * (1-0.0283)#45097 * (1-0.0699)#35548
nBkgLoose = 44725 - nSigLoose



# # ### Test Tight cut
nSigMLTight = 37897*(1-0.0220)
nBkgMLTight = 37897-nSigMLTight
# # ### Test Medium cut
nSigMLMedium = 43179*(1-0.0267)
nBkgMLMedium = 43179-nSigMLMedium
# # ### Test Loose cut
nSigMLLoose = 43979*(1-0.0269)
nBkgMLLoose = 43979-nSigMLLoose
#


# ------------------------- #
#        FOR ML SS          #
# ------------------------- #

dataMC_SS = dataMC[(dataMC["muo1_truthPdgId"]*dataMC["muo2_truthPdgId"] > 0)]
dataMC_SS_invMrange = dataMC_SS[(dataMC_SS["invM"] > 60) & (dataMC_SS["invM"] < 130) ]
dataMC_OS = dataMC[(dataMC["muo1_truthPdgId"]*dataMC["muo2_truthPdgId"] < 0)]

nSigSS = 0
nBkgSS = len(dataMC_SS_invMrange)
# # nSig
nBkgTightSS = len(dataMC_SS_invMrange[((dataMC_SS_invMrange["muo1_LHTight"] == 1) & (dataMC_SS_invMrange["muo2_LHTight"] == 1))])
# # #nSigMedium
nBkgMediumSS = len(dataMC_SS_invMrange[((dataMC_SS_invMrange["muo1_LHMedium"] == 1) & (dataMC_SS_invMrange["muo2_LHMedium"] == 1))])
# nSigLoose
nBkgLooseSS = len(dataMC_SS_invMrange[((dataMC_SS_invMrange["muo1_LHLoose"] == 1) & (dataMC_SS_invMrange["muo2_LHLoose"] == 1))])


# # ### Test Tight cut
# nSigMLTight = 37080*(1-0.0226)

nBkgMLTightSS = len(dataMC_SS_invMrange[dataMC_SS_invMrange["predLGBM"] > 10])
# # ### Test Medium cut
# nSigMLMedium = 43401*(1-0.0263)
nBkgMLMediumSS = len(dataMC_SS_invMrange[dataMC_SS_invMrange["predLGBM"] > 8.06])

# # ### Test Loose cut
# nSigMLLoose = 44595*(1-0.0273)
nBkgMLLooseSS = len(dataMC_SS_invMrange[dataMC_SS_invMrange["predLGBM"] > 7.43])
#

# ------------------------- #
#        Plotting           #
# ------------------------- #

fig, ax = plt.subplots(figsize=(6,5))
LHLooseSig, LHLooseBkg = nSigLoose/nSig, nBkgLoose/nBkg
LHMediumSig, LHMediumBkg = nSigMedium/nSig, nBkgMedium/nBkg
LHTightSig, LHTightBkg = nSigTight/nSig, nBkgTight/nBkg
ax.plot(LHLooseSig, LHLooseBkg, 'bo')
ax.plot(LHMediumSig, LHMediumBkg, 'ro')
ax.plot(LHTightSig, LHTightBkg, 'go')

MLLooseBkg = nBkgMLLoose/nBkg
MLMediumBkg = nBkgMLMedium/nBkg
MLTightBkg = nBkgMLTight/nBkg
ax.plot(LHLooseSig, MLLooseBkg, 'b*')
ax.plot(LHMediumSig, MLMediumBkg, 'r*')
ax.plot(LHTightSig, MLTightBkg, 'g*')

LHLooseBkgSS = nBkgLooseSS/nBkgSS
LHMediumBkgSS = nBkgMediumSS/nBkgSS
LHTightBkgSS = nBkgTightSS/nBkgSS
ax.scatter(LHLooseSig, LHLooseBkgSS, edgecolor = 'b', facecolors='none')
ax.scatter(LHMediumSig, LHMediumBkgSS, edgecolor = 'r', facecolors='none')
ax.scatter(LHTightSig, LHTightBkgSS, edgecolor = 'g', facecolors='none')

MLLooseBkgSS = nBkgMLLooseSS/nBkgSS
MLMediumBkgSS = nBkgMLMediumSS/nBkgSS
MLTightBkgSS = nBkgMLTightSS/nBkgSS
ax.scatter(LHLooseSig, MLLooseBkgSS, edgecolor = 'b', marker = "*", facecolors='none')
ax.scatter(LHMediumSig, MLMediumBkgSS, edgecolor = 'r',  marker = "*",facecolors='none')
ax.scatter(LHTightSig, MLTightBkgSS, edgecolor = 'g',  marker = "*", facecolors='none')

ax.set(xlim = (0.7,1), ylim = (-0.1,1.07), xlabel = "Signal Efficiency", ylabel = "Background Efficiency")

#Making the legend in only one color
ax.plot([8], [8], 'ko', label = "Likelihood cuts, opposite-sign")
ax.plot([8], [8], 'k*', label = "ML cuts, opposite-sign")
ax.scatter([8], [8], edgecolor = 'k', marker = "o", facecolors='none', label = "Likelihood cuts, same-sign")
ax.scatter([8], [8], edgecolor = 'k', marker = "*", facecolors='none', label = "ML cuts, same-sign")
ax.legend(loc=2, prop={'size': 12}, frameon = False)

from matplotlib.lines import Line2D
from matplotlib.legend import Legend
custom_lines = [Line2D([0], [0], color='g', lw=4),
                Line2D([0], [0], color='r', lw=4),
                Line2D([0], [0], color='b', lw=4)]
labels = ["Tight", "Medium", "Loose"]
leg = Legend(ax, handles = custom_lines, labels = labels, frameon = False, prop={'size': 12}, loc = 1)
ax.add_artist(leg);


divider = make_axes_locatable(ax)
ax2 = divider.append_axes("bottom", size="25%", pad=0)
ax.figure.add_axes(ax2)

ax2.plot(LHLooseSig, LHLooseBkg/MLLooseBkg, marker = 's', color = 'b')
# ax2.scatter(LHLooseSig, LHLooseBkgSS/MLLooseBkgSS, marker = 's', color = 'b', facecolor='none')

ax2.plot(LHMediumSig, LHMediumBkg/MLMediumBkg, marker = 's', color = 'r')
# ax2.scatter(LHMediumSig, LHMediumBkgSS/MLMediumBkgSS, marker = 's', color = 'r', facecolor='none')

ax2.plot(LHTightSig, LHTightBkg/MLTightBkg, marker = 's', color = 'g')
# ax2.scatter(LHTightSig, LHTightBkgSS/MLTightBkgSS, marker = 's', color = 'g', facecolor='none')

ax2.set(xlim = (0.7,1), xlabel = "Signal Efficiency", ylabel = "Ratio LH/ML") #ylim = (-0.09,1)
ax2.yaxis.set_label_coords(-0.1,0.5)

for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
             ax2.get_xticklabels() + ax2.get_yticklabels()):
    item.set_fontsize(14)
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(14)
ax.set_xticks([])
fig.tight_layout()
fig.savefig("ROCcutsData_MC.pdf")



# ------------------------- #
#        FOR ML OS Truth    #
# ------------------------- #
dataMC_OS = dataMC[(dataMC["muo1_truthPdgId"]*dataMC["muo2_truthPdgId"] < 0)]
dataMC_OS_invMrange = dataMC_OS[(dataMC_OS["invM"] > 60) & (dataMC_OS["invM"] < 130) ]

nSig = len(dataMC_OS_invMrange[dataMC_OS_invMrange["label"] == 1])
nBkg = len(dataMC_OS_invMrange[dataMC_OS_invMrange["label"] == 0])
nBkg
# # nSig
nSigTight = len(dataMC_OS_invMrange[((dataMC_OS_invMrange["muo1_LHTight"] == 1) & (dataMC_OS_invMrange["muo2_LHTight"] == 1)) & (dataMC_OS_invMrange["label"] == 1)])
nSigTight
nBkgTight = len(dataMC_OS_invMrange[((dataMC_OS_invMrange["muo1_LHTight"] == 1) & (dataMC_OS_invMrange["muo2_LHTight"] == 1)) & (dataMC_OS_invMrange["label"] == 0)])
nBkgTight

# # #nSigMedium
nSigMedium = len(dataMC_OS_invMrange[((dataMC_OS_invMrange["muo1_LHMedium"] == 1) & (dataMC_OS_invMrange["muo2_LHMedium"] == 1)) & (dataMC_OS_invMrange["label"] == 1)])
nSigMedium
nBkgMedium = len(dataMC_OS_invMrange[((dataMC_OS_invMrange["muo1_LHMedium"] == 1) & (dataMC_OS_invMrange["muo2_LHMedium"] == 1)) & (dataMC_OS_invMrange["label"] == 0)])
nBkgMedium

nSigLoose = len(dataMC_OS_invMrange[((dataMC_OS_invMrange["muo1_LHLoose"] == 1) & (dataMC_OS_invMrange["muo2_LHLoose"] == 1)) & (dataMC_OS_invMrange["label"] == 1)])
nBkgLoose = len(dataMC_OS_invMrange[((dataMC_OS_invMrange["muo1_LHLoose"] == 1) & (dataMC_OS_invMrange["muo2_LHLoose"] == 1)) & (dataMC_OS_invMrange["label"] == 0)])
nBkgLoose



# # ### Test Tight cut
cuts = np.linspace(8,11,num=100)
for cut in cuts:
    nSigMLTight = len(dataMC_OS_invMrange[ (dataMC_OS_invMrange["predLGBM"] > cut) & (dataMC_OS_invMrange["label"] == 1)])
    sig_ratio = nSigMLTight/nSig
    if (sig_ratio < nSigTight/nSig + 0.005) & (sig_ratio > nSigTight/nSig - 0.005):
        nBkgMLTight = len(dataMC_OS_invMrange[(dataMC_OS_invMrange["predLGBM"] > cut) & (dataMC_OS_invMrange["label"] == 0)])
        print(f"breaking, cut at {cut} with ratio {sig_ratio}")
        break
# # ### Test Medium cut
cuts = np.linspace(7,9,num=100)
for cut in cuts:
    nSigMLMedium = len(dataMC_OS_invMrange[ (dataMC_OS_invMrange["predLGBM"] > cut) & (dataMC_OS_invMrange["label"] == 1)])
    sig_ratio = nSigMLMedium/nSig
    if (sig_ratio < nSigMedium/nSig + 0.005) & (sig_ratio > nSigMedium/nSig - 0.005):
        nBkgMLMedium = len(dataMC_OS_invMrange[(dataMC_OS_invMrange["predLGBM"] > cut) & (dataMC_OS_invMrange["label"] == 0)])
        print(f"breaking, cut at {cut} with ratio {sig_ratio}")
        break
# # ### Test Loose cut
cuts = np.linspace(6,8,num=100)
for cut in cuts:
    nSigMLLoose = len(dataMC_OS_invMrange[ (dataMC_OS_invMrange["predLGBM"] > cut) & (dataMC_OS_invMrange["label"] == 1)])
    sig_ratio = nSigMLLoose/nSig
    if (sig_ratio < nSigLoose/nSig + 0.005) & (sig_ratio > nSigLoose/nSig - 0.005):
        nBkgMLLoose = len(dataMC_OS_invMrange[(dataMC_OS_invMrange["predLGBM"] > cut) & (dataMC_OS_invMrange["label"] == 0)])
        print(f"breaking, cut at {cut} with ratio {sig_ratio}")
        break
len(dataMC_OS_invMrange[ (dataMC_OS_invMrange["predLGBM"] > 3) & (dataMC_OS_invMrange["label"] == 1)])
len(dataMC_OS_invMrange[ (dataMC_OS_invMrange["predLGBM"] > -10) & (dataMC_OS_invMrange["label"] == 0)])
nSig

# ------------------------- #
#        FOR ML SS          #
# ------------------------- #

dataMC_SS = dataMC[(dataMC["muo1_truthPdgId"]*dataMC["muo2_truthPdgId"] > 0)]
dataMC_SS_invMrange = dataMC_SS[(dataMC_SS["invM"] > 60) & (dataMC_SS["invM"] < 130) ]

nSigSS = 0
nBkgSS = len(dataMC_SS_invMrange)
# # nSig
nBkgTightSS = len(dataMC_SS_invMrange[((dataMC_SS_invMrange["muo1_LHTight"] == 1) & (dataMC_SS_invMrange["muo2_LHTight"] == 1))])
# # #nSigMedium
nBkgMediumSS = len(dataMC_SS_invMrange[((dataMC_SS_invMrange["muo1_LHMedium"] == 1) & (dataMC_SS_invMrange["muo2_LHMedium"] == 1))])
# nSigLoose
nBkgLooseSS = len(dataMC_SS_invMrange[((dataMC_SS_invMrange["muo1_LHLoose"] == 1) & (dataMC_SS_invMrange["muo2_LHLoose"] == 1))])

# # ### Test Tight cut
# nSigMLTight = 37080*(1-0.0226)

nBkgMLTightSS = len(dataMC_SS_invMrange[dataMC_SS_invMrange["predLGBM"] > 10])
# # ### Test Medium cut
# nSigMLMedium = 43401*(1-0.0263)
nBkgMLMediumSS = len(dataMC_SS_invMrange[dataMC_SS_invMrange["predLGBM"] > 8.06])

# # ### Test Loose cut
# nSigMLLoose = 44595*(1-0.0273)
nBkgMLLooseSS = len(dataMC_SS_invMrange[dataMC_SS_invMrange["predLGBM"] > 7.43])
#

# ------------------------- #
#        Plotting           #
# ------------------------- #

fig, ax = plt.subplots(figsize=(6,5))
LHLooseSig, LHLooseBkg = nSigLoose/nSig, nBkgLoose/nBkg
LHMediumSig, LHMediumBkg = nSigMedium/nSig, nBkgMedium/nBkg
LHTightSig, LHTightBkg = nSigTight/nSig, nBkgTight/nBkg
ax.plot(LHLooseSig, LHLooseBkg, 'bo')
ax.plot(LHMediumSig, LHMediumBkg, 'ro')
ax.plot(LHTightSig, LHTightBkg, 'go')

MLLooseBkg = nBkgMLLoose/nBkg
MLMediumBkg = nBkgMLMedium/nBkg
MLTightBkg = nBkgMLTight/nBkg
ax.plot(LHLooseSig, MLLooseBkg, 'b*')
ax.plot(LHMediumSig, MLMediumBkg, 'r*')
ax.plot(LHTightSig, MLTightBkg, 'g*')

LHLooseBkgSS = nBkgLooseSS/nBkgSS
LHMediumBkgSS = nBkgMediumSS/nBkgSS
LHTightBkgSS = nBkgTightSS/nBkgSS
ax.scatter(LHLooseSig, LHLooseBkgSS, edgecolor = 'b', facecolors='none')
ax.scatter(LHMediumSig, LHMediumBkgSS, edgecolor = 'r', facecolors='none')
ax.scatter(LHTightSig, LHTightBkgSS, edgecolor = 'g', facecolors='none')

MLLooseBkgSS = nBkgMLLooseSS/nBkgSS
MLMediumBkgSS = nBkgMLMediumSS/nBkgSS
MLTightBkgSS = nBkgMLTightSS/nBkgSS
ax.scatter(LHLooseSig, MLLooseBkgSS, edgecolor = 'b', marker = "*", facecolors='none')
ax.scatter(LHMediumSig, MLMediumBkgSS, edgecolor = 'r',  marker = "*",facecolors='none')
ax.scatter(LHTightSig, MLTightBkgSS, edgecolor = 'g',  marker = "*", facecolors='none')

ax.set(xlim = (0.7,1), ylim = (-0.1,1.07), xlabel = "Signal Efficiency", ylabel = "Background Efficiency")

#Making the legend in only one color
ax.plot([8], [8], 'ko', label = "Likelihood cuts, opposite-sign")
ax.plot([8], [8], 'k*', label = "ML cuts, opposite-sign")
ax.scatter([8], [8], edgecolor = 'k', marker = "o", facecolors='none', label = "Likelihood cuts, same-sign")
ax.scatter([8], [8], edgecolor = 'k', marker = "*", facecolors='none', label = "ML cuts, same-sign")
ax.legend(loc=2, prop={'size': 12}, frameon = False)

from matplotlib.lines import Line2D
from matplotlib.legend import Legend
custom_lines = [Line2D([0], [0], color='g', lw=4),
                Line2D([0], [0], color='r', lw=4),
                Line2D([0], [0], color='b', lw=4)]
labels = ["Tight", "Medium", "Loose"]
leg = Legend(ax, handles = custom_lines, labels = labels, frameon = False, prop={'size': 12}, loc = 1)
ax.add_artist(leg);


divider = make_axes_locatable(ax)
ax2 = divider.append_axes("bottom", size="25%", pad=0)
ax.figure.add_axes(ax2)

# ax2.plot(LHLooseSig, LHLooseBkg/MLLooseBkg, marker = 's', color = 'b')
# ax2.scatter(LHLooseSig, LHLooseBkgSS/MLLooseBkgSS, marker = 's', color = 'b', facecolor='none')

# ax2.plot(LHMediumSig, LHMediumBkg/MLMediumBkg, marker = 's', color = 'r')
# ax2.scatter(LHMediumSig, LHMediumBkgSS/MLMediumBkgSS, marker = 's', color = 'r', facecolor='none')

# ax2.plot(LHTightSig, LHTightBkg/MLTightBkg, marker = 's', color = 'g')
# ax2.scatter(LHTightSig, LHTightBkgSS/MLTightBkgSS, marker = 's', color = 'g', facecolor='none')

ax2.set(xlim = (0.7,1), xlabel = "Signal Efficiency", ylabel = "Ratio LH/ML") #ylim = (-0.09,1)
ax2.yaxis.set_label_coords(-0.1,0.5)

for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
             ax2.get_xticklabels() + ax2.get_yticklabels()):
    item.set_fontsize(14)
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(14)
ax.set_xticks([])
fig.tight_layout()
fig.savefig("ROCcutsData_MCtruth.pdf")



fig, ax = plt.subplots(figsize=(5,5))
ax.hist(dataOS["predLGBM"], bins = 70, color = "tab:purple", histtype = "step", label = "Opposite-sign")
ax.hist(dataSS["predLGBM"], bins = 70, color = "g", histtype = "step", label = "Same-sign")
ax.set(xlabel = "LGBM score", ylabel = "Frequency", xlim = (-50,20))
ax.legend(prop={'size': 15})
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)
fig.tight_layout()
fig.savefig("LGBMscoreCutData.pdf")

dataMC_SS = dataMC[(dataMC["muo1_truthPdgId"]*dataMC["muo2_truthPdgId"] > 0)]
dataMC_OS = dataMC[(dataMC["muo1_truthPdgId"]*dataMC["muo2_truthPdgId"] < 0)]

fig, ax = plt.subplots(figsize=(5,5))
ax.hist(dataMC_OS["predLGBM"], color = 'r', bins = 50, histtype = "step", label = "Opposite-sign")
ax.hist(dataMC_SS["predLGBM"], color = 'b', bins = 50, histtype = "step", label = "Same-sign")
ax.set(xlabel = "LGBM score", ylabel = "Frequency")
ax.legend(prop={'size': 15})
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)
fig.tight_layout()
fig.savefig("LGBMscoreCutMC.pdf")
