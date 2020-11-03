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

from peakfit import PeakFit_likelihood


data = pd.read_pickle("/groups/hep/sda/work/MastersThesis/Zmmg/output/mumugamPredictions/20201030_ZgamData/pred_data.pkl")

data["predLGBM"] = logit(data["predLGBM"])
# residues = []
# nsigs = []
# nbkgs = []
# ntotals = []
SigEff = []
BkgEff = []

cut_vals = []
bkgRatio_ATLAS = 0.006197696374022418

# cuts = np.linspace(6,8, num = 50)
#cuts = np.linspace(-30,-22, num = 10)
cuts = np.linspace(2,3, num = 60)
#cuts = [7.09]
# cuts = [-40]
i = 1
j = 1
#cutval = [2.18]
for cutval in cuts[::-1]:
# cutval = -50
# for x in [1]:
    # Likelihood_cut = data["isATLAS"]
    Likelihood_cut = (data["predLGBM"] > cutval)
    BL_sig, BL_bkg, sig_ratio, bkg_ratio = PeakFit_likelihood(Likelihood_cut*1, data["invM"], cutval, plots = True, constant_mean = True,
                                           constant_width = True, classifier_name = 'Likelihood', CB = True, Gauss = False, bkg_comb = False,
                                           bkg_exp = True, bkg_cheb = False);
    print(f"The bkg_ratio is {bkg_ratio}")
    if BL_sig != 0:
        SigEff.append(sig_ratio)
        BkgEff.append(bkg_ratio)

        if (bkg_ratio < bkgRatio_ATLAS + 0.0005) & (bkg_ratio > bkgRatio_ATLAS - 0.0005):
            print(f"Found correct background efficiency for a cut at {bkg_ratio} compared to true {bkgRatio_ATLAS}")
            print(f"My cut is {cutval}")
            print(f"Breaking....")

            break
print("Signal Efficiencies:")
print(SigEff)
print("Background Efficiencies:")
print(BkgEff)


#SigEff, BkgEff = np.array(sigEffuse),np.array(bkgEffuse)
#
#
# def calcAuc(x,y, xlast, ylast):
#     return (x-xlast) * (y-ylast)
#
# for i in range(len(SigEff)):
#     auc_calc = 0
#     if i == 0:
#         auc_calc += calcAuc(SigEff[i], BkgEff[i], 0, 0)
#     else:
#         auc_calc += calcAuc(SigEff[i], BkgEff[i], SigEff[i-1], BkgEff[i-1])
#
# print(f"AUC was {1-auc_calc}")
# np.save("SigEff.npy", SigEff)
# np.save("BkgEff.npy", BkgEff)
# #
# fig, ax = plt.subplots(figsize=(7,5))
# ## SMothing the curve for auc
# # x_new = np.linspace(min(np.array(SigEff)),max(np.array(SigEff)), 50)
# # #0.96763063
# # SigEffSpline = np.array(SigEff)
# # for i,s in enumerate(SigEffSpline):
# #     if s > 0.96763063:
# #         SigEffSpline[i] = 0.96763063
# # tck, u = interpolate.splprep([SigEff[::-1], BkgEff[::-1]], s=0)
# # new_points = interpolate.splev(u, tck)
# # ax.plot(new_points[0], new_points[1], color = 'c', label = f"smooth curve")
# # print("could not make spline")
# ax.plot(SigEff,BkgEff, color = 'k', label = f"LGBM cut, auc = {1-auc_calc}")
# ax.plot(np.array(SigLoose), np.array(BkgLoose), color = 'r', label = f"LH Loose, auc = {auc(np.array(BkgLoose), np.array(SigLoose))}")
# ax.plot(np.array(SigMedium), np.array(BkgMedium), color = 'g', label = f"LH Medium, auc = {auc(np.array(BkgMedium), np.array(SigMedium))}")
# ax.plot(np.array(SigTight), np.array(BkgTight), color = 'tab:blue', label = f"LH Tight, auc = {auc(np.array(BkgTight), np.array(SigTight))}")
#
# if MC:
#     fpr_train, tpr_train, thresholds_train = roc_curve(data["label"], data["predLGBM"], sample_weight=data["regWeight_nEst10"])
#     auc_train = auc(fpr_train, tpr_train)
#     ax.plot(tpr_train, fpr_train, color = 'tab:purple', label=f"True LGBM roc curve, auc = {auc_train}")
# ax.set(xlabel = "Signal ratio", ylabel = "Background ratio")
# ax.legend()
# fig.tight_layout()
# fig.savefig('./output/figuresRooFit/figuresRooFit_BWxCB_combined_MC/Ratios.pdf')

    #max_residue, ntotal, nsig, nbkg
#     if max_residue < 4:
#         #print("I am saving this one\n")
#         #print(f"The residue was {max_residue} and the fraction of signal was {nsig/ntotal}\n")
#         residues.append(max_residue)
#         nsigs.append(nsig)
#         nbkgs.append(nbkg)
#         ntotals.append(ntotal)
#         SigEff.append(nsig/ntotal)
#         BkgEff.append(nbkg/ntotal)
#         cut_vals.append(cutval)
#
#
#
# fig, ax = plt.subplots(figsize=(7,5))
# ax.plot(SigEff, BkgEff, 'k.')
# ax.set(xlabel = "Signal Efficiency", ylabel = "Background Efficiency")
# fig.tight_layout()
# fig.savefig('./output/figuresRooFit/ROC_curve.pdf')


#
# fig = plt.figure(figsize=(12,10))
#
# ax0 = fig.add_subplot(221)
# ax1 = fig.add_subplot(222)
# ax2 = fig.add_subplot(212)
# #ax = ax.flatten()
# ax0.plot(SigEff, nsigs, 'k.')
# ax0.set(xlabel = "Signal Efficiency (# signal / # total)", ylabel = "# signal")
# #ax2 = ax0.twinx()
# for ns, cut in zip(nsigs, cut_vals):
#     if np.round(cut,2) == 4.39:
#         ax0.axhline(ns, linestyle = 'dashed', color = 'r', alpha = 1)
#     ax0.axhline(ns, linestyle = 'dashed', color = 'r', alpha = 0.2)
#     ax0.text(ax0.get_xlim()[-1]+.0005, ns, np.round(cut,2), horizontalalignment='left',
#         verticalalignment='center', fontsize=8)#,
#         #transform=ax0.transAxes)
# ax0.text(1.106, 0.5, "Cuts", horizontalalignment='left',
#     verticalalignment='center',
#     rotation='vertical',
#     transform=ax0.transAxes, fontsize=15)
# for item in ([ax0.title, ax0.xaxis.label, ax0.yaxis.label] +
#              ax0.get_xticklabels() + ax0.get_yticklabels()):
#     item.set_fontsize(17)
#
# for se, ns in zip(SigEff, nsigs):
#     if (ns > 2000) & (ns < 2700):
#         ax1.plot(se, ns, 'k.')
# ax1.set(xlabel = "Signal Efficiency (# signal / # total)", ylabel = "# signal")
# #ax2 = ax1.twinx()
# for ns, cut in zip(nsigs, cut_vals):
#     if (ns > 2000) & (ns < 2700):
#         if np.round(cut,2) == 4.39:
#             ax1.axhline(ns, linestyle = 'dashed', color = 'r', alpha = 1)
#             ax1.text(ax1.get_xlim()[-1]+.0005, ns, np.round(cut,2), horizontalalignment='left',
#                 verticalalignment='center', fontsize=9)#,
#                 #transform=ax1.transAxes)
#         else:
#             ax1.axhline(ns, linestyle = 'dashed', color = 'r', alpha = 0.2)
#             ax1.text(ax1.get_xlim()[-1]+.0005, ns, np.round(cut,2), horizontalalignment='left',
#             verticalalignment='center', fontsize=9)#,
#             #transform=ax1.transAxes)
# ax1.text(1.106, 0.5, "Cuts", horizontalalignment='left',
#     verticalalignment='center',
#     rotation='vertical',
#     transform=ax1.transAxes, fontsize=15)
# for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
#              ax1.get_xticklabels() + ax1.get_yticklabels()):
#     item.set_fontsize(17)
#
#
# ax2.hist(data["predLGBM"], bins = 100, histtype = "step", range = (-50,20));
# for cut in cut_vals:
#     if np.round(cut,2) == 4.39:
#         ax2.axvline(cut, linestyle = 'dashed', color = 'r', alpha = 1)
#     ax2.axvline(cut, linestyle = 'dashed', color = 'r', alpha = 0.2)
#
# ax2.set(xlabel = "LGBM score (logit transformed)", ylabel = "Frequency")
# for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
#              ax2.get_xticklabels() + ax2.get_yticklabels()):
#     item.set_fontsize(17)
# fig.tight_layout()
# fig.savefig('./output/figuresRooFit/nSignals.pdf')
