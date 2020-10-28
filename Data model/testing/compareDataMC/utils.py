# -*- coding: utf-8 -*-

"""
Common utility methods for egamma project.
"""

import os
import time
import logging as log
from subprocess import call
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches




def mkdir (path):
    """
    Script to ensure that the directory at `path` exists.

    Arguments:
        path: String specifying path to directory to be created.
    """

    # Check mether  output directory exists
    if not os.path.exists(path):
        # print( "mdkir: Creating output directory:\n  {}".format(path) )
        try:
            os.makedirs(path)
        except OSError:
            # Apparently, `path` already exists.
            pass
        pass

    return



def cautious_remove (path):
    """
    ...
    """
    if path.startswith('/') or '*' in path:
        log.info("cautious_remove: Refusing to remove {}".format(path))
    else:
        log.debug("cautious_remove: Removing.")
        call(['rm', path])
        pass
    pass



def unique_tmp (path):
    """
    Utility script to create a unique, temporary file path.
    """
    ID = int(time.time() * 1E+06)
    basedir = '/'.join(path.split('/')[:-1])
    filename = path.split('/')[-1]
    filename = 'tmp.{:s}.{:d}'.format(filename, ID)
    return '{}/{}'.format(basedir, filename)


def Histogram(data, signal, weights, bins, rangemin, rangemax):
    counts_sig, edges_sig = np.histogram(data[signal>0.5], bins=bins, range=(rangemin, rangemax))
    counts_bkg, edges_bkg = np.histogram(data[signal<0.5], bins=bins, range=(rangemin, rangemax))
    counts_bkgrw, edges_bkgrw = np.histogram(data[signal<0.5], bins=bins, weights = weights[signal < 0.5], range=(rangemin, rangemax))

    return counts_sig, edges_sig, counts_bkg, edges_bkg, counts_bkgrw, edges_bkgrw

def Plot(input, fig, ax, xlabel, includeN = False, legend = True):
    counts_sig, edges_sig, counts_bkg, edges_bkg, counts_bkgrw, edges_bkgrw = input[0], input[1], input[2], input[3], input[4], input[5]
    bw = edges_sig[1] - edges_sig[0]
    if legend:
        ax.step(x=edges_sig, y=np.append(counts_sig, 0), where="post", color = "k", alpha = 1, label = "Signal");
        ax.step(x=edges_bkg, y=np.append(counts_bkg, 0), where="post", color = "b", alpha = 1, label = "Background");
        ax.step(x=edges_bkgrw, y=np.append(counts_bkgrw, 0), where="post", color = "r", linestyle = 'dashed', alpha = 1, label = "Background reweighted");
    else:
        ax.step(x=edges_sig, y=np.append(counts_sig, 0), where="post", color = "k", alpha = 1);
        ax.step(x=edges_bkg, y=np.append(counts_bkg, 0), where="post", color = "b", alpha = 1);
        ax.step(x=edges_bkgrw, y=np.append(counts_bkgrw, 0), where="post", color = "r", linestyle = 'dashed', alpha = 1);
    ax.set(xlim = (edges_sig[0], edges_sig[-1]), xlabel = xlabel, ylabel = f"Events/{bw:4.2f}");
    if includeN:
        handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white",
                                  lw=0, alpha=0)] * 5
        labels = []
        labels.append("Total samples sig: {0:.0f}".format(np.sum(counts_sig)))
        labels.append("Total samples bkg: {0:.0f}".format(np.sum(counts_bkg)))
        legend0 = ax.legend(handles, labels, loc='best', fontsize='small',
        fancybox=True, framealpha=0.5,
        handlelength=0, handletextpad=0)
        if legend:
            ax.legend(loc = 9, fontsize='small')
        ax.add_artist(legend0)
    if legend and not includeN:
        ax.legend()
    return fig, ax
