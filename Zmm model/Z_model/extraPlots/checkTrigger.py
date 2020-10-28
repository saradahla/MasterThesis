import numpy as np
import uproot as ur
import matplotlib.pyplot as plt

d = ur.open("/Users/sda/hep/storage/dataAnalysis/mc16_13TeV.361107.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Zmumu.deriv.DAOD_MUON1.e3601_e5984_s3126_r10201_r10210_p3629_wTrigger.root")
d.keys()
tree = d[b'analysis;477']
tree.keys()


pt = tree["muo_pt"].array().flatten()/1000
trigger = tree["muo_trigger"].array().flatten()
Ztrue = abs(tree["muo_truthPdgId"].array().flatten()) == 13

Ztrue


counts, edges = np.histogram(pt, bins = 80, range = (0, 80))
counts_Ztrue, edges_Ztrue = np.histogram(pt[Ztrue], bins = 80, range = (0, 80))
counts_trig, edges_trig = np.histogram(pt[trigger], bins = 80, range = (0, 80))

fig, ax = plt.subplots(figsize=(5,5))
ax.step(x=edges, y=np.append(counts, 0), where="post", color = 'k', label = "pt for all muons");
ax.step(x=edges_Ztrue, y=np.append(counts_Ztrue, 0), where="post", color = 'b', label = "pt for Z muons");
ax.step(x=edges_trig, y=np.append(counts_trig, 0), where="post", color = 'r', label = "pt for triggered muons");
ax.set(xlim = (edges[0], edges[-1]), xlabel = "pt", ylabel = "Frequency");
ax.legend()
fig.tight_layout()
fig.savefig("ptTrigger_wZtrue.pdf")

fig, ax = plt.subplots(figsize=(5,5))
ax.step(x=edges, y=np.append(counts_trig/counts, 0), where="post", color = 'k', label = "Ratio: pt for triggered muons / pt for all muons");
ax.step(x=edges, y=np.append(counts_trig/counts_Ztrue, 0), where="post", color = 'b', label = "Ratio: pt for triggered muons / pt for Z muons");
ax.set(xlim = (edges[0], edges[-1]), ylim = (-0.05, 1), xlabel = "pt", ylabel = "Ratio");
ax.legend()
fig.tight_layout()
fig.savefig("ratioTriggerPt_wZtrue.pdf")


eta = tree["muo_eta"].array().flatten()
counts_eta, edges_eta = np.histogram(eta, bins = 80, range = (-3.5, 3.5))
counts_eta_Ztrue, edges_eta_Ztrue = np.histogram(eta[Ztrue], bins = 80, range = (-3.5, 3.5))
counts_eta_trig, edges_eta_trig = np.histogram(eta[trigger], bins = 80, range = (-3.5, 3.5))

fig, ax = plt.subplots(figsize=(5,5))
ax.step(x=edges_eta, y=np.append(counts_eta, 0), where="post", color = 'k', label = "eta for all muons");
ax.step(x=edges_eta_Ztrue, y=np.append(counts_eta_Ztrue, 0), where="post", color = 'b', label = "eta for Z muons");
ax.step(x=edges_eta_trig, y=np.append(counts_eta_trig, 0), where="post", color = 'r', label = "eta for triggered muons");
ax.set(xlim = (edges_eta[0], edges_eta[-1]), xlabel = r"$\eta$", ylabel = "Frequency");
ax.legend()
fig.tight_layout()
fig.savefig("etaTrigger_wZtrue.pdf")

fig, ax = plt.subplots(figsize=(5,5))
ax.step(x=edges_eta, y=np.append(counts_eta_trig/counts_eta, 0), where="post", color = 'k', label = "Ratio: eta for triggered muons / eta for all muons");
ax.step(x=edges_eta, y=np.append(counts_eta_trig/counts_eta_Ztrue, 0), where="post", color = 'b', label = "Ratio: eta for triggered muons / eta for Z muons");
ax.set(xlim = (edges_eta[0], edges_eta[-1]), ylim = (-0.05, 1), xlabel = r"$\eta$", ylabel = "Ratio")#: eta for triggered muons / eta for all muons");
ax.legend()
fig.tight_layout()
fig.savefig("ratioTriggerEta_wZtrue.pdf")
