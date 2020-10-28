import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ROOT
from ROOT import RooChebychev, RooFit , RooCBShape, RooRealVar, gPad, RooArgList, RooFFTConvPdf, RooGaussian
import seaborn as sns
import pandas as pd

def root_plot(m_ee, distribution, fit, mZmin: int, mZmax: int, title: str):
    # Initiate mZ
    m_ee.setRange("MC_mZfit_range", mZmin, mZmax);
    c_mZ = ROOT.TCanvas("c_mZ", "", 0, 0, 1000, 1000); #Make canvas
    pad_mZdist = ROOT.TPad("pad_mZdist", "pad_mZdist", 0.0, 0.35, 1.0, 1.0);
    c_mZ.SetLeftMargin(1.5);
    pad_mZdist.SetLeftMargin(2.5);
    pad_mZdist.SetBottomMargin(1.0);
    pad_mZdist.Draw();
    pad_mZdist.cd();

    # Frame
    ZeeWenu_mZ_Frame = m_ee.frame(ROOT.RooFit.Title(f"{title}"));
    legend = ROOT.TLegend(0.2, 0.25, 0.4, 0.45, '', 'NBNDC');

    # Plotting on Frame
    distribution.plotOn(ZeeWenu_mZ_Frame, RooFit.Name(str(title)), RooFit.LineColor(2), RooFit.MarkerColor(2));
    fit.plotOn(ZeeWenu_mZ_Frame, RooFit.Range("MC_mZfit_range"), RooFit.LineColor(2), RooFit.Name(str(title))); #Rød
    fit.paramOn(ZeeWenu_mZ_Frame, ROOT.RooFit.Layout(0.55,0.90,0.90));
    mc_ZeeWenu_mZ_resid = ZeeWenu_mZ_Frame.pullHist(); #pullHist residHist
    mc_ZeeWenu_mZ_resid.GetXaxis().SetRangeUser(mZmin, mZmax);
    mc_ZeeWenu_mZ_resid.SetLineColor(2);
    mc_ZeeWenu_mZ_resid.SetMarkerColor(2);
    fit = ZeeWenu_mZ_Frame.findObject("LH fit");
    mc_ZeeWenu_mZ_resid.Draw('same');
    legend.AddEntry( fit , str(title), "lep" );
    legend.SetFillStyle(0);
    ZeeWenu_mZ_Frame.Draw(); #50
    legend.Draw('same');
    c_mZ.cd();

    # Ratio plot
    pad_sigbkg_mZratio = ROOT.TPad("pad_sigbkg_mZratio", "pad_sigbkg_mZratio", 0.0, 0.0, 1.0, 0.35);
    pad_sigbkg_mZratio.SetLeftMargin(2.5);
    pad_sigbkg_mZratio.SetTopMargin(1.00);
    pad_sigbkg_mZratio.SetBottomMargin(0.5);
    pad_sigbkg_mZratio.Draw("same");
    pad_sigbkg_mZratio.cd();
    mc_ZeeWenu_mZ_resid.GetXaxis().SetRangeUser(mZmin, mZmax);
    mc_ZeeWenu_mZ_resid.SetLineColor(2);
    mc_ZeeWenu_mZ_resid.SetMarkerColor(2);
    mc_ZeeWenu_mZ_resid.Draw();
    pad_sigbkg_mZratio.cd();

    # Make line at zero
    unitline_sigbkg_mZ = ROOT.TLine(mZmin, 0.0, mZmax, 0.0);
    unitline_sigbkg_mZ.SetLineStyle(3);
    unitline_sigbkg_mZ.Draw();

    c_mZ.Draw();
    gPad.Modified();
    gPad.Update();
    c_mZ.Print(f'./output/figuresRooFit/081020figuresRooFit_BWxCB_exp_MC/{title}.png') # Save file. Remember that the folder needs to exist
#
    return mc_ZeeWenu_mZ_resid

def PeakFit_likelihood(Likelihood_cut: pd.DataFrame, mass_energy: pd.DataFrame, cutval, plots = True, constant_mean = True, constant_width = True, classifier_name = 'Likelihood', CB = True, Gauss = False, bkg_comb = True, bkg_exp = False, bkg_cheb = False):
    print('Starting fit...')
    matplotlib.use('Agg')
    # Check if we have mass in MeV or GeV
    if np.mean(mass_energy) > 1000:
        normalization_mass = 1000
    else:
        normalization_mass = 1
    sns.set_style("whitegrid") # White background on plot
    prediction = Likelihood_cut # rename to prediction
    # Set range
    mZmin   =  60.0
    mZmax   = 130.0
    # Number of bins
    NbinsZmass = 100

    #Initiate the mass variable
    m_ee = ROOT.RooRealVar("m_ee", "Invariant mass (GeV/c^{2})",mZmin, mZmax);
    m_ee.setRange("MC_mZfit_range", mZmin, mZmax);

    # =============================================================================
    #    fit signal
    # =============================================================================

    # Make a mask in the signal range. Prediction is 0 or 1, so above 0.5 is signal
    mask_mass = (mass_energy/normalization_mass > mZmin) & (mass_energy/normalization_mass < mZmax) & (prediction > 0.5)
    Z_mass_signal = np.array(mass_energy[mask_mass]/normalization_mass); #Make np.array

    # Initiate 1D histogram
    h_mZ_all = ROOT.TH1D("h_mZ_all", "Histogram of Z mass", NbinsZmass, mZmin, mZmax);

    for isample in range(Z_mass_signal.shape[0]):
        score = Z_mass_signal[isample]
        h_mZ_all.Fill(score)

    # Constructs histogram with m_ee as argument from the 1d histogram h_mZ_all
    mc_Zee_mZ = ROOT.RooDataHist("mc_Zee_mZ", "Dataset with Zee m_ee", RooArgList(m_ee), h_mZ_all);

    # Define variables for the fits.
    # BW: Breit-Wigner. CB: Crystal-Ball
    meanBW = ROOT.RooRealVar("meanBW",   "meanBW", 91.1876, 60.0, 120.0); #91.1876
    meanBW.setConstant(True); # this is a theoretical constant

    sigmaBW = ROOT.RooRealVar("sigmaBW", "sigmaBW", 2.4952,  2.0,   20.0); #2.4952
    sigmaBW.setConstant(True); # this is a theoretical constant
    # if constant_mean:

    func_BW = ROOT.RooBreitWigner("func_BW", "Breit-Wigner", m_ee, meanBW, sigmaBW); # Make the function from the constants

    # Crystal ball
    if CB:
        meanCB = RooRealVar("meanCB",   "meanCB",  -0.0716, -10.0, 10.0);
        # meanCB.setConstant(True) #if commented out, it can float between the minimum and maximum
        sigmaCB =  RooRealVar("sigmaCB", "sigmaCB", 0.193, 0, 15);
        # sigmaCB.setConstant(True)
        alphaCB = RooRealVar("alphaCB", "alphaCB", 1.58,  0.0,  10.0);
        # alphaCB.setConstant(True)
        nCB = RooRealVar("nCB", "nCB",0.886, -10, 50.0);
        # nCB.setConstant(True)
        func_sig_CB = RooCBShape("func_CB", "Crystal Ball", m_ee, meanCB, sigmaCB, alphaCB, nCB); # Define Crystal-Ball function
    # Gaussian
    elif Gauss: # Use Gaussian if True in function call
        meanGA = RooRealVar("meanGA",   "meanGA",  10.0, -10.0, 10.0);
        sigmaGA =  RooRealVar("sigmaGA", "sigmaGA", 3.0,   0.01, 10.0);
        if constant_width:
            sigmaGA.setConstant(True)

        nGA = RooRealVar("nGA", "nGA",1.5, 0.0, 20.0);
        func_GA = RooGaussian("func_GA", "Gaussian", m_ee, meanGA, sigmaGA);#, nGA);


    if CB: # Convolute Breit-Wigner and Crystal-Ball
        print("Convoluting a Crystal-Ball and Breit-Wigner for signal")
        func_BWxCB_unextended = RooFFTConvPdf("func_BWxCB_unextended", "Breit-Wigner (X) Crystal Ball", m_ee, func_BW, func_sig_CB);

    elif Gauss: # Convolute Breit-Wigner and Gauss
        print("Convoluting a Gauss and Breit-Wigner for signal")
        func_BWxCB_unextended = RooFFTConvPdf("func_BWxCB_unextended", "Breit-Wigner (X) Gaussian", m_ee, func_BW, func_GA);

    else: # only Breit-Wigner fit on the signal
        print("Fitting only with Breit-Wigner for signal")
        func_BWxCB_unextended = func_BW

    m_ee.setRange("MC_mZfit_range", 85, 97); # Set the fit range for the signal

    nsig = RooRealVar("ntotal", "ntotal", 1000, 0, 10e6); # Define the variable for the number of signal
    func_BWxCB = ROOT.RooExtendPdf("signal_func_Zee", "signal_func_Zee", func_BWxCB_unextended, nsig); # Adding the nsig term to the pdf

    func_BWxCB.fitTo(mc_Zee_mZ, RooFit.Range("MC_mZfit_range")); # Fit the signal

    if plots: # Plots the signal using the function "root_plot" defined above
        mc_Zee_signal = root_plot(m_ee=m_ee, distribution=mc_Zee_mZ, fit=func_BWxCB,
                  mZmin=mZmin, mZmax=mZmax, title=f'signal for cut {cutval}');
#cut {cutval}
    # =============================================================================
    #    background
    # =============================================================================

    nbkg = RooRealVar("nbkg", "nbkg", 1000, 0, 10e6); # Define the variable for the number of background

    #if True:
    m_ee.setRange("MC_mZfit_range", mZmin, mZmax); # Set range for fit as defined in the beginning
    c_bkg_mZ = ROOT.TCanvas("c_bkg_mZ", "", 0, 0, 1000, 500); # Make the canvas for plotting

    Z_mass_background = np.array(mass_energy[mask_mass]/normalization_mass); # Mask for background
    h_mZWenu_all = ROOT.TH1D("h_mZ_all", "Histogram of Z mass", NbinsZmass, mZmin, mZmax); # Initiate 1D histogram

    for isample in range(Z_mass_background.shape[0]):
        score = Z_mass_background[isample]
        h_mZWenu_all.Fill(score);

    # Create the lin + exponential fit
    lam = RooRealVar("lambda", "Exponent", -0.04, -5.0, 0.0);
    func_expo = ROOT.RooExponential("func_expo", "Exponential PDF", m_ee, lam);


    #coef_pol1 =  RooRealVar("coef_pol1", "Slope of background", 0.0, -10.0, 10.0);
    #func_pol1 = ROOT.RooPolynomial("func_pol1", "Linear PDF", m_ee, RooArgList(coef_pol1));

    # Create Chebychev polymonial
    a0 = RooRealVar("a0", "a0",  -0.4,    -5.0, 5.0);
    a1=  RooRealVar("a1", "a1",  -0.03,   -5.0, 5.0);
    a2=  RooRealVar("a2", "a2",   0.02,   -5.0, 5.0);
    a3=  RooRealVar("a3", "a3",   0.02,   -5.0, 5.0);

    # Polynomials with different order
    func_Cpol1 = RooChebychev("func_Cpol1", "Chebychev polynomial of 1st order", m_ee, RooArgList(a0,a1));
    func_Cpol2 = RooChebychev("func_Cpol2", "Chebychev polynomial of 2nd order", m_ee, RooArgList(a0,a1,a2));
    func_Cpol3 = RooChebychev("func_Cpol3", "Chebychev polynomial of 3rd order", m_ee, RooArgList(a0,a1,a2,a3));
    f_exp_mZ = RooRealVar("N_lin_mZ", "CLinear fraction", 0.50, 0, 1);

    m_ee.setRange("low", 60, 70);
    m_ee.setRange("high", 110, 130);

    # Adding exponential and Chebychev if comb:
    if bkg_comb:
        func_ExpLin_mZ_unextended = ROOT.RooAddPdf("func_ExpLin_mZ_unextended", "Exponential and Linear PDF", RooArgList(func_Cpol3, func_expo), RooArgList(f_exp_mZ));
    elif bkg_exp:
        func_ExpLin_mZ_unextended = func_expo
    elif bkg_cheb:
        func_ExpLin_mZ_unextended = func_Cpol3
    else:
        print("No background fit called. Exiting")
        return None

    func_ExpLin_mZ = ROOT.RooExtendPdf("func_ExpLin_mZ", "func_ExpLin_mZ", func_ExpLin_mZ_unextended, nbkg); # Adding the nbkg term to the pdf
    # Constructs histogram with m_ee as argument from the 1d histogram h_mZ_all
    mc_Wenu_mZ = ROOT.RooDataHist("mc_Zee_mZ", "Dataset with Zee m_ee", RooArgList(m_ee), h_mZWenu_all);
    func_ExpLin_mZ.fitTo(mc_Wenu_mZ, RooFit.Range("MC_mZfit_range"));#ROOT.RooFit.Range("low,high")); # Fits background

    #Plotting background
    residue = root_plot(m_ee=m_ee, distribution=mc_Wenu_mZ, fit=func_ExpLin_mZ,
            mZmin=mZmin, mZmax=mZmax, title=f'Background for cut {cutval}');
    #
    # =============================================================================
    #    Combining signal and background
    # =============================================================================

    m_ee.setRange("MC_mZfit_range", mZmin, mZmax);

    Z_mass = np.array(mass_energy[mask_mass]/normalization_mass);
    h_mZWenu = ROOT.TH1D("h_mZ_all", "Histogram of Z mass", NbinsZmass, mZmin, mZmax);

    for isample in range(Z_mass.shape[0]):
        score = Z_mass[isample]
        h_mZWenu.Fill(score);

    # Constructs histogram with m_ee as argument from the 1d hist ogram h_mZ_all
    mc_ZeeWenu_mZ = ROOT.RooDataHist("mc_Zee_mZ", "Dataset with Zee m_ee", RooArgList(m_ee), h_mZWenu);

    ## Fits the data and returns the fraction of background
    f_bkg_mZ = RooRealVar("f_bkg_mZ", "Signal fraction", nbkg.getVal()/nsig.getVal(), 0.0, 1);

    ## Combining the signal and background fits
    func_SigBkg_mZ_unextended = ROOT.RooAddPdf("func_SigBkg_mZ", "Signal and Background PDF", RooArgList(func_ExpLin_mZ_unextended, func_BWxCB_unextended), RooArgList(f_bkg_mZ));
    # func_SigBkg_mZ_unextended = func_BWxCB_unextended;#ROOT.RooAddPdf("func_SigBkg_mZ", "Signal and Background PDF", RooArgList(func_BWxCB_unextended, func_BWxCB_unextended), RooArgList(f_bkg_mZ));
    ntotal = RooRealVar("ntotal", "ntotal", 10000, 0, 10e6);
    func_SigBkg_mZ = ROOT.RooExtendPdf("func_ExpLin_mZ", "func_ExpLin_mZ", func_SigBkg_mZ_unextended, ntotal);

    func_SigBkg_mZ.fitTo(mc_ZeeWenu_mZ); # Fits the full data set

    if plots:
        mc_ZeeWenu_mZ_resid = root_plot(m_ee=m_ee, distribution=mc_ZeeWenu_mZ, fit=func_SigBkg_mZ,
                  mZmin=mZmin, mZmax=mZmax, title=f'Bkg+Sig for cut {cutval}');

    # Baseline ntotal = 41231 (Data)
    # fraction 0.9333
    # Baseline ntotal = 74747 (MC)
    # fraction 0.4427
    # Malte script len(Z_mass)
    bkg = len(Z_mass)*f_bkg_mZ.getVal()
    sig = len(Z_mass)*(1-f_bkg_mZ.getVal())
    print(f_bkg_mZ.getVal())
    #DATA
    #BL_sig = 41231*(1-0.9333) # BL = baseline, the number is the fraction of bkg in baseline
    #BL_bkg = 41231*0.9333     # BL = baseline

    # DATA OS
    # BL_sig = 22276 * (1-0.853) # BL = baseline, the number is the fraction of bkg in baseline
    # BL_bkg = 22276 * 0.853    # BL = baseline

    # DATA SS
    # BL_sig = 18925 * (1-0.993552)#74747 * (1-0.4427)#41054
    # BL_bkg = 18925 - BL_sig

    #MC OS
    # exp
    BL_sig = 46547 * (1-0.0350)#74747 * (1-0.4427)#41054
    BL_bkg = 46547 * 0.0350

    #comb
    #BL_sig = 74747*(1-0.4427) # BL = baseline, the number is the fraction of bkg in baseline
    #BL_bkg = 74747*0.4427     # BL = baseline

    bkg_ratio = bkg/BL_bkg
    sig_ratio = sig/BL_sig

    max_residue = max(abs(mc_ZeeWenu_mZ_resid.getYAxisMax()), abs(mc_ZeeWenu_mZ_resid.getYAxisMin()))
    print(max_residue)
    print(bkg_ratio)
    print(sig_ratio)


    if (bkg_ratio < 1.009) & (sig_ratio < 1.009) &  (abs(mc_ZeeWenu_mZ_resid.getYAxisMin()) < 4) & (abs(mc_ZeeWenu_mZ_resid.getYAxisMax()) < 4):
        # input('....')

        return BL_sig, BL_bkg, sig_ratio, bkg_ratio#max_residue, ntotal.getVal(), nsig.getVal(), nbkg.getVal()return sigmaCB if CB else sigmaGA #sig_ratio, sigma_sig, bkg_ratio, sigma_bkg
    else:
        return 0, 0, 0, 0
    #return BL_sig, BL_bkg, sig_ratio, bkg_ratio#max_residue, ntotal.getVal(), nsig.getVal(), nbkg.getVal()
