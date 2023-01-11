'''
script used to plot color-color plots and color-magnitude plots
'''

############################################
##           import libs/packages         ##

import numpy as np
import numexpr as ne
import os.path as osp
import copy
import os 
from collections import namedtuple
from numpy.random import seed, rand
from slugpy.cluster_slug import cluster_slug
from slugpy import slug_pdf, read_cluster
import scipy as sc 
import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np
from slugpy import read_cluster
from slugpy.cluster_slug import cluster_slug
from slugpy import slug_pdf, read_cluster
from collections import namedtuple
from catalog_reader_l import reader_register
from clean_legus import clean_legus_comp
import sys 
import emcee
import time 
from pctnbin import pctnbin
from comp_cal import *
from varname import nameof # convert strings to var names using varname class
from matplotlib import ticker
import argparse 
from matplotlib.ticker import AutoMinorLocator 
from matplotlib.ticker import MaxNLocator
import matplotlib 
# use latex format
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=12)
##############utility classes#################
def norm_hist(histx,x):
    db = np.array(np.diff(x), float)
    return histx/db/histx.sum()


##############################################
parser = argparse.ArgumentParser(
    description="Script to plot the 2D color diagnostics for LEGUS-SLUG pipeline")
parser.add_argument("libname",
                    help="name of cluster_slug library to use")
parser.add_argument("massPDF",
                    help="name of the mass PDF file "
                    "used to create the library; needed to set "
                    "the sampling density correctly")
parser.add_argument("agePDF",
                    help="name of the age PDF file "
                    "used to create the library; needed to set "
                    "the sampling density correctly")
parser.add_argument("AVPDF",
                    help="names of the AV PDF file "
                    "used to create the library; needed to 
parser.add_argument("-compdir", "--compdir", default=None,
                    help="name of the directory where the "
                    "completeness files are located")
parser.add_argument("-mdd", "--mdd", default=False,
                    action="store_true",
                    help="fit to mass-dependent disruption model"
                    " instead of mass-independent disruption model"
                    " (default)")
parser.add_argument("-de_UV", "--de_UV", default=False,
                    action="store_true",
                    help="Use UV detection or not")
args = parser.parse_args()
##############################################

# set a boolean variable de_UV to flag UV detection
# default is false, when UV is included in color-color plot, set this boolean variable to True.
de_UV = args.de_UV
class sample_den(object):
    def __init__(self, mpdf, tpdf, avpdf):
        self.mpdf = mpdf
        self.tpdf = tpdf
        self.avpdf = avpdf
    def sample_den(self, physprop):
        m = 10.**physprop[:,0]
        # Fix out of range errors due to mass-limited sampling
        m[m < self.mpdf.bkpts[0]] = self.mpdf.bkpts[0]
        m[m > self.mpdf.bkpts[-1]] = self.mpdf.bkpts[-1]
        t = 10.**physprop[:,1]
        av = physprop[:,2]
        sden = self.mpdf(m) * self.tpdf(t) * self.avpdf(av) * m * t
        return sden


def lnprob(params):

    global evalctr

    # Parameters are, in order:
    # 0 = alphaM
    # 1 = log10(mBreak)
    # 2 = alphaT (if args.mdd is False) or gammaMdd (otherwise)
    # 3 = log10(tMid) (if args.mdd is True) or log10(tMddMin) (otherwise)
    # 4 ... 6+args.nav = log p(A_V) at points distributed at A_V = 0 - 3 mag

    # Start clock
    tstart = time.time()
    twgt = tstart

    # Construct wgts object
    wgts = libwgts(params, mid = True)

    # Enforce limits for pure powerlaw model
    logL = 0.0
    if params[0] < -4.0 or \
       params[0] > 0.0 :
        # alpha_M in [-4, 0], log m_Break is 10 # alpha_T in [-3, 0], log T_mid in [5, 10]
        logL = -np.inf
    mdd = args.mdd
    if not mdd:
        if  params[1] < -3.0 or \
            params[1] > 0.0 or \
            params[2] < 5.0 or \
            params[2] > 10.0:
            logL = -np.inf
    else:
        if params[1] <= 0.0 or \
           params[1] > 2.0 or \
           params[2] < 4.0 or \
           params[2] > 10.0:
            # gamma_mdd in (0, 2], log T_mdd,min in [4, 10]
            logL = -np.inf
    if wgts.pav[-1] < 0.0:
        # p(A_V) > 0 everywhere
        logL = -np.inf

    # Evaluate unless we're out of bounds
    if logL == 0.0:
        # Adjust catalog weights for this set of parameters
        for cat in catalogs:
            for cs in cat['cs']:
                cs.priors = wgts.wgts

        # Stop timer for application of weights
        twgt = time.time()

        # Loop over catalogs and filter sets, adding contribution of
        # each to likelihood function
        for cat in catalogs:
            for cs, phot, photerr in zip(cat['cs'], 
                                         cat['phot_filterset'],
                                         cat['photerr_filterset']):
                logL += np.sum(
                    cs.logL(None, phot,
                            photerr=photerr,
                            margindim=range(3)))
    # Stop clock
    # if args.verbose:
    tend = time.time()
    evalctr += 1
    print(("lnprob evaluation {:d} "
            "completed in {:f} sec (reweighting = {:f} sec); "
            "input paramters are "
            "{:s}, logL = {:f}").format(
                evalctr,
                tend - tstart,
                twgt - tstart,
                repr(params),
                logL))

    # Return log likelihood
    return logL 
import numpy as np

# Read the catalogs, and do some organizing on them
catalogs = []
allfilters = []
filterslist = []
ncl = 0
catalog = ['hlsp_628c.tab','hlsp_628e.tab']
for cat in catalog:
    data = reader_register['LEGUS'].read(cat)
    filterslist.append(data["filters"])
    print("   "+"filtersets for ngc628"" loaded and saved")
    # Add filters used in this catalog to global list of filters
    for f in data["filters"]:
        if not f in allfilters:
            allfilters.append(f)
    # Construct the list of all combinations of filters found in this
    # catalog; for each cluster, assign it to one of the filter sets
    filtersets = []
    filtersets_detect = []
    fset = np.zeros(len(data["phot"]))
    for i, d in enumerate(data["detect"]):
        f = list(np.array(data["filters"])[d])
        if f not in filtersets:
            filtersets.append(f)
            filtersets_detect.append(np.copy(d))
        fset[i] = filtersets.index(f)

    data['filtersets'] = filtersets
    data['filtersets_index'] = fset
    data['filtersets_detect'] = filtersets_detect

    # Create a "completeness calculator" object for each filter set
    cid_filterset = []
    phot_filterset = []
    photerr_filterset = []
    for i, d in enumerate(data['filtersets_detect']):
        idx = data['filtersets_index'] == i
        cid_filterset.append(data['cid'][idx])
        phot_filterset.append(data['phot'][idx][:,d])
        photerr_filterset.append(data['photerr'][idx][:,d])
    data['cid_filterset'] = cid_filterset
    data['phot_filterset'] = phot_filterset
    data['photerr_filterset'] = photerr_filterset
    data['comp_filterset']= completeness_LEGUS(data['path'],data['phot_filterset']).comp_LEGUS()
    # Increment total number of clusters
    ncl = ncl + len(data["phot"])
    catalogs.append(data)

# For LEGUS data we need to do some cleanup: (1) remove clusters that
# appear in more than one catalog; (2) remove clusters that nominally
# have zero change of being observed, which can get into the sample
# anyway due to peculiarities of the way that the magnitude limit for
# visual classification was combined with the aperture correction
# if args.cattype == "LEGUS":
from clean_legus import clean_legus_comp

ncl = clean_legus_comp(catalogs, True)
    

print("Completed reading the following input catalogs:")
for cat in catalogs:
    print("   "+cat["basename"]+":")
    print("      {:d} clusters".format(len(cat["phot"])))
    print("      filters: {:s}".format(repr(cat["filters"])))
    print("      filter combinations:")
    for p, f in zip(cat['phot_filterset'], cat['filtersets']):
        print("         {:s} ({:d} clusters)".format(
            repr(f), len(p)))
            
# Read the PDF files that were used to generate cluster slug catalog,
# so that we can set the sample density
mass_pdf = slug_pdf(args.massPDF)
age_pdf = slug_pdf(args.agePDF)
av_pdf = slug_pdf(args.AVPDF)
lib_den = sample_den(mass_pdf, age_pdf, av_pdf)

# Read the slug library
print("Loading cluster_slug library data")
lib_all = read_cluster(args.libname,photsystem='Vega',
                       read_filters=allfilters)

# Save memory by extracting the fields we need and deleting the rest
cid = lib_all.id
actual_mass = lib_all.actual_mass
form_time = lib_all.form_time
eval_time = lib_all.time
A_V = lib_all.A_V
phot_neb_ex = lib_all.phot_neb_ex
filter_names = lib_all.filter_names
filter_units = lib_all.filter_units
del lib_all


# Compute observational completeness of library clusters for all
# catalogs and filter sets
ncl_init = len(actual_mass)
keep = np.zeros(ncl_init, dtype=np.bool)

for cat in catalogs:
    cat['libcomp'] = []
    for d in cat['filtersets_detect']:
        if '628c' in cat['basename'] : 
            if np.sum(d) == 5:
                cat['libcomp'].append(np.load(args.compdir+'libngc628c_comp.npy'))
                keep = np.logical_or(keep,cat['libcomp'][-1] > 0.0)
                print("Loaded library for c with all 5 bands")
            elif np.sum(d) == 4:
                cat['libcomp'].append(np.load(args.compdir+'libngc628cnoUV_comp.npy'))
                keep = np.logical_or(keep,cat['libcomp'][-1] > 0.0)
                print("Loaded library for c with 4 bands, no UV band")
            else :
                print('Completeness not valid')
                exit(0)
        elif '628e' in cat['basename']:
            if  np.sum(d) == 5:
                cat['libcomp'].append(np.load(args.compdir+'/libngc628e_comp.npy'))
                keep = np.logical_or(keep,cat['libcomp'][-1] > 0.0)
                print("Loaded library for e with all 5 bands")
            elif  np.sum(d) == 4:
                cat['libcomp'].append(np.load(args.compdir+'libngc628enoUV_comp.npy'))
                keep = np.logical_or(keep,cat['libcomp'][-1] > 0.0)
                print("Loaded library for e with 4 bands, no UV band")
            else :
                print('Completeness not valid')
                exit(0)

# Prune the library of clusters for which there is 0 probability of
# the cluster being observed in any catalog and filter set
cid = cid[keep]
actual_mass = actual_mass[keep]
eval_time = eval_time[keep]
form_time = form_time[keep]
A_V = A_V[keep]
phot_neb_ex = phot_neb_ex[keep]
for cat in catalogs:
    for i in range(len(cat['libcomp'])):
        cat['libcomp'][i] = cat['libcomp'][i][keep]

for cat in catalogs:
    print("   {:s}:".format(cat["basename"]))
    cat['cs'] = []
    for i in range(len(cat['filtersets'])):
        print("      filters {:s}...".
                  format(repr(cat["filtersets"][i])))
        idx_cat = []
        for j,id in enumerate(cat['filtersets'][i]):
            idx_cat.append(filter_names.index(id))
        field_list = ['id', 'actual_mass', 'time', 'form_time', 'A_V',
                      'phot_neb_ex', 'filter_names', 'filter_units']
        idx = cat['filtersets_detect'][i]
        keep = cat['libcomp'][i] > 0
        fields = [np.copy(cid[keep]), 
                  np.copy(actual_mass[keep]), 
                  np.copy(eval_time[keep]),
                  np.copy(form_time[keep]),
                  np.copy(A_V[keep]),
                  np.copy(phot_neb_ex[:,idx_cat][keep]),
                  list(np.array(cat['filters'])[idx]),
                  list(np.array(filter_units[:len(idx_cat)]))]
        lib_type = namedtuple('cluster_data', field_list)
        lib = lib_type(*fields)
        cat['cs'].append(
          cluster_slug(lib = lib,
                         sample_density = lib_den.sample_den,
                         reltol = 1.0e-2,
                         bw_phot = 0.05,
                         bw_phys = 0.05))
        cat['cs'][-1].add_filters(lib.filter_names, 
                                  pobs=cat['libcomp'][i][keep])
        cat['cs'][-1].make_cache(range(3), filters=lib.filter_names)

# OBSERVED CLUSTER LUMINOSITIES 
# Class used to set the weights on the library; there are effectively
# two versions, one for mass-independent disruption and one for
# mass-dependent disruption, but we combine them into the same class
# for simplicity
class libwgts(object):
    def __init__(self, p, mid=True):
        self.alphaM = p[0]
        self.mBreak = 10.**p[1]
        self.mid = mid
        if self.mid:
            self.alphaT = p[2]
            self.tMid   = 10.**p[3]
        else:
            self.gammaMdd = p[2]
            self.tMddMin  = 10.**p[3]
        self.nav       = len(p) - 4
        self.delta_av = 3.0/self.nav
        self.av = np.arange(0, 3.0+self.delta_av/2.0, self.delta_av)
        self.pav = np.zeros(self.nav+1)
        self.pav[:-1] = 10.**p[4:]
        self.pav[-1] = 2.0/self.delta_av - self.pav[-2] \
                       - np.sum(self.pav[:-2]+self.pav[1:-1])
    def wgts(self, physprop):
        # Note: need to make local, non-sliced references in order to
        # use numexpr

        # Inputs
        logm = physprop[:,0]
        logt = physprop[:,1]
        av = physprop[:,2]

        # Stored parameters
        alphaM = self.alphaM
        mBreak = self.mBreak
        if self.mid:
            alphaT = self.alphaT
            tMid = self.tMid
            logtMid = np.log10(tMid)
        else:
            gammaMdd = self.gammaMdd
            tMddMin = self.tMddMin

        # Get weight for M, T distributions
        if self.mid:
            wgt = ne.evaluate(
                "10.**((alphaM+1)*logm)*"
                "exp(-10.**logm/mBreak) * "
                "where( logt <= logtMid, "
                "       10.**logt/tMid, "
                "       (10.**logt/tMid)**(alphaT+1) )")
        else:
            eta = ne.evaluate(
                "(1.0 + gammaMdd*(100.0/10.**logm)**gammaMdd"
                " * 10.**logt/tMddMin)**(1.0/gammaMdd)")
            wgt = ne.evaluate(
                "10.**((alphaM+1)*logm)*"
                "eta**(alphaM+1.0-gammaMdd)*"
                "exp(-10.**logm*eta/mBreak)*"
                "10.**logt")

        # Add A_V weights
        for i in range(self.nav):
            avlo = self.av[i]
            avhi = self.av[i+1]
            pavlo = self.pav[i]
            pavhi = self.pav[i+1]
            avslope = (pavhi-pavlo) / (avhi-avlo)
            wgt = ne.evaluate(
                "wgt * where( (av >= avlo) & (av < avhi),"
                "             pavlo + (av-avlo)*avslope,"
                "             1.0 )"
            )

        # Return final result
        return wgt

# append phot data in each filtersets and catalogs

if args.de_UV :
    id_filt = ['UV','U','B','V','I'] 
    for i,filtn in enumerate(id_filt):
        globals()[filtn] = []
        ic = i - 5
        for cat in catalogs:
            for filt in cat['phot_filterset'][0::2]:
                for cl in filt:
                    globals()[filtn].append(cl[ic])
        globals()[filtn] = np.array(globals()[filtn])
else :
    id_filt = ['UV','U','B','V','I'] 
    for i,filtn in enumerate(id_filt):
        globals()[filtn] = []
        ic = i - 5
        for cat in catalogs:
            for filt in cat['phot_filterset']:
                try:
                    for cl in filt:
                        globals()[filtn].append(cl[ic])
                        
                except IndexError:
                    None
        globals()[filtn] = np.array(globals()[filtn])
# sampling using obs ranges
ncr = []
for cat in catalogs:
    for phot in cat['phot_filterset']:
        ncr.append(np.shape(phot)[0])
nt = np.sum(ncr)
ncr = ncr/nt # number of clusters of each filterset 
if args.de_UV :
    ncr = ncr[0::2]
# main function to plot the pdf of colors/magnitudes 
"""
This is a little utility function that makes 2d PDF plots with
flanking histograms.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from scipy.stats import binned_statistic_2d
from scipy.stats import binned_statistic
from matplotlib.ticker import MaxNLocator 
from matplotlib.ticker import LogLocator
def pdfplot(catalogs, obs_x, obs_y, ncr_val, idx1, idx2, idy1, idy2, fignum=None, xlim=None, ylim=None,
            nxbin=50, nybin=50, thresh=None, threshfrac=0.99, vmin=None, vmax=None,
            log=True, zmax=None, zscaling='density', xhistlim=[1e-3,3], yhistlim=[1e-3,5], colorbar=True, de_UV=args.de_UV,
            nticks=3, xlabel=None, ylabel=None, poi_err=True,
            scat_alpha=1.0, aspect='auto'):
    """
    Makes a 2D PDF plot with flanking histograms.

    Parameters:
       catalogs: cluster catalog files, .tab files
          cluster data 
       obs_x : array, shape (N,)
          observed x axis color such as UV-U. UV and U are arrays of magnitude data in different filters 
       obs_y : array, shape (N,)
          observed y axis color such as U-B. U and B are arrays of magnitude data in different filters 
       ncrval : array, shape(N_filterset,)
          total number of filtersets looping over all catalog files
       idx1 : int 
          index of the first entry of x axis of color, such as UV-U on the x axis, UV has an index of 0
       idx1 : int 
          index of the second entry of x axis of color, such as UV-U on the x axis, U has an index of 1
       idy0 : 
          index of the first entry of y axis of color, such as U-B on the y axis, U has an index of 0
       idy1 : 
          index of the second entry of y axis of color, such as U-B on the y axis, B has an index of 1
       fignum : int
          figure number in which to create the plot
       xlim : arraylike, shape (2,)
          plotting limits in x direction
       ylim : arraylike, shape (2,)
          plotting limits in y direction
       nxbin : int or arraylike, shape (N,)
          number of bins in x direction
       nybin : int or arraylike, shape (N,)
          number of bins in y direction
       thresh : float
          threshhold below which to hide the density histogram and
          show individual points; default is to use threshfrac instead
       threshfrac : float
          minimum fraction of the points that should be within the
          density threshhold; points in lower density regions are shown
          individually in a scatter plot
       log : bool
          use linear or logarithmic scale for PDFs
       zmax : float
          maximum value for 2D density PDF plots; default if this is
          not set depends on zscaling
       zscaling : 'max' | 'count' | 'frac' | 'density' | 'normed'
          method used to scale the PDFs; 'max' means that all
          histograms / PDFs are normalised to have a maximum of unity,
          'count' means that histograms / PDFs show absolute number
          counts in each bin, 'frac' means that histograms / PDFs show
          the fraction of the points in each bin, 'density' means
          that histograms / PDFs show the density of points in each
          bin, and 'normed' means that histograms / PDFs show the
          probability density in each bin
       xhistlim : arraylike, shape (2,)
          limits on the histogram in the x direction
       yhistlim : arraylike, shape (2,)
          limits on the histogram in the x direction
       colorbar : bool
          include a color bar for the density 2D PDFs or not
       de_UV : bool
          include UV detection or not, default is False
       nticks : int
          number of tick marks on the color bar
       xlabel : string
          labels for x dimension
       ylabel : string
          labels for y dimension
       poi_err : bool
          include poission error on each bin or not 
       scat_alpha : float
          alpha value for scatter plot points

    Returns
       Nothing
    """
    # Set plotting grid 
    xgrd = np.linspace(xlim[0], xlim[1], nxbin+1)
    ygrd = np.linspace(ylim[0], ylim[1], nybin+1)
    xgrd_h = 0.5*(xgrd[1:]+xgrd[:-1])
    ygrd_h = 0.5*(ygrd[1:]+ygrd[:-1])
    xx, yy = np.meshgrid(xgrd_h, ygrd_h)
    obxgrd_h = 0.5*(xgrd[1:]+xgrd[:-1])
    obygrd_h = 0.5*(ygrd[1:]+ygrd[:-1])
    xx, yy = np.meshgrid(xgrd_h, ygrd_h)
    obxx, obyy = np.meshgrid(xgrd_h, ygrd_h)
    
    xlim1 = np.sort(xlim)
    ylim1 = np.sort(ylim)
    # set data limits 
    obs_xlim = xlim
    obs_ylim = ylim 
    obxlim1 = np.sort(obs_xlim)
    obylim1  = np.sort(obs_ylim)
    
    # plt.rcParams['font.size'] = 8
    plt.rc('font', family='serif') 
    

    # loop through all catalogs and filtersets 
    if de_UV:
        for j,cat in enumerate(catalogs):
                dataset = cat['cs'][0]._cluster_slug__filtersets[0]['bp']._bp__dataset
                phys = dataset[:,0:3]
                pobs = cat['cs'][0]._cluster_slug__filtersets[0]['bp']._bp__pobs
                # default param, best parameter of mid fits 
                param =  np.array([-0.99927755,  3.85746251, -0.43310727,  6.75511899, -0.10067485,\
                    -0.31077786, -0.47420001, -0.87471347, -1.91308959, -0.31790502])# best param
                libwgt = libwgts(param, mid = True).wgts(phys[pobs>0.])
                count, xe, ye, binidx = binned_statistic_2d(dataset[:,idx1][pobs>0]-dataset[:,idx2][pobs>0],\
                dataset[:,idy1][pobs>0]-dataset[:,idy2][pobs>0],\
                libwgt*pobs[pobs>0.]/lib_den.sample_den(phys[pobs>0.]), statistic='sum',\
                range = [[float(obxlim1[0]), float(obxlim1[1])],
                            [float(obylim1[0]), float(obylim1[1])]],\
                bins = nxbin, expand_binnumbers=True)
                # compute the weighted histogram for 1D colors

                count_x, xe_, binidx_x = binned_statistic(dataset[:,idx1][pobs>0]-dataset[:,idx2][pobs>0],\
                libwgt*pobs[pobs>0.]/lib_den.sample_den(phys[pobs>0.]), statistic='sum',\
                range = [float(obxlim1[0]), float(obxlim1[1])],\
                bins = nxbin)

                count_y, ye_, binidx_y = binned_statistic(dataset[:,idy1][pobs>0]-dataset[:,idy2][pobs>0],\
                libwgt*pobs[pobs>0.]/lib_den.sample_den(phys[pobs>0.]), statistic='sum',\
                range = [float(obylim1[0]), float(obylim1[1])],\
                bins = nybin)
                

                count += count*ncrval[j]
                count_x += count_x*ncrval[j]
                count_y += count_y*ncrval[j]
    else:
        for j,cat in enumerate(catalogs):
            for k in range(len((cat['cs']))):
                ncr_i = 2*j+k # calculate the index of filterset 
                dataset = cat['cs'][k]._cluster_slug__filtersets[0]['bp']._bp__dataset
                phys = dataset[:,0:3]
                pobs = cat['cs'][k]._cluster_slug__filtersets[0]['bp']._bp__pobs
                # default param, best parameter of mid fits 
                param =  np.array([-0.99927755,  3.85746251, -0.43310727,  6.75511899, -0.10067485,\
                    -0.31077786, -0.47420001, -0.87471347, -1.91308959, -0.31790502])# best param
                libwgt = libwgts(param, mid = True).wgts(phys[pobs>0.])
                
                count, xe, ye, binidx = binned_statistic_2d(dataset[:,idx1][pobs>0]-dataset[:,idx2][pobs>0],\
                dataset[:,idy1][pobs>0]-dataset[:,idy2][pobs>0],\
                libwgt*pobs[pobs>0.]/lib_den.sample_den(phys[pobs>0.]), statistic='sum',\
                range = [[float(obxlim1[0]), float(obxlim1[1])],
                            [float(obylim1[0]), float(obylim1[1])]],\
                bins = nxbin, expand_binnumbers=True)
                # compute the weighted histogram for 1D colors

                count_x, xe_, binidx_x = binned_statistic(dataset[:,idx1][pobs>0]-dataset[:,idx2][pobs>0],\
                libwgt*pobs[pobs>0.]/lib_den.sample_den(phys[pobs>0.]), statistic='sum',\
                range = [float(obxlim1[0]), float(obxlim1[1])],\
                bins = nxbin)

                count_y, ye_, binidx_y = binned_statistic(dataset[:,idy1][pobs>0]-dataset[:,idy2][pobs>0],\
                libwgt*pobs[pobs>0.]/lib_den.sample_den(phys[pobs>0.]), statistic='sum',\
                range = [float(obylim1[0]), float(obylim1[1])],\
                bins = nybin)
                

                count += count*ncrval[ncr_i]
                count_x += count_x*ncrval[ncr_i]
                count_y += count_y*ncrval[ncr_i]


    # Get 2D histogram; note that we have to handle the case of
    # inverted axis limits with care, because binned_statistic_2d
    # doesn't natively support them




    if xlim[0] > xlim[1]:
        count = count[::-1, :]
        xe = xe[::-1]
        binidx[0,:] = nxbin+1 - binidx[0,:]

    if ylim[0] > ylim[1]:
        count = count[:, ::-1]
        ye = ye[::-1]
        binidx[1,:] = nybin+1 - binidx[1,:]        

    # Set z
    if zscaling == 'max':
        z = count / np.amax(count)
    elif zscaling == 'count':
        z = count
    elif zscaling == 'frac':
        z = count / len(x)
    elif zscaling == 'density':
        z = count / np.abs((xe[1]-xe[0])*(ye[1]-ye[0]))
    elif zscaling == 'normed':
        z = count / np.abs(len(x)*(xe[1]-xe[0])*(ye[1]-ye[0]))

    # Set minima and maxima for 2D plot
    if zmax is None:
        if zscaling == 'max':
            zmax = 1.0
        else:
            zmax = np.amax(z)
    if thresh is not None:
        zmin = thresh
    else:
        zsort = np.sort(z, axis=None)
        csum = np.cumsum(zsort)
        csum = csum/csum[-1]
        zmin = zsort[np.argmax(csum > 1.0-threshfrac)]
    if log:
        zmin = np.log10(zmin)
        zmax = np.log10(zmax)

    # Take log if requested
    if log:
        if np.amax(z) == 0.0:
            raise ValueError("cannot use log scale: no positive z values")
        z[z == 0] = 1.0e-6*np.amin(z[z > 0])
        z[z == 0] = np.amin(z[z > 0])
        z = np.log10(z)
        
    # Get indices of individual points to show
        
    # Set up plot
    fig = plt.figure(figsize=(3.5, 2.9), dpi=250)
    plt.clf()
    gs = gridspec.GridSpec(4, 4)
    axcen = plt.subplot(gs[1:, :-1])
                  

    axcen.scatter(obs_x[::25],obs_y[::25],
                  color='lightskyblue', s=11, alpha=scat_alpha,
                  edgecolor='k')
                  
    # Plot density map
    img = axcen.imshow(np.transpose(z), 
                       origin='lower', aspect=aspect,
                       vmin=vmin, vmax=vmax, cmap='hot_r',
                       extent=(xlim[0], xlim[-1],\
                            ylim[0], ylim[-1]))
                            
    # Set ticks 
    axcen.set_xlim(xlim)
    axcen.set_ylim(ylim)
    
    axcen.xaxis.set_major_locator(MaxNLocator(4))
    axcen.yaxis.set_major_locator(MaxNLocator(5))

    axcen.xaxis.set_minor_locator(AutoMinorLocator(5))
    axcen.yaxis.set_minor_locator(AutoMinorLocator(5))
    axcen.tick_params(axis = 'both',which='both',direction='in')
    
    # Set plot range
    axcen.set_xlim(xlim)
    axcen.set_ylim(ylim)

    # Add labels
    if xlabel is not None:
        axcen.set_xlabel(xlabel)
    if ylabel is not None:
        axcen.set_ylabel(ylabel)

    # Get 1D histograms
    histx, xe \
        = count_x, xe_
    histy, ye \
        = count_y, ye_

    
    # Get 1D histograms for data and lib 
    histobx, obxe \
        = np.histogram(obs_x, bins=nxbin, range=xlim1)
    histoby, obye \
        = np.histogram(obs_y, bins=nybin, range=ylim1)
    if poi_err:
        histobx_err = np.array([pctnbin(0.84,hx) for hx in histobx]) 
        histoby_err = np.array([pctnbin(0.84,hy) for hy in histoby])

        
    if xlim[0] > xlim[1]:
        histx = histx[::-1]
        xe = xe[::-1]
    if ylim[0] > ylim[1]:
        histy = histy[::-1]
        ye = ye[::-1]
    if obs_xlim[0] > obs_xlim[1]:
        histobx = histobx[::-1]
        obxe = obxe[::-1]
    if obs_ylim[0] > obs_ylim[1]:
        histoby = histoby[::-1]
        obye = obye[::-1]

    if zscaling == 'max':
        histx = histx / float(np.amax(histx))
        histy = histy / float(np.amax(histy))
        
    elif zscaling == 'count':
        pass
    elif zscaling == 'density':
        histx = histx / (np.array(np.diff(xe), float)*histx.sum())
        histy = histy / (np.array(np.diff(ye), float)*histy.sum())
        histobx = histobx / (np.array(np.diff(obxe), float)*histobx.sum())
        histoby = histoby / (np.array(np.diff(obye), float)*histoby.sum())
        histobx_err = histobx_err / (np.array(np.diff(obxe), float)*histobx_err.sum())
        histoby_err = histoby_err / (np.array(np.diff(obye), float)*histoby_err.sum())
        
        
    else:
        raise ValueError("bad value of zscaling")
    # Add flanking histograms
    if zscaling == 'max':
        label = 'Scaled PDF'
    elif zscaling == 'count':
        label = r'N'
    elif zscaling == 'frac':
        label = 'Fraction'
    elif zscaling == 'density':
        label = 'Log PDF'
    elif zscaling == 'normed':
        label = 'PDF'
    axtop = plt.subplot(gs[0, :-1])
    axtop.bar((xe[:-1] + xe[1:]) / 2, histx, xe[1]-xe[0],
               align='center',
                facecolor='blue', 
                edgecolor='blue',
                alpha = 0.5)
    axtop.bar(obxe[:-1],histobx,obxe[1]-obxe[0],
                align='edge',
                 facecolor='red',hatch='\\', 
                 alpha= 0.6,
                 edgecolor='red')
    axtop.plot(0.5*(obxe[:-1]+obxe[1:]),histobx_err,'-',c='red',linewidth=1.5)
    axtop.set_yscale('log')
    axtop.set_ylabel(label)
    axtop.set_xlim(xlim)
    if yhistlim is not None:
        axtop.set_ylim(yhistlim)
    axtop.set_xticklabels([])
    
    axtop.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
    axtop.yaxis.set_minor_locator(AutoMinorLocator(5))
    axtop.xaxis.set_major_locator(MaxNLocator(7))
    axtop.xaxis.set_minor_locator(AutoMinorLocator(5))
    
    axtop.tick_params(axis = 'both',which='both',direction='in',labelsize=6)
                
    axright = plt.subplot(gs[1:, -1])
    axright.barh((ye[:-1] + ye[1:]) / 2, histy, ye[1]-ye[0], 
                 align='center',
                facecolor='blue', 
                edgecolor='blue',alpha=0.5)

    axright.barh(obye[:-1],histoby,obye[1]-obye[0],
                  align='edge',
                 facecolor='red',
                 hatch='\\',
                 alpha= 0.6,
                 edgecolor='red')
  
    axright.plot(histoby_err,0.5*(obye[:-1]+obye[1:]),'-',c='red',linewidth=1.5)
    axright.set_xscale('log')
    axright.set_xlim(xhistlim)
    axright.set_ylim(ylim)
    axright.set_yticklabels([])
    axright.set_xlabel(label)
    
    axright.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
    axright.xaxis.set_minor_locator(AutoMinorLocator(5))
    axright.yaxis.set_major_locator(MaxNLocator(7))
    axright.yaxis.set_minor_locator(AutoMinorLocator(5))
    axright.tick_params(axis = 'both',which='both',direction='in',labelsize=6)
    
    
    
    # Add colorbar
    if colorbar:
        if zscaling == 'max':
            if log:
                label = 'log Scaled PDF'
            else:
                label = 'Scaled PDF'
        elif zscaling == 'count':
            if log:
                label = 'log N'
            else:
                label = 'N'
        elif zscaling == 'frac':
            if log:
                label = 'log fraction'
            else:
                label = 'fraction'
        elif zscaling == 'density':
            if log:
                label = 'Log PDF'
            else:
                label = 'Density'
        elif zscaling == 'normed':
            if log:
                label = 'log PDF'
            else:
                label = 'PDF'
        axcbar = plt.subplot(gs[0,-1])
        from matplotlib.ticker import FormatStrFormatter
        
        cbar = plt.colorbar(img,ax=axcbar, orientation='vertical',
                            fraction=0.8,
                            aspect=10
                            )
        if nticks is not None:
            cbar.set_ticks(np.linspace(vmin, vmax, 4))
            cbar.ax.tick_params(labelsize=6) 
            cbar.set_label(label, size='x-small')
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axcbar.remove()
    # save the whole figure 
    if de_UV : 
        fig.savefig('color_mag_UV_U_B.pdf',bbox_inches='tight')
    else :
        fig.savefig('color_mag_'+xlabel[0]+'_'+xlabel[2]+'_'+ylabel[2]+'.pdf',bbox_inches='tight')
    # Return handles to axes
    return (axcen, axtop, axright)

# call function 
if args.de_UV:
  pdfplot(catalogs,UV-U,U-B, xlim = [-1.7,1.9], ylim = [-2.1,1.4],vmin=6,vmax=8.6,xhistlim=[0.008,2],yhistlim = [0.008,3],idx1=-5,idx2=-4,idy1=-4,idy2=-3,ncrval=ncr,de_UV=True,xlabel='UV-U [mag]',ylabel='U-B [mag]')
else :
  pdfplot(catalogs,U-B,B-V, xlim = [-3.3,1.7], ylim = [-0.5,1.7],vmin=6,vmax=8.7,xhistlim=[0.008,3],yhistlim = [0.008,3.5],idx1=-4,idx2=-3,idy1=-3,idy2=-2,ncrval=ncr,de_UV = args.de_UV,xlabel='U-B [mag]',ylabel='B-V [mag]')
  pdfplot(catalogs,B-V,V-I, xlim = [-0.5,1.3], ylim = [-0.8,2.3],vmin=7,vmax=8.6,xhistlim=[0.006,3],yhistlim = [0.01,4],idx1=-3,idx2=-2,idy1=-2,idy2=-1,ncrval=ncr,de_UV = args.de_UV,log=True, xlabel='B-V [mag]',ylabel='V-I [mag]')


