#!/usr/bin/env python3
# this scirpt produces Figure 3 in Tang et.al. 2023, comparison between observed and model-predicted luminosity functions in UV, U, B, V, I bands. 


#####import classes/libraries########
import scipy as sc 
import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np
import matplotlib.pyplot as plt
from slugpy import read_cluster
from slugpy.cluster_slug import cluster_slug
from slugpy import slug_pdf, read_cluster
from collections import namedtuple
from catalog_readers import reader_register
from completeness_calculator import comp_register
from clean_legus import clean_legus
import sys 
import emcee
import os 

######import functions for 2D color/magnnitude plots##########
def norm_hist(histx,x):
    db = np.array(np.diff(x), float)
    return histx/db/histx.sum()

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
    
########use argparse###############
import argparse
parser = argparse.ArgumentParser(
    description="Script to plot the color/magnitude plots")
parser.add_argument("-compdir", "--compdir", default=None,
                    help="name of the directory where the "
                    "completeness files are located")
parser.add_argument("-libdir", "--libdir", default=None,
                    help="name of the library directory")
parser.add_argument("-photdir", "--photdir", default=None,
                    help="name of the directory where the photometric pdfs are located")
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
                    "used to create the library; needed to set "
                    "the sampling density correctly")
parser.add_argument("-ct", "--cattype", default="LEGUS",
                    help="type of input catalog; currently "
                    "known values are 'mock' and 'LEGUS'; this "
                    "parameter specifies both the reader used and "
                    "how the completeness is calculated")
parser.add_argument("-p", "--photsystem", default="Vega",
                    help="photometric system used in the catalogs")
parser.add_argument("--tol", type=float, default=1.0e-2,
                    help="tolerance to use when evaluating the "
                    "likelihood function")
parser.add_argument("--bwphot", type=float, default=0.1,
                    help="photometric bandwidth")
parser.add_argument("--bwphys", type=float, default=0.1,
                    help="physical bandwidth")
parser.add_argument("-mdd", "--mdd", default=False,
                    action="store_true",
                    help="fit to mass-dependent disruption model"
                    " instead of mass-independent disruption model"
                    " (default)")
parser.add_argument("-nav", "--nav", type=int, default=6,
                    help="number of intervals to use in approximating "
                    "the A_V distribution")
parser.add_argument("-nl", "--nlib", type=int, default=-1,
                    help="if set to a positive value, this causes "
                    "the analysis to use only the first nlib clusters "
                    "in the library")
parser.add_argument("-v", "--verbose", default=False,
                    action='store_true',
                    help="produce verbose output")
args = parser.parse_args()
# /pkg/linux/anaconda/bin/python ./plot_1DL.py --verbose --bwphot 0.05 --bwphys 0.05 > plot_1PLOTL.log

# load cluster_slug 
mass_pdf = slug_pdf(args.massPDF)
age_pdf = slug_pdf(args.agePDF)
av_pdf = slug_pdf(args.ageAV)
lib_den = sample_den(mass_pdf, age_pdf, av_pdf)

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
    wgts = libwgts(params, mid = not args.mdd)

    # Enforce limits for pure powerlaw model
    logL = 0.0
    if params[0] < -4.0 or        params[0] > 0.0 :
        # alpha_M in [-4, 0], log m_Break is 10 # alpha_T in [-3, 0], log T_mid in [5, 10]
        logL = -np.inf
    mdd = False 
    if not mdd:
        if  params[1] < -3.0 or             params[1] > 0.0 or             params[2] < 5.0 or             params[2] > 10.0:
            logL = -np.inf
    else:
        if params[1] <= 0.0 or            params[1] > 2.0 or            params[2] < 4.0 or            params[2] > 10.0:
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


# Read the catalogs, and do some organizing on them
catalogs = []
allfilters = []
filterslist = []
ncl = 0
cluster_data_dir = './cluster_data/'
catalog = ['hlsp_628c.tab','hlsp_628e.tab']

for cat in catalog:
    data = reader_register['LEGUS'].read(os.path.join(cluster_data_dir,cat))
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


    # Organize the photometric data into groups by filter set, since
    # each will use a different cluster_slug tree to calculate its log
    # likelihood function; for each filter set, retain only the
    # photometry with detections
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
    
    # Increment total number of clusters
    ncl = ncl + len(data["phot"])
    catalogs.append(data)

# For LEGUS data we need to do some cleanup: (1) remove clusters that
# appear in more than one catalog; (2) remove clusters that nominally
# have zero change of being observed, which can get into the sample
# anyway due to peculiarities of the way that the magnitude limit for
# visual classification was combined with the aperture correction
# if args.cattype == "LEGUS":
ncl = clean_legus(catalogs, True)
    

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
lib_all = read_cluster(args.libdir,photsystem='Vega',
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
        #phot_tmp = phot_neb_ex[:,d]
        # if '628c' in cat : 
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
                cat['libcomp'].append(np.load(args.compdir+'libngc628e_comp.npy'))
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
    def __init__(self, p, mid = not args.mdd):
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
        self.pav[-1] = 2.0/self.delta_av - self.pav[-2] - np.sum(self.pav[:-2]+self.pav[1:-1])
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


# In[ ]:


# import observations and compute percentile 
from pctnbin import pctnbin
import numpy as  np

# note that we have two filter sets so change non-detection value to nan 
catalogs_p = catalogs.copy()

bin_ed_l= [] 
hist_l= [] 

if args.mdd:
    params = np.array([-1.35771231,  3.95419843,  0.99826982,  6.64590571, -0.1323126 ,\
       -0.37808041, -0.47548531, -0.97712959, -1.41716597, -0.41051633])
else :
    params = np.array([-0.99927755,  3.85746251, -0.43310727,  6.75511899, -0.10067485,\
        -0.31077786, -0.47420001, -0.87471347, -1.91308959, -0.31790502])
for i in range(-5,0):
    dataset = catalogs[0]['cs'][0]._cluster_slug__filtersets[0]['bp']._bp__dataset
    phys = dataset[:,0:3]
    pobs = catalogs[0]['cs'][0]._cluster_slug__filtersets[0]['bp']._bp__pobs
    libwgt = libwgts(params,True).wgts(phys[pobs>0.])
    hist, bin_ed = np.histogram(dataset[:,i][pobs>0.],bins=800,\
                    weights = libwgt*pobs[pobs>0.]*lib_den.sample_den(phys[pobs>0.]),density=True)
    bin_ed_l.append(bin_ed)
    hist_l.append(hist)

for i,cat in enumerate(catalogs_p):
    idx = np.where(cat['detect'] == False)[0]
    for j in range(len(idx)):
        cat['phot'][idx[j]][0] = np.nan
# concatenate observations from NGC 628c and NGC 628e 
obs = np.concatenate((catalogs_p[0]['phot'],catalogs_p[1]['phot']),axis=0)
obs_hist = []
obs_x = []
for i in range(5):
    hist , x = np.histogram(obs[:,i][~np.isnan(obs[:,0])],bins=20,density=False)
    obs_hist.append(hist)
    obs_x.append(x)

obs_pct = []
# compute percentiles :
for i in range(5):
    pct = np.zeros((len(obs_hist[i]),2))
    for j in range(len(obs_hist[i])):
        pct[j,0] = pctnbin(0.05,obs_hist[i][j])
        pct[j,1] = pctnbin(0.95,obs_hist[i][j])
    obs_pct.append(pct)


# plot 1D luminosity 
allfs = ['UV','U','B','V','I']
color = ['blue','purple','green','brown','red']
fig, axs = plt.subplots(5,sharey=True,figsize=(3.5,10),dpi=300)
plt.subplots_adjust(left = 0.25, bottom = 0.09, hspace = 0.35)
plt.rcParams['font.size'] = 12
plt.rc('font', family='serif') 
plt.rc('text', usetex=True)
fig.supxlabel('magnitude[mag]')
fig.supylabel('log PDF')

LEGUS_comp = args.photdir # directory of pre-computed completeness values of data 
for i in range(5):
    #  load pre-computed library photometric distribustions at the 5th, 50th and 95th percentiles. 
    if args.mdd:
        hist5 = np.load(LEGUS_comp+allfs[i]+'MDDhist.npy')[:,0]
        hist50 = np.load(LEGUS_comp+allfs[i]+'MDDhist.npy')[:,1]
        hist95 = np.load(LEGUS_comp+allfs[i]+'MDDhist.npy')[:,2]
    else :
        hist5 = np.load(LEGUS_comp+allfs[i]+'hist.npy')[:,0]
        hist50 = np.load(LEGUS_comp+allfs[i]+'hist.npy')[:,1]
        hist95 = np.load(LEGUS_comp+allfs[i]+'hist.npy')[:,2]
    ed = 0.5*(bin_ed_l[i-5][1:]+bin_ed_l[i-5][:-1])
    
    if args.mdd : 
        axs[i].plot(ed,hist5 ,label='MDD 5%',linewidth=1.,color='grey',ls='--')
        axs[i].plot(ed,hist50 ,label='MDD 50%',linewidth=1.,color='black')
        axs[i].plot(ed,hist95 ,label='MDD 95%',linewidth=1.,color='grey',ls='--')
    else : 
        axs[i].plot(ed,hist50 ,label=r'50\%',linewidth=0.8,color='black')
        
    axs[i].fill_between(ed,hist5,hist95,color='grey',alpha =0.7)
    axs[i].annotate(allfs[i], xy=(0.5, 0.25), xycoords='axes fraction',color=color[i],fontsize=10)
    norm_cof = np.array(np.diff(obs_x[i]), float)*obs_hist[i].sum()
    axs[i].vlines(x=0.5*(obs_x[i][1:]+obs_x[i][:-1]), ymin=obs_pct[i][:,0]/norm_cof, ymax=obs_pct[i][:,1]/norm_cof, colors=color[i], ls='-', lw=2.5, label='Percentiles')
    axs[i].scatter(0.5*(obs_x[i][1:]+obs_x[i][:-1]),0.5*(obs_pct[i][:,0]/norm_cof+obs_pct[i][:,1]/norm_cof),marker='o',s=20,color=color[i])
    axs[i].set_yscale('log')
    axs[i].invert_xaxis()
    axs[i].set_xlim(np.max(0.5*(obs_x[i][1:]+obs_x[i][:-1]))+0.5,np.min(0.5*(obs_x[i][1:]+obs_x[i][:-1]))-0.5)
    axs[i].set_ylim(1e-4,10)
    from matplotlib.ticker import LogLocator

    axs[i].xaxis.set_major_locator(plt.MaxNLocator(5))
    axs[i].yaxis.set_major_locator(LogLocator(base=10, numticks=4))
    axs[i].tick_params(axis = 'both', which='both', direction='out')
    
if args.mdd:
    plt.savefig('1D_L_mdd.pdf',bbox_inches='tight')
else :
    plt.savefig('1D_L_mid.pdf',bbox_inches='tight')

