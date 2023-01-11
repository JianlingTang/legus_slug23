"""
This script finds the cluster population parameters using truncated model for catalog.
"""

import argparse
import numpy as np
import numexpr as ne
import os.path as osp
import time
import copy
from collections import namedtuple
from numpy.random import seed, rand
from slugpy.cluster_slug import cluster_slug
from slugpy import slug_pdf, read_cluster
import emcee
from catalog_readers import reader_register
from completeness_calculator import comp_register
from clean_legus import clean_legus
import sys 


# Parse the inputs
parser = argparse.ArgumentParser(
    description="Script to find best fit cluster populuation "
    "parameters for star cluster catalogs")
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
                    "used to create the library; needed to set "
                    "the sampling density correctly")
parser.add_argument("catalogs", nargs="+", default=None,
                    help="names of catalog files to be processed")
parser.add_argument("-ld", "--libdir", default=None,
                    help="name of the directory where the "
                    "cluster_slug library is located")
parser.add_argument("-ct", "--cattype", default="LEGUS",
                    help="type of input catalog; currently "
                    "known values are 'mock' and 'LEGUS'; this "
                    "parameter specifies both the reader used and "
                    "how the completeness is calculated")
parser.add_argument("-o", "--outname", default=None,
                    help="name of output file; default is first "
                    "catalog file with extension changed to .chain")
parser.add_argument("-p", "--photsystem", default="Vega",
                    help="photometric system used in the catalogs")
parser.add_argument("--tol", type=float, default=1.0e-2,
                    help="tolerance to use when evaluating the "
                    "likelihood function")
parser.add_argument("--bwphot", type=float, default=0.1,
                    help="photometric bandwidth")
parser.add_argument("--bwphys", type=float, default=0.1,
                    help="physical bandwidth")
parser.add_argument("-nw", "--nwalkers", type=int, default=100,
                    help="number of walkers to use in the MCMC")
parser.add_argument("-ni", "--niter", type=int, default=500,
                    help="number of MCMC iterations")
parser.add_argument("-mdd", "--mdd", default=False,
                    action="store_true",
                    help="fit to mass-dependent disruption model"
                    " instead of mass-independent disruption model"
                    " (default)")
parser.add_argument("--restart", default=False, action="store_true",
                    help="restart from existing output file")
parser.add_argument("-nav", "--nav", type=int, default=6,
                    help="number of intervals to use in approximating "
                    "the A_V distribution")
parser.add_argument("-nparam", "--nparam", type=int, default=4,
                    help="number of parameters needed in model "
                    "despite the A_V distribution")
parser.add_argument("-c", "--c", default=False,
                    action='store_true',
                    help="use 628c or 628e")
parser.add_argument("-pl", "--pl", default=False,
                    action='store_true',
                    help="use pure powerlaw model")
parser.add_argument("-trun", "--trun", default=False,
                    action='store_true',
                    help="use the truncated model")
parser.add_argument("-nl", "--nlib", type=int, default=-1,
                    help="if set to a positive value, this causes "
                    "the analysis to use only the first nlib clusters "
                    "in the library")
parser.add_argument("-v", "--verbose", default=False,
                    action='store_true',
                    help="produce verbose output")
args = parser.parse_args()
################################
# Helper classes and functions #
################################

# Define a class that is built from the sampling PDFs, and has a
# function that takes as input an array of physical properties from
# cluster_slug and returns the sample density at that set of physical
# properties. We will use this to tell cluster_slug the sampling
# density of our library. Note that there is a minor subtlety in that
# cluster_slug works on log mass and log age, while the PDFs we have
# just read are PDFs on mass and age; we multiply by a factor of m * t
# to correct for this.
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

# Class used to set the weights on the library; there are effectively
# two versions, one for mass-independent disruption and one for
# mass-dependent disruption, but we combine them into the same class
# for simplicity
class libwgts(object):
    def __init__(self, p, mid = True): # for powerlaw 
        self.alphaM = p[0]
        self.mBreak = 10.**p[1]
        self.mid = mid
        if self.mid:
            self.alphaT = p[1]
            self.tMid   = 10.**p[3]
        else:
            self.gammaMdd = p[1]
            self.tMddMin  = 10.**p[3]
        self.nav       = len(p) - args.nparam
        self.delta_av = 3.0/self.nav
        self.av = np.arange(0, 3.0+self.delta_av/2.0, self.delta_av)
        self.pav = np.zeros(self.nav+1)
        self.pav[:-1] = 10.**p[args.nparam:]
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

    
# Function to return log likelihood for mass-independent disruption
evalctr = 0
def lnprob(params):

    global evalctr

    # Parameters are, in order:
    # 0 = alphaM
    # 1 = log10(mBreak)
    # 2 = alphaT (if args.mdd is False) or gammaMdd (otherwise)
    # 3 = log10(tMid) (if args.mdd is True) or log10(tMddMin) (otherwise)
    # 4 ... 6+args.nav = log p(A_V) at points distributed at A_V = 0 - 3 mag

    # Start clock
    if args.verbose:
        tstart = time.time()
        twgt = tstart

    # Construct wgts object
    wgts = libwgts(params, mid = not args.mdd)

    # Enforce limits
    logL = 0.0
    if params[0] < -4.0 or \
       params[0] > 0.0 or \
       params[1] < 2.0 or \
       params[1] > 8.0 : 
        # alpha_M in [-4, 0], log m_Break in [2, 11]
        logL = -np.inf
    if not args.mdd:
        if params[2] < -3.0 or \
           params[2] > 0.0 or \
           params[3] < 5.0 or \
           params[3] > 10.0 :
            # alpha_T in [-3, 0], log T_mid in [5, 10]
            logL = -np.inf
    else:
        if params[2] <= 0.0 or \
           params[2] > 2.0 or \
           params[3] < 4.0 or \
           params[3] > 10.0 :
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
        if args.verbose:
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
    if args.verbose:
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

        



###############
# Main script #
###############


# Exit immediately if given no input catalogs
if len(args.catalogs) == 0:
    exit(0)

# Read the catalogs, and do some organizing on them
catalogs = []
allfilters = []
ncl = 0
for cat in args.catalogs:

    # Read the data
    if args.cattype == "mock":
        data = reader_register['mock'].read(cat)
    elif args.cattype == "LEGUS":
        data = reader_register['LEGUS'].read(cat)
    else:
        raise ValueError("unknown catalog type {:s}".format(args.cattype))

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
if args.cattype == "LEGUS":
    ncl = clean_legus(catalogs, args.verbose)
    
# We're now done ingesting the input catalogs; print status if verbose
if args.verbose:
    print("Completed reading the following input catalogs:")
    for cat in catalogs:
        print("   "+cat["basename"]+":")
        print("      {:d} clusters".format(len(cat["phot"])))
        print("      filters: {:s}".format(repr(cat["filters"])))
        print("      filter combinations:")
        for p, f in zip(cat['phot_filterset'], cat['filtersets']):
            print("         {:s} ({:d} clusters)".format(
                repr(f), len(p)))

if args.verbose:
    print("Completed reading the classifications of clusters of the following input catalogs:")
    for cat in catalogs:
        print("   "+cat["basename"]+":")
        # print the number of clusters in each class defined in Adamo et al 2017. 
        Ncl1 = len(cat["phot"]["class"] == 1)
        Ncl2 = len(cat["phot"]["class"] == 1)
        Ncl3 = len(cat["phot"]["class"] == 1)
        print(" Class 1      {:d} clusters".format(len(cat["phot"]["class"])))
        print(" Class 2      {:d} clusters".format(len(cat["phot"])))
        print(" Class 3      {:d} clusters".format(len(cat["phot"])))
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
if args.verbose:
    print("Loading cluster_slug library data")
lib_all = read_cluster(args.libname,
                       photsystem=args.photsystem,
                       read_filters=allfilters) # here all filters should have all filters used in catalog data 

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

# Import pre-computed completeness functions for library.
for cat in catalogs:
    cat['libcomp'] = []
    for d in cat['filtersets_detect']:
        if '628c' in cat['basename'] : 
            if np.sum(d) == 5:
                cat['libcomp'].append(np.load('libngc628c_comp_13_Mar.npy'))
                keep = np.logical_or(keep,cat['libcomp'][-1] > 0.0)
                print("Loaded library for c with all 5 bands")
            elif np.sum(d) == 4:
                cat['libcomp'].append(np.load('libngc628cnoUV_4dim_comp_Mar_11.npy'))
                keep = np.logical_or(keep,cat['libcomp'][-1] > 0.0)
                print("Loaded library for c with 4 bands, no UV band")
            else :
                print('Completeness not valid')
                exit(0)
        elif '628e' in cat['basename']:
            if  np.sum(d) == 5:
                cat['libcomp'].append(np.load('libngc628e_comp_14_Mar.npy'))
                keep = np.logical_or(keep,cat['libcomp'][-1] > 0.0)
                print("Loaded library for e with all 5 bands")
            elif  np.sum(d) == 4:
                cat['libcomp'].append(np.load('libngc628enoUV_4dim_comp_14_Mar.npy'))
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
if args.verbose:
    print("Pruned library from {:d} to {:d} clusters".
          format(ncl_init, len(actual_mass)))

# Initialize a cluster_slug library for every catalog and filter set
if args.verbose:
    print("Initializing cluster_slug objects for input catalogs:")
for cat in catalogs:
    if args.verbose:
        print("   {:s}:".format(cat["basename"]))
    cat['cs'] = []
    for i in range(len(cat['filtersets'])):
        indx = []
        for f in cat['filtersets'][0]:
            indx.append(allfilters.index(f))
        if args.verbose:
            print("      filters {:s}...".
                  format(repr(cat["filtersets"][i])))
        field_list = ['id', 'actual_mass', 'time', 'form_time', 'A_V',
                      'phot_neb_ex', 'filter_names', 'filter_units']
        idx = cat['filtersets_detect'][i]
        keep = cat['libcomp'][i] > 0
        fields = [np.copy(cid[keep]), 
                  np.copy(actual_mass[keep]), 
                  np.copy(eval_time[keep]),
                  np.copy(form_time[keep]),
                  np.copy(A_V[keep]),
                  np.copy(phot_neb_ex[:,indx][:,idx][keep]),
                  list(np.array(cat['filters'])[idx]),
                  list(np.array(filter_units)[indx][idx])]
        lib_type = namedtuple('cluster_data', field_list)
        lib = lib_type(*fields)
        cat['cs'].append(
            cluster_slug(lib = lib,
                         sample_density = lib_den.sample_den,
                         reltol = args.tol,
                         bw_phot = args.bwphot,
                         bw_phys = args.bwphys))
        cat['cs'][-1].add_filters(lib.filter_names, 
                                  pobs=cat['libcomp'][i][keep])
        cat['cs'][-1].make_cache(range(3), filters=lib.filter_names)

# Construct name of output file if not specified
if args.outname is None:
    outname = osp.splitext(args.catalogs[0])[0] + ".chain"
else:
    outname = args.outname

# Set the initial walker positions; if this is a restart, read them
# from restart file, and if not start with an initial guess and
# disperse them around that
# Set the initial walker positions; if this is a restart, read them
# from restart file, and if not start with an initial guess and
# disperse them around that
ndim = args.nparam+args.nav   # Number of free parameters in priors
p0 = np.zeros((args.nwalkers, ndim))
if args.restart:
    nread = 0
    fp = open(outname, "r")
    chaindat = fp.read().split('\n')
    fp.close()
    while '' in chaindat:
        chaindat.remove('')
    for p, line in zip(p0, chaindat[-args.nwalkers-1:]):
        p[:] = [float(f) for f in line.split()[1:-1]]
    nread = len(chaindat) // args.nwalkers
    if args.verbose:
        print("Loaded {:d} iterations of MCMC chain from {:s}".
              format(nread, outname))
else:
    nread = 0
    p0[:,0] = -2.0 + 2.0*(rand(args.nwalkers)-0.5) # alphaM = -3 to -1
    p0[:,1] = 6.0 + 2.0*(rand(args.nwalkers)-0.5) # mBreak = 5 to 7 
    if not args.mdd:
        p0[:,2] = -1.0 + 2.0*(rand(args.nwalkers)-0.5) # alphaT = -2 to 0
        p0[:,3] = 7.0 + 2.0*(rand(args.nwalkers)-0.5)  # log Tmid = 6 - 8
    else:
        p0[:,2] = 0.5 + 0.5*(rand(args.nwalkers)-0.5)  # gammaMdd = 0.25 - 0.75
        p0[:,3] = 5.0 + 2.0*(rand(args.nwalkers)-0.5)  # log Tmddmin = 4 - 6
    delta_av = 3.0/args.nav   # p_AV scattered uniformly in log around 1/3
    for i in range(args.nwalkers):
        pav = 1./3.*(1.0 + 0.25*(rand(args.nav)-0.5))
        integ = delta_av * (0.5*pav[0] + np.sum(pav[1:]))
        if integ >= 1.0:
            pav = 0.999/integ * pav
        p0[i,args.nparam:] = np.log10(pav)
    if args.verbose:
        print("Initialized walkers")
    # # Open empty file for output
    filename = args.outname+".h5"
    backend = emcee.backends.HDFBackend(filename)   
    # fp = open(outname, 'w')
    # fp.close()
# Run the MCMC, saving periodically
if args.verbose:
    print("Starting MCMC")
sampler = emcee.EnsembleSampler(args.nwalkers, ndim, lnprob,backend=backend)
# for samp in sampler.sample(p0, iterations=args.niter-nread):
sampler.run_mcmc(p0,args.niter-nread)
for i, result in enumerate(sampler.sample(p0, iterations=args.niter-nread)):
    position = result.coords
    lnp = result.log_prob
    if args.verbose:
        print("Completed {:d} / {:d} iterations, saving state"
              .format(i+nread+1, args.niter))
    fp = open('./chains/test.chain', "a")
    for k in range(position.shape[0]):
        fp.write("{0:4d}".format(k))
        for j in range(position.shape[1]):
            fp.write("  {:f}".format(position[k,j]))
        fp.write("   {:f}".format(lnp[k]))
        fp.write("\n")
    fp.close()

