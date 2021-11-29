#!/usr/bin/env python3
import numpy as np
from astropy.table import Table
import scipy
from astropy.io import fits
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib as mpl
from scipy import stats 
import pandas as pd
from slugpy.cluster_slug import cluster_slug
from slugpy import slug_pdf, read_cluster
import numexpr as ne
import os.path as osp
from astropy.io import ascii as asc
import emcee
from numpy.random import seed, rand
import numexpr as ne

# import the completeness function, see comp.py for details 
comp = np.loadtxt('kafc18lib_comp.txt')
nav = 6 

######INITAILIZATION######
lib = '/home/janett/slug2/cluster_slug/kafc18' # add your own library location as a string 

# Define a class of sample density built from the sampling PDFs, and has a
# function that takes as input an array of physical properties (physprop) from
# cluster_slug and returns the sample density gien physical
# properties. This class is used to specify the sampling
# density of our library given the PDFs. This will avoid the cluster_slug sampling
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
        
# compute the PDFs
massPDF = '/home/janett/slug2/cluster_slug/lib_mass.pdf'
agePDF = '/home/janett/slug2/cluster_slug/lib_time.pdf'
AVPDF = '/home/janett/slug2/cluster_slug/lib_av.pdf'
mass_pdf = slug_pdf(massPDF)
age_pdf = slug_pdf(agePDF)
av_pdf = slug_pdf(AVPDF)
# compute the library density required, output size : N*1 
lib_den = sample_den(mass_pdf, age_pdf, av_pdf)
# set filters sets follwing input photometry
filters = ['WFC3_UVIS_F275W', 'WFC3_UVIS_F336W', 'WFC3_UVIS_F438W',
           'WFC3_UVIS_F555W', 'WFC3_UVIS_F814W']

#prune the library to save memory and read clusters from the library  
lib_all = read_cluster(lib,
                       photsystem='Vega',
                       read_filters=filters)

# extract the useful fields then delete the rest
cid = lib_all.id
actual_mass = lib_all.actual_mass
form_time = lib_all.form_time
eval_time = lib_all.time
A_V = lib_all.A_V
phot_neb_ex = lib_all.phot_neb_ex
filter_names = lib_all.filter_names
filter_units = lib_all.filter_units
del lib_all



# delete the lib clusters if comp = 0.0 means the cluster with intrinsic luminosity L' observed in the same manner as observed cluster i will not be included in sample
ncl_init = len(actual_mass)
keep = np.where(comp>0)

cid = cid[keep]
actual_mass = actual_mass[keep]
eval_time = eval_time[keep]
form_time = form_time[keep]
A_V = A_V[keep]
phot_neb_ex = phot_neb_ex[keep]

from collections import namedtuple
field_list = ['id', 'actual_mass', 'time', 'form_time', 'A_V',
                      'phot_neb_ex', 'filter_names','filter_units']
fields = [cid, actual_mass, eval_time,form_time,A_V,phot_neb_ex,filters,filter_units]
lib_type = namedtuple('cluster_data', field_list)
lib = lib_type(*fields)
#make cluster_slug from lib (pruned) with cuts
cs = cluster_slug(lib = lib,sample_density = lib_den.sample_den,
                  reltol = 1.0e-2,
                  bw_phot = 0.05,
                  bw_phys = 0.05)
cs.add_filters(lib.filter_names, pobs=comp[keep])
cs.make_cache(range(3), filters=lib.filter_names)

##############################################################
# This is a specialization of the interface to the legus     #
# catalogs                                                   #
##############################################################
#Mode and mean of the classifications given by the visual classifiers; 
'''
0 = source was not visually classified (too faint); 
1 = symmetric, compact cluster; 
2 = concentrated object with some degree of asymmetry or color gradient; 
3 = diffuse or multiple peak system, possibly spurious alignment; 
4 = probable spurious detection (foreground/background source, single bright star, artifact.
'''
def LEGUS_read(fname):
    fmeta = osp.splitext(fname)[0]+'.tab'
    fp = open(fmeta, 'r')
    #get completeness from file 
    comp = np.loadtxt('comp_legus.txt')
    metadata = fp.read().splitlines()
    fp.close()
        # Extract data
    dmod = 29.98
    filters = ['WFC3_UVIS_F275W', 'WFC3_UVIS_F336W', 'WFC3_UVIS_F438W',
           'WFC3_UVIS_F555W', 'WFC3_UVIS_F814W']
        # Now read the main photometry data
    data = asc.read(fname)
    cid = np.array(data['col1'], dtype='int')
    nc = len(cid)
    nf = len(filters)
    phot = np.zeros((nc, nf))
    photerr = np.zeros((nc, nf))
    detect = np.ones((nc, nf), dtype=bool)
    for i in range(nf):
        phot[:,i] = data['col{:d}'.format(2*i+6)] - dmod
        photerr[:,i] = data['col{:d}'.format(2*i+7)]
        detect[:,i] = np.logical_and(data['col{:d}'.format(2*i+6)]<99.999,data['col{:d}'.format(2*i+6)] != 66.666) # Flag value
    ra = np.array(data['col4'])
    dec = np.array(data['col5'])
    classification = data['col34']
    #classification of 
    classcut=[0, 3.5]
    #excluding class zero clusters from legus 
    i = np.logical_and(classification > classcut[0], 
                             classification < classcut[1])
    cid = cid[i]
    phot = phot[i]
    photerr = photerr[i]
    detect = detect[i]
    ra = ra[i]
    dec = dec[i]

        # Package the output
    out = { "path"       : fname,
            "basename"   : osp.splitext(osp.basename(fname))[0],
            "cid"        : cid,
            "phot"       : phot,
            "photerr"    : photerr,
            "detect"     : detect,
            "filters"    : filters,
            "dmod"       : dmod,
            "ra"         : ra,
            "dec"        : dec
        }

        # Return
    return out


# Class used to set the weights on the library; there are effectively
# two versions, one for mass-independent disruption and one for
# mass-dependent disruption, but we combine them into the same class
# for simplicity
class libwgts(object):
    # this ois mid model 
    def __init__(self, p):
        self.alphaM = p[0]
        self.mBreak = 10.**p[1]
        self.alphaT = p[2]
        self.tMid   = 10.**p[3]

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
        alphaT = self.alphaT
        tMid = self.tMid
        logtMid = np.log10(tMid)
        # Get weight for M, T distributions
        wgt = ne.evaluate(
                "10.**((alphaM+1)*logm)*"
                "exp(-10.**logm/mBreak) * "
                "where( logt <= logtMid, "
                "       10.**logt/tMid, "
                "       (10.**logt/tMid)**(alphaT+1) )")
        # Add A_V weights
        for i in range(self.nav):
            avlo = self.av[i]
            avhi = self.av[i+1]
            pavlo = self.pav[i]
            pavhi = self.pav[i+1]
            avslope = (pavhi-pavlo) / (avhi-avlo)
            # Return final result
            wgt = ne.evaluate(
                "wgt * where( (av >= avlo) & (av < avhi),"
                "             pavlo + (av-avlo)*avslope,"
                "             1.0 )"
            )
        return wgt

inc = -1
indexlst=[]
# import the catalog file
cat =  'hlsp_legus_hst.tab'
# read the catalog and do some organizing on them
data = LEGUS_read(cat)

#select clusters observed in all 5 filters 
for i in data["detect"]:
    inc+=1
    if i.all()==True:
        indexlst.append(inc)
    else :
        continue 
phot=data["phot"][indexlst]
photerr=data["photerr"][indexlst]

#define lnprob used in emcee
def lnprob(params):

    global evalctr

    # Parameters are, in order:
    # 0 = M slope : dN/dlogM = -2 to 0
    # 1 =  T slope : dN/dlogT = -2 to 0
    # 2 =  log T_break = 6 - 8
    # 3: = log p(A_V) at points distributed at A_V = 0 - 3 mag
    
    #enforce limits on alphaM, alphaT and Av 
    wgts = libwgts(params)
    logL = 0.0
    # alpha_M in [-4, 0], log alpha_T in [-3, 0]
    if params[0] < -4. or \
       params[0] > -1. or \
       params[1] < 2. or \
       params[1] > 8. :
        logL = -np.inf
    if params[2] < -3. or \
           params[2] > 0. or \
           params[3] < 5. or \
           params[3] > 10.:
            # alpha_T in [-3, 0], log T_mid in [5, 10]
            logL = -np.inf
    if wgts.pav[-1] < 0.:
        # p(A_V) > 0 everywhere
        logL = -np.inf
        
    if logL == 0.:
        #set up cluster_slug object
        cs.priors = wgts.wgts
        #compute logL 
        logL+=np.sum(cs.logL(None, phot,photerr=photerr,filters =filters, margindim=range(3)))
  
    # Return log likelihood
    return logL
    
outname = osp.splitext('emcee_ngc628c')[0] + ".chain"
# Set the initial walker positions; if this is a restart, read them
# from restart file, and if not start with an initial guess and
# disperse them around that

#number of bins of A_v, we adopt N = 6, corresponding to breaking the extinction PDF into bins 0.5 mag wide K2019 
ndim = 4+nav  # Number of free parameters in priors, note that we have T slope and M slope to be 
nwalkers = 100 #trials for number of walkers 100 
p0 = np.zeros((nwalkers, ndim))
# default is false 
restart = False  
# restart method 
if restart==True:
    nread = 0
    fp = open(outname, "r")
    chaindat = fp.read().split('\n')
    fp.close()
    while '' in chaindat:
        chaindat.remove('')
    for p, line in zip(p0, chaindat[-nwalkers-1:]):
        p[:] = [float(f) for f in line.split()[1:-1]]
    nread = len(chaindat) // nwalkers
    print("Loaded {:d} iterations of MCMC chain from {:s}".
              format(nread, outname))
else:
#specify position 0 
    p0[:,0] = -2. + 2.0*(rand(nwalkers)-0.5) # alphaM = -3 to -1
    p0[:,1] = 6.0 + 2.0*(rand(nwalkers)-0.5)  # log Mbreak = 5 - 7
    p0[:,2] = -1.0 + 2.0*(rand(nwalkers)-0.5) # alphaT = -2 to 0
    p0[:,3] = 7.0 + 2.0*(rand(nwalkers)-0.5)  # log Tmid = 6 - 8
    delta_av = 3.0/nav   # p_AV scattered uniformly in log around 1/3
    for i in range(nwalkers):
        pav = 1./3.*(1.0 + 0.25*(rand(nav)-0.5))
        integ = delta_av * (0.5*pav[0] + np.sum(pav[1:]))
        if integ >= 1.0:
            pav = 0.999/integ * pav
        p0[i,4:] = np.log10(pav)

    # Open empty file for output
    fp = open(outname, 'w')
    fp.close()

# Run the MCMC, saving periodically
print("Starting MCMC")
#starting sampler 
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)



for i, result in enumerate(sampler.sample(p0, iterations=1000,storechain=False)):
    position = result[0]
    lnp = result[1]
    fp = open(outname, "a")
    for k in range(position.shape[0]):
        fp.write("{0:4d}".format(k))
        for j in range(position.shape[1]):
            fp.write("  {:f}".format(position[k,j]))
        fp.write("   {:f}".format(lnp[k]))
        fp.write("\n")

    fp.close()
