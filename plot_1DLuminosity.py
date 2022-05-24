import scipy as sc 
import numpy as np
import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np
import matplotlib.pyplot as plt
from slugpy import read_cluster
from slugpy.cluster_slug import cluster_slug
from slugpy import slug_pdf, read_cluster
from collections import namedtuple

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
from slugpy import slug_pdf, read_cluster
import emcee
mass_pdf = slug_pdf('./cluster_slug/lib_mass.pdf')
age_pdf = slug_pdf('./cluster_slug/lib_time.pdf')
av_pdf = slug_pdf('./cluster_slug/lib_av.pdf')
lib_den = sample_den(mass_pdf, age_pdf, av_pdf)
from catalog_reader_l import reader_register
from completeness import comp_register
from clean_legus import clean_legus
import sys 
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
    mdd = False 
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
#  Exit immediately if given no input catalogs
# if len(args.catalogs) == 0:
#     exit(0)

# Read the catalogs, and do some organizing on them
catalogs = []
allfilters = []
filterslist = []
ncl = 0
catalog = ['hlsp_628c.tab','hlsp_628e.tab']
for cat in catalog:

    # # Read the data
    # if args.cattype == "mock":
    #     data = reader_register['mock'].read(cat)
    # elif args.cattype == "LEGUS":
    data = reader_register['LEGUS'].read(cat)
    # else:
    #     raise ValueError("unknown catalog type {:s}".format(args.cattype))
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

    # # Create a "completeness calculator" object for each filter set
    # comp = []
    # for f, d in zip(data['filtersets'], data['filtersets_detect']):
    #     if args.cattype == 'mock':
    #         comp.append(comp_register['mock'])
    # # try a simpler completeness function with linear interpolation 
    #     elif args.cattype == 'LEGUS':
    #         comp.append(comp_register['simple'])
    # data['comp'] = comp

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
mass_pdf = slug_pdf('./cluster_slug/lib_mass.pdf')
age_pdf = slug_pdf('./cluster_slug/lib_time.pdf')
av_pdf = slug_pdf('./cluster_slug/lib_av.pdf')
lib_den = sample_den(mass_pdf, age_pdf, av_pdf)

# Read the slug library
print("Loading cluster_slug library data")
lib_all = read_cluster('./cluster_slug/tang',photsystem='Vega',
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
                cat['libcomp'].append(np.load('./tabulated_comp/libngc628c_comp.npy'))
                keep = np.logical_or(keep,cat['libcomp'][-1] > 0.0)
                print("Loaded library for c with all 5 bands")
            elif np.sum(d) == 4:
                cat['libcomp'].append(np.load('./tabulated_comp/libngc628cnoUV_comp.npy'))
                keep = np.logical_or(keep,cat['libcomp'][-1] > 0.0)
                print("Loaded library for c with 4 bands, no UV band")
            else :
                print('Completeness not valid')
                exit(0)
        elif '628e' in cat['basename']:
            if  np.sum(d) == 5:
                cat['libcomp'].append(np.load('./tabulated_comp/libngc628e_comp.npy'))
                keep = np.logical_or(keep,cat['libcomp'][-1] > 0.0)
                print("Loaded library for e with all 5 bands")
            elif  np.sum(d) == 4:
                cat['libcomp'].append(np.load('./tabulated_comp/libngc628enoUV_comp.npy'))
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
# if args.verbose:
print("Pruned library from {:d} to {:d} clusters".
        format(ncl_init, len(actual_mass)))
print("Initializing cluster_slug objects for input catalogs:")
for cat in catalogs:
    print("   {:s}:".format(cat["basename"]))
    cat['cs'] = []
    for i in range(len(cat['filtersets'])):
        indx = []
        for f in cat['filtersets'][0]:
            indx.append(allfilters.index(f))
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
# class to bin library luminosities 
class Luminosity_bins:
    def __init__(self, params , mid = True,filter = None ,bins = None,range=None): # for mid model or mdd model 
        self.p = params
        self.mid = mid
        self.filter = filter 
        self.bins = bins 
        self.range = range 
        
    def Luminosity(self,catalogs):
        ncr = []
        for cat in catalogs:
            for phot in cat['phot_filterset']:
                ncr.append(np.shape(phot)[0])
        nt = np.sum(ncr)
        ncr = ncr/nt # number of clusters of each filterset 
        bins_sums = []
        bins_eds = [] 
        bins_sigma = []

        for i,cat in enumerate(catalogs):
            for j in range(len((cat['cs']))):
                index = [idx for idx, s in enumerate(cat['filtersets'][0]) if self.filter in s][0]
                dataset = cat['cs'][j]._cluster_slug__filtersets[0]['bp']._bp__dataset
                phys = dataset[:,0:3]
                pobs = cat['cs'][j]._cluster_slug__filtersets[0]['bp']._bp__pobs
                libwgt = libwgts(self.p,mid = self.mid).wgts(phys[pobs>0.])
                if j == 1 and index == 0 : 
                    print(str(j)+str(index))
                    continue 
                else : 
                    if index == 2: # if filterset is F435W, change to F555W with index -2
                        hist, bin_ed = np.histogram(dataset[:,2-5][pobs>0.],bins=self.bins,\
                        weights = lib_den.sample_den(phys[pobs>0.])*libwgt*pobs[pobs>0.],density=True)
                        sigma = np.sqrt(stats.binned_statistic(dataset[:,2-5][pobs>0.],(lib_den.sample_den(phys[pobs>0.])*libwgt*pobs[pobs>0.])**2,\
                        statistic='sum',\
                        bins=self.bins)[0])
                    elif index == 3 : # if filterset is F555W, change to F435W with index = -3
                        hist, bin_ed = np.histogram(dataset[:,3-5][pobs>0.],bins=self.bins,\
                        weights = lib_den.sample_den(phys[pobs>0.])*libwgt*pobs[pobs>0.],density=True)
                        sigma = np.sqrt(stats.binned_statistic(dataset[:,3-5][pobs>0.],(lib_den.sample_den(phys[pobs>0.])*libwgt*pobs[pobs>0.])**2,\
                        statistic='sum',\
                        bins=self.bins)[0])
                    else : 
                        hist, bin_ed = np.histogram(dataset[:,index-5][pobs>0.],bins=self.bins,\
                        weights = lib_den.sample_den(phys[pobs>0.])*libwgt*pobs[pobs>0.],density=True)
                        sigma = np.sqrt(stats.binned_statistic(dataset[:,index-5][pobs>0.],(lib_den.sample_den(phys[pobs>0.])*libwgt*pobs[pobs>0.])**2,\
                    statistic='sum',\
                    bins=self.bins)[0])
                bins_sums.append(hist)
                bins_eds.append(bin_ed)
                bins_sigma.append(sigma)
        

        all_bin_sum = []
        all_bin_sigma = []

        for i in range(len(bins_sums)):
            all_bin_sum.append(bins_sums[i]*ncr[i])
            all_bin_sigma.append(bins_sigma[i]*ncr[i])
            
 
        all_bin_sum = np.sum(all_bin_sum,axis=0)
        all_bin_sigma = np.sum(all_bin_sigma,axis=0)
        return bins_eds, all_bin_sum , all_bin_sigma

# loop through two catalogs and 2 filtersets 4* 
r_l = [[-11.042000000000002, -4.036999999999999],[-10.872, -4.91],[-10.21, -6.030000000000001],[-9.674, -5.376999999999999],[-10.696000000000002, -6.042999999999999]]
def bin_data(catalogs,index):
    hist_l = []
    bins_l = []
    nc_l = []
    std_l=[]
    for i ,cat in enumerate(catalogs):
        for j in range(len(cat['phot_filterset'])):
            # for every filterset in each catalog 
            try : 
                nc_l.append(len(cat['phot_filterset'][j][:,index]))
                hist, bins = np.histogram(cat['phot_filterset'][j][:,index],bins=30,range=r_l[index],density=True)
                std = np.sqrt(hist)
                # range_l.append([np.min(cat['phot_filterset'][j][:,index]),np.max(cat['phot_filterset'][j][:,index])])
                # print([np.min(cat['phot_filterset'][j][:,index]),np.max(cat['phot_filterset'][j][:,index])])
                
            except IndexError:
                None 
            # hist, bins = np.histogram(cat['phot_filterset'][j][:,-5],range = [-20,-1],bins=200,density=True)
            hist_l.append(hist)
            bins_l.append(bins)
            std_l.append(std)
            # range_l.append([np.min(cat['phot_filterset'][0][:,index]),np.max(cat['phot_filterset'][0][:,index])])
    hist_all = []
    std_all = []
    sumc = np.sum(nc_l)
    print(sumc)
    for i in range(len(hist_l)):
        try : 
            hist_all.append(hist_l[i]*nc_l[i]/sumc)
            std_all.append(std_l[i]*nc_l[i]/sumc)
        except IndexError:
            None 
    # hist_all = np.sum(hist_all)
    hist_all = np.sum(hist_all,axis=0)
    std_all = np.sum(std_all,axis=0)
    return hist_all,bins, std_all
all_heights = np.zeros((10000,5,30))
for k in range(100):
    data_l =[]
    for i in range(5):
        d = bin_data(catalogs,i)
        data_l.append(d)
        all_heights[k,i,:] = d[0] # store 
    
# compute median for 100 trials 
median = np.median(all_heights,axis=0)
lower_err = np.percentile(all_heights,95,axis=0)
upp_err = np.percentile(all_heights,5,axis=0)
nc = 0 
import matplotlib.pyplot as plt 
plt.rc('font', family='serif')

print('The total number of clusters in 628c and 628e is {:}'.format(nc))

phot_all = np.zeros(nc)
all_dicts = []
# set the filtername 
allfs = ['F275W','F336W','F435W','F555W','F814W']
color = ['gold','cyan','green','orange','red']
mod = ['MID']
bol = [True]

params = np.array([-0.99927755,  3.85746251, -0.43310727,  6.75511899, -0.10067485,
       -0.31077786, -0.47420001, -0.87471347, -1.91308959, -0.31790502])
count = 0 

for k in range(len(allfs)):
    index = [idx for idx, s in enumerate(cat['filtersets'][0]) if allfs[k] in s][0]
    L_class = Luminosity_bins( params, mid = True, filter = allfs[k],bins=600,range=[-15,-3.5])
    ed, b, sigma = L_class.Luminosity(catalogs)
    dicts = {"filter" : allfs[k], # specify filter name
                 "mid_bool" : True, # whether the model used is MID (mass-independent mmodel)
                "model_edges" : ed ,
                "model_bsums" : b ,
                "model_std" : sigma}

        # append dictionary for further use 
    all_dicts.append(dicts)

# plot 1D luminosity 
allfs = ['F275W','F336W','F435W','F555W','F814W']
color = ['blue','cyan','green','brown','red']

fig, axs = plt.subplots(5,sharex=True,sharey=True,figsize=(10,12),dpi=50)
fig.text(0.5,0.1, "magnitude[mag]", ha="center", va="center")
fig.text(0.05,0.5, "log(Normalised Luminosity Distribution)", ha="center", va="center", rotation=90)

for i in range(len(all_dicts)):
    axs[i].plot(0.5*(all_dicts[i]["model_edges"][0][1:]+all_dicts[i]["model_edges"][0][:-1]), all_dicts[i]['model_bsums'],label='MID',linewidth=1.5,color='black')
    axs[i].fill_between(0.5*(all_dicts[i]["model_edges"][0][1:]+all_dicts[i]["model_edges"][0][:-1]), all_dicts[i]['model_bsums']-\
    3.*np.array(all_dicts[i]['model_std']),all_dicts[i]['model_bsums']+3.*np.array(all_dicts[i]['model_std']),color='grey', alpha=0.1,label = r'99% CI, $\mu \pm 3\sigma$')
    axs[i].fill_between(0.5*(all_dicts[i]["model_edges"][0][1:]+all_dicts[i]["model_edges"][0][:-1]), all_dicts[i]['model_bsums']-\
    2.*np.array(all_dicts[i]['model_std']),all_dicts[i]['model_bsums']+2.*np.array(all_dicts[i]['model_std']),color='orange', alpha=0.2,label = r'96% CI, $\mu\pm 2\sigma$')
    axs[i].fill_between(0.5*(all_dicts[i]["model_edges"][0][1:]+all_dicts[i]["model_edges"][0][:-1]), all_dicts[i]['model_bsums']-\
    all_dicts[i]['model_std'], all_dicts[i]['model_bsums']+all_dicts[i]['model_std'],color='green', alpha=0.25,  label = r'68% CI, $\mu \pm\sigma$')
    if i == 0 :
        axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
          fancybox=True, shadow=True, ncol=4)
    # calcculate data binning errors
    axs[i].text(-6.3,0.003,allfs[i], ha="center", va="center",color=color[i], fontsize=15)
    upp_er = 1.96*np.sqrt(np.array(data_l[i][0])*(1-data_l[i][0])/1178)
    asymmetric_error = [upp_er,upp_er]
    print(asymmetric_error)
    axs[i].errorbar(0.5*(data_l[i][1][1:]+data_l[i][1][:-1]),data_l[i][0],yerr=asymmetric_error,fmt = '*',color=color[i])
    # plt.scatter(0.5*(data_l[i][1][1:]+data_l[i][1][:-1]),data_l[i][0],s=15,marker='*',c=color[i])
    axs[i].set_yscale('log')
    # plt.legend()
    axs[i].set_xlim(np.min(0.5*(data_l[i][1][1:]+data_l[i][1][:-1]))-1,np.max(0.5*(data_l[i][1][1:]+data_l[i][1][:-1]))+0.6)
    axs[i].set_ylim(1e-4,1)
    plt.show()
plt.savefig('final_L.png',dpi=300)