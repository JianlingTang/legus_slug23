
"""
This is a little utility function that makes 2d PDF plots with
flanking histograms.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from scipy.stats import binned_statistic_2d
from matplotlib.colors import LogNorm
from matplotlib import colors
from matplotlib.ticker import MaxNLocator 
from matplotlib.ticker import AutoMinorLocator
import emcee


def samp(filename, ndis=350, param_i=0, mdd=False):
    """
    post-processing function for MCMC chains.
    Parameters:
       filename : string
          input filename, ended with .chain or .h5
       ndis: int
          number of steps to be discarded 
       params_i : int
          index of the chosen parameter 
       mdd : boolean 
          whether the input file is modelled by MDD (Mass-Dependent Disruption) model 
       
    
    Returns
       if MDD model: 
           param_samp : array, shape (N,) 
               chosen parameter values array 
           samples_plot_t4 : array, shape (N,) 
              parameter values of t4 returned by MCMC 
           log_prob_samples : array, shape (N,)
               log posterior probabilities of samples returned by MCMC 
       if MID model :
           param_samp : array, shape (N,) 
               chosen parameter values array 
           samples_plot_a4 : array, shape (N,) 
              parameter values of alpha 4 returned by MCMC 
           log_prob_samples : array, shape (N,)
               log posterior probabilities of samples returned by MCMC 
    """
    
    reader = emcee.backends.HDFBackend('./' + filename)
    if 'mid' in filename :
        
        mdd = False 
    elif 'mdd' in filename :
        mdd = True
    else :
        print('Unknown model')
        exit(0)

    if not mdd:
        flat_samples = reader.get_chain(discard = ndis,flat=True)[:3500*1000]
        log_prob_samples = reader.get_log_prob(discard = ndis,flat=True)[:3500*1000]
        samples_plot = np.zeros((np.shape(flat_samples[:,0:4])[0],4))
        samples_plot[:,:4] = flat_samples[:,0:4]
        param_samp = samples_plot[:,param_i]
        samples_plot_a4 = flat_samples[:,0]-(4/flat_samples[:,1])
        print('a4 lower limit is,',-6,'and :',-0.5)
    else:
        flat_samples = reader.get_chain(discard = ndis,flat=True)[:5000*1000]
        log_prob_samples = reader.get_log_prob(discard = ndis,flat=True)[:5000*1000]
        idx_keep = np.logical_and(flat_samples[:,2] <=999.,flat_samples[:,2]>=-999.)
        samples_plot =  np.zeros((np.shape(flat_samples[:,0:4][idx_keep])[0],4))
        samples_plot[:,:4] = flat_samples[:,0:4][idx_keep]
        samples_plot = samples_plot[idx_keep]
        log_prob_samples = log_prob_samples[idx_keep]
        param_samp = samples_plot[:,param_i]
        samples_plot_t4 = (10**samples_plot[:,3]/((100/1e4)**samples_plot[:,2]))
        
    if not mdd:
        return param_samp, samples_plot_a4,log_prob_samples
    else :
        return param_samp[samples_plot_t4<1e9],\
    samples_plot_t4[samples_plot_t4<1e9],log_prob_samples[samples_plot_t4<1e9] 
    
def pdfplot(x, y, m,n, fignum=None, xlim=None, ylim=None, cen_loc = None,\
            alphaxy = 1 , alphamn = 1, xyclim = [1,2], mnclim = [20,30], \
            xystyle = 'contour',mnstyle='contour',xycmap = 'hot',mncmap='winter',\
            nxbin=80, nybin=80, nxticks= 4, nyticks = 3, thresh=None, threshfrac=0.96,\
            log=True,zmax=None, zscaling='normed', xhistlim=None, \
            yhistlim=None, colorbar=True, xlabel=None, ylabel=None, sigma = True, hist_fs = 6, \
            label1=None,label2=None , scat_alpha=1.0, aspect='auto'):
    """
    Makes a 2D PDF plot with flanking histograms.

    Parameters:
       x : array, shape (N,)
          x coordinates of points
       y : array, shape (N,)
          y coordinates of points
       m : array, shape (N,)
          m coordinates of points
       n : array, shape (N,)
          n coordinates of points
       fignum : int
          figure number in which to create the plot
       xlim : arraylike, shape (2,)
          plotting limits in x direction
       ylim : arraylike, shape (2,)
          plotting limits in y direction
       cen_loc : int 
          location of the labels of the central panel
       alphaxy : float
          alpha value of contour xy (the first contour)
       alphamn : float
          alpha value of contour mn (the second contour)
       xyclim : array, shape(2,)
          (min,max) of the first chosen color map (xy)
       mnclim : array, shape(2,) 
          (min,max) of the second chosen color map (mn)
       xystyle : string 
          plot style, options are 1) 'contour' 2) 'imshow' 
       mnstyle : string 
          same as xystyle 
       xycmap : string 
          color map name of the first dataset (xy)
       mncamp : string 
          same as xycmap, for the second dataset (mn)
       nxbin : int or arraylike, shape (N,)
          number of bins in x direction
       nybin : int or arraylike, shape (N,)
          number of bins in y direction
       nxticks : int
          number of ticks on the x axis of the 1D histogram on the right panel
       nyticks : int
          number of ticks on the y axis of 1D histogram on the top panel 
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
       nticks : int
          number of tick marks on the color bar
       xlabel : string
          labels for x dimension
       ylabel : string
          labels for y dimension
       scat_alpha : float
          alpha value for scatter plot points
       sigma : bool
          whether showing confidence intervals of z (68-95-99 rule)
       hist_fs : int
          histogram x, y axis labels fontsize
       label1 : string 
          label of the first dataset (xy)
       label2 : string 
          label of the second dataset (mn)
       scat_alpha : float lies within [0,1]
          scatter points alpha transparency value 
       aspect : string 
          figure aspect control 
       
    Returns
       Nothing
    """

    # Define the plotting grid
    if xlim is None:
        xlim = [min(np.amin(x),np.amin(m)), max(np.amax(x),np.amax(m))]
    if ylim is None:
        ylim = [min(np.amin(y),np.amin(n)), max(np.amax(y),np.amax(n))]
    xgrd = np.linspace(xlim[0], xlim[1], nxbin+1)
    ygrd = np.linspace(ylim[0], ylim[1], nybin+1)
    xgrd_h = 0.5*(xgrd[1:]+xgrd[:-1])
    ygrd_h = 0.5*(ygrd[1:]+ygrd[:-1])
    xx, yy = np.meshgrid(xgrd_h, ygrd_h)
    
    # Get 2D histogram; note that we have to handle the case of
    # inverted axis limits with care, because binned_statistic_2d
    # doesn't natively support them
    xlim1 = np.sort(xlim)
    ylim1 = np.sort(ylim)
    
    count_xy, xe, ye, binidx_xy \
        = binned_statistic_2d(x, y,
                              np.ones(x.shape),
                              statistic='sum',
                              bins=[nxbin, nybin],
                              range = [[float(xlim1[0]), float(xlim1[1])],
                                       [float(ylim1[0]), float(ylim1[1])]],
                              expand_binnumbers=True)
    count_mn, me, ne, binidx_mn \
        = binned_statistic_2d(m, n,
                              np.ones(m.shape),
                              statistic='sum',
                              bins=[nxbin, nybin],
                              range = [[float(xlim1[0]), float(xlim1[1])],
                                       [float(ylim1[0]), float(ylim1[1])]],
                              expand_binnumbers=True)
    if xlim[0] > xlim[1]:
        count_xy = count_xy[::-1, :]
        xe = xe[::-1]
        binidx_xy[0,:] = nxbin+1 - binidx_xy[0,:]
    if ylim[0] > ylim[1]:
        count_xy = count_xy[:, ::-1]
        ye = ye[::-1]
        binidx_xy[1,:] = nybin+1 - binidx_xy[1,:]    
    if xlim[0] > xlim[1]:
        count_mn = count_mn[::-1, :]
        xe = xe[::-1]
        binidx_mn[0,:] = nxbin+1 - binidx_mn[0,:]
    if ylim[0] > ylim[1]:
        count_mn = count_mn[:, ::-1]
        ye = ye[::-1]
        binidx_mn[1,:] = nybin+1 - binidx_mn[1,:]  

    # Set z
    if zscaling == 'max':
        z = count_xy / np.amax(count)
    elif zscaling == 'count':
        z_xy = count_xy
        z_mn = count_mn
    elif zscaling == 'frac':
        z = count / len(x)
    elif zscaling == 'density':
        z_xy = count_xy / np.abs((xe[1]-xe[0])*(ye[1]-ye[0]))
        z_mn = count_mn / np.abs((xe[1]-xe[0])*(ye[1]-ye[0]))
    elif zscaling == 'normed':
        z_xy = count_xy / count_xy.sum() / (np.ones((np.shape(count_xy))) * (xe[1]-xe[0])*(ye[1]-ye[0]))
        z_mn = count_mn / count_mn.sum() / (np.ones((np.shape(count_mn))) * (xe[1]-xe[0])*(ye[1]-ye[0]))
    # Set minima and maxima for 2D plot
    if zmax is None:
        if zscaling == 'max':
            zmax = 1.0
        else:
            zmax_xy = np.amax(z_xy)
            zmax_mn = np.amax(z_mn)
    if thresh is not None:
        zmin = thresh
    else:
        zsort_xy = np.sort(z_xy, axis=None)
        csum = np.cumsum(zsort_xy)
        csum = csum/csum[-1]
        zmin_xy = zsort_xy[np.argmax(csum > 1.0-threshfrac)]
        zsort_mn = np.sort(z_mn, axis=None)
        csum = np.cumsum(zsort_mn)
        csum = csum/csum[-1]
        zmin_mn = zsort_mn[np.argmax(csum > 1.0-threshfrac)]
    if log:
        zmin_xy = np.log10(zmin_xy)
        zmax_xy = np.log10(zmax_xy)
        zmin_mn = np.log10(zmin_mn)
        zmax_mn = np.log10(zmax_mn)

    # Take log if requested
    if log:
        if np.logical_or(np.amax(z_xy),np.amax(z_mn)) == 0.0:
            raise ValueError("cannot use log scale: no positive z values")
        z_xy[z_xy == 0] = 1.0e-6*np.amin(z_xy[z_xy > 0])
        z_xy[z_xy == 0] = np.amin(z_xy[z_xy > 0])
        z_xy = np.log10(z_xy)
        
        z_mn[z_mn == 0] = 1.0e-6*np.amin(z_mn[z_mn > 0])
        z_mn[z_mn == 0] = np.amin(z_mn[z_mn > 0])
        z_mn = np.log10(z_mn)
    # Plot contours with 1,2,3 sigmas percentages 
    if sigma:
        zlevels_xy = []
        zlevels_mn = []
        sigmas = [1-0.998,1-0.97, 1-0.84]
        for sigma_thres in sigmas:
            zsort_xy = np.sort(z_xy, axis=None)
            csum_xy = np.cumsum(zsort_xy)
            csum_xy = csum_xy/csum_xy[-1]
            z_xylevel = zsort_xy[np.argmax(csum_xy > sigma_thres)]
            zlevels_xy.append(z_xylevel)
            zsort_mn = np.sort(z_mn, axis=None)
            csum_mn = np.cumsum(zsort_mn)
            csum_mn = csum_mn/csum_mn[-1]
            z_mnlevel = zsort_mn[np.argmax(csum_mn > sigma_thres)]
            zlevels_mn.append(z_mnlevel)
    print(zlevels_xy)
    # Get indices of individual points to show
    flag_xy = np.logical_and.reduce((binidx_xy[0,:] > 0,
                                  binidx_xy[1,:] > 0,
                                  binidx_xy[0,:] <= count_xy.shape[0],
                                  binidx_xy[1,:] <= count_xy.shape[1]))
    scatteridx_xy = np.zeros(len(x), dtype=bool)
    scatteridx_xy[flag_xy] \
        = z_xy[binidx_xy[0,flag_xy]-1, binidx_xy[1,flag_xy]-1] < zmin_xy
    
    flag_mn = np.logical_and.reduce((binidx_mn[0,:] > 0,
                                  binidx_mn[1,:] > 0,
                                  binidx_mn[0,:] <= count_mn.shape[0],
                                  binidx_mn[1,:] <= count_mn.shape[1]))
    scatteridx_mn = np.zeros(len(m), dtype=bool)
    scatteridx_mn[flag_mn] \
        = z_mn[binidx_mn[0,flag_mn]-1, binidx_mn[1,flag_mn]-1] < zmin_mn
        
    # Set up plot
    fig = plt.figure(figsize=(3.5, 4), dpi=200)
    plt.clf()
    gs = gridspec.GridSpec(5, 5)
    
    axcen = plt.subplot(gs[1:, :4])
    # plot empty arrays to show contour labels
    axcen.plot([],[],color='r',lw=2,label = label1)
    axcen.plot([],[],color='b',lw=2,label = label2)
    axcen.legend(loc=cen_loc,frameon=False,fontsize=10)
#     Plot contour at threshhold
    zlevels_xy = [i for n, i in enumerate(zlevels_xy) if i not in zlevels_xy[:n]] 

    imgxy = axcen.contour(np.transpose(z_xy), levels= zlevels_xy,
              cmap =  xycmap, origin='lower', extend = 'both',vmin=xyclim[0],vmax=xyclim[1],
                   extent=[xlim[0], xlim[1], ylim[0], ylim[1]],alpha = alphaxy,linewidths=0.8)

    fmt = {}
    strs = ['']
    for l, s in zip(imgxy.levels, strs):
        fmt[l] = s
    imgmn = axcen.contour(np.transpose(z_mn), levels = zlevels_mn,
              cmap = mncmap, origin='lower',extend = 'both',vmin=1,vmax=2.8,
              extent=[xlim[0], xlim[1], ylim[0], ylim[1]],alpha = alphamn,linewidths=0.8)
    zlevels_mn  = [i for n, i in enumerate(zlevels_mn) if i not in zlevels_mn[:n]]
    fmt = {}
    strs = ['']
    
    axcen.yaxis.set_major_locator(MaxNLocator(4))
    axcen.yaxis.set_minor_locator(AutoMinorLocator(5))
    axcen.xaxis.set_major_locator(MaxNLocator(4))
    axcen.xaxis.set_minor_locator(AutoMinorLocator(5))
    axcen.tick_params(axis = 'both',which='both',direction='in')
    if xyclim == None or mnclim == None:
        print('Colorbar limits not set.') 
    else :
        None
        

        
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
        = np.histogram(x, bins = nxbin, range=xlim1)
    histy, ye \
        = np.histogram(y, bins = nybin, range=ylim1)
    histm, me \
        = np.histogram(m, bins = nxbin, range=xlim1)
    histn, ne \
        = np.histogram(n, bins = nybin, range=ylim1)
    if xlim[0] > xlim[1]:
        histx = histx[::-1]
        xe = xe[::-1]
    if ylim[0] > ylim[1]:
        histy = histy[::-1]
        ye = ye[::-1]
    if zscaling == 'max':
        histx = histx / float(np.amax(histx))
        histy = histy / float(np.amax(histy))
    elif zscaling == 'count':
        pass
    elif zscaling == 'frac':
        histx = histx / float(len(x))
        histy = histy / float(len(y))
    elif zscaling == 'density':
        histx = histx / np.abs(xe[1]-xe[0])
        histy = histy / np.abs(ye[1]-ye[0])
        histm = histm / np.abs(xe[1]-xe[0])
        histn = histn / np.abs(ye[1]-ye[0])
    elif zscaling == 'normed':
        histx = histx / np.diff(xe) / histx.sum()
        histy = histy / np.diff(ye) / histy.sum()
        histm = histm / np.diff(me) / histm.sum()
        histn = histn / np.diff(ne) / histn.sum()

    # Add flanking histograms
    if zscaling == 'max':
        label = 'Scaled PDF'
    elif zscaling == 'count':
        label = r'N'
    elif zscaling == 'frac':
        label = 'Fraction'
    elif zscaling == 'density':
        label = 'Density'
    elif zscaling == 'normed':
        label = 'PDF'
    
    axtop = plt.subplot(gs[0, :-1])
    axtop.bar(xe[:-1], histx, xe[1]-xe[0],
              facecolor='red', edgecolor='black',alpha=0.8)
    axtop.bar(me[:-1], histm, xe[1]-xe[0],
              facecolor='blue', hatch='/',edgecolor='black',alpha=0.4,label=label2)
    axtop.set_xlim(xlim)
    
    if xhistlim is not None:
        axtop.set_ylim(xhistlim)
    if log:
        axtop.set_yscale('log')
    axtop.set_xticklabels([])
    axtop.yaxis.set_major_locator(plt.MaxNLocator(nyticks))
    axtop.yaxis.set_minor_locator(AutoMinorLocator(5))
    axtop.xaxis.set_major_locator(MaxNLocator(7))
    axtop.xaxis.set_minor_locator(AutoMinorLocator(5))
    axtop.tick_params(axis = 'x',which='both',direction='out')
    axtop.tick_params(axis = 'y',which='both',direction='in')
    axtop.set_ylabel(label,fontsize = 12)
    axtop.yaxis.set_tick_params(labelsize=hist_fs)
    axright = plt.subplot(gs[1:,-1])
    axright.barh(ye[:-1], histy, ye[1]-ye[0], 
                 align='edge',
                 facecolor='red',
                 edgecolor='black',alpha=0.8,label=label1)
    axright.barh(ne[:-1], histn, ne[1]-ne[0], 
                 align='edge',
                 facecolor='blue', hatch='/',
                 edgecolor='black',alpha=0.4,label=label2)
    axright.set_ylim(ylim)
    
    if yhistlim is not None:
        axright.set_xlim(yhistlim)
    if log:
        axright.set_xscale('log')
        axright.set_yscale('log')
    axright.set_yticklabels([])
    axright.xaxis.set_major_locator(plt.MaxNLocator(nxticks))
    axright.set_xlabel(label,fontsize = 12)
    axright.xaxis.set_major_locator(plt.MaxNLocator(2))
    axright.xaxis.set_minor_locator(AutoMinorLocator(5))
    axright.yaxis.set_major_locator(MaxNLocator(5))
    axright.yaxis.set_minor_locator(AutoMinorLocator(5))
    axright.tick_params(axis = 'x',which='both',direction='in')
    axright.tick_params(axis = 'y',which='both',direction='out')
    axright.xaxis.set_tick_params(labelsize=hist_fs)
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
                label = 'log density'
            else:
                label = 'density'
        elif zscaling == 'normed':
            if log:
                label = 'log-normed'
            else:
                label = 'normed'

#         axcen.text(-1,7, imshowlb, bbox={'facecolor': 'white', 'pad': 5})
    # Return handles to axes
    return (axcen, axtop, axright)


