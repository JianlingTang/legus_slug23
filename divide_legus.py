'''
a tool for galactic coordinate transformation and catalog division
given the division radii of chioce.  
'''
import scipy as sc 
import numpy as np
import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from slugpy import read_cluster
from slugpy.cluster_slug import cluster_slug
from slugpy import slug_pdf, read_cluster
from collections import namedtuple
from catalog_readers import reader_register
from completeness_calculator import *
from clean_legus import clean_legus
import astropy.constants as c 
import astropy.units as u
import astropy 
from astropy.io import ascii
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename




def divide(fitsname,data,Rcr_pix):
    # first import fits files and .tab with cluster RA/DEC coordinates 
    # 'hlsp_legus_hst_acs-wfc3_ngc628-mosaic_f555w_v1_sci.fits'
    f = fits.open(fitsname)
    mywcs = WCS(f[0].header)
    x , y = 6763, 6354 # ned center coorinates from NED 
    ra_cen, dec_cen = mywcs.all_pix2world([[x, y]], 0)[0] 
    # this is the ra and dec of the galactic centre  
    # convert all ra dec of different pointings to ra and dec, 
    # then transfrom to this pixel frame using all_world2pix 
    all_coor = []
    for i in range(len(data['ra'])):
        all_coor.append([data['ra'][i],data['dec'][i]])
    cl_dec_frame = mywcs.all_world2pix(all_coor,1)
    gal_coor = cl_dec_frame - [x,y]
    # use Li et al 2020 to deproject and rotate the galaxy image 
    PA = 25.*u.deg # position angle deg (Grasha et al 2015)
    inc = 25.2*u.deg # inclination angle deg of ngc628 (Grasha et al 2015)
    new_x = np.cos(PA*np.pi/(180.))*gal_coor[:,0] + np.sin(PA*np.pi/(180.))*gal_coor[:,1]
    new_y = -np.sin(PA*np.pi/(180.))/np.cos(inc*np.pi/(180.))*gal_coor[:,0] + np.cos(PA*np.pi/(180.))/np.cos(inc*np.pi/(180.))*gal_coor[:,1]
    sep = np.sqrt(new_x**2+new_y**2)
    # store divivded data 
    data_Rcr = []
    idx_f = sep < Rcr_pix
    idx_s = sep >= Rcr_pix

    data_Rcr.append({ "path"       : data['path'],
            "basename"   : data['path'],
            "cid"        : data['cid'][idx_f],
            "phot"       : data['phot'][idx_f,:],
            "photerr"    : data['photerr'][idx_f,:],
            "detect"     : data['detect'][idx_f,:],
            "filters"    : data['filters'],
            "dmod"       : data['dmod'],
            "ra"         : data['ra'][idx_f],
            "dec"        : data['dec'][idx_f],
             "viscat"     : True
        })
    data_Rcr.append({ "path"       : data['path'],
            "basename"   : data['path'],
            "cid"        : data['cid'][idx_s],
            "phot"       : data['phot'][idx_s,:],
            "photerr"    : data['photerr'][idx_s,:],
            "detect"     : data['detect'][idx_s,:],
            "filters"    : data['filters'],
            "dmod"       : data['dmod'],
            "ra"         : data['ra'][idx_s],
            "dec"        : data['dec'][idx_s],
             "viscat"     : True
        })
    return data_Rcr 
