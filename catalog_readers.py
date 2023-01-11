import numpy as np
import os.path as osp
from astropy.io import ascii as asc

#############################################
# This defines the generic reader interface #
#############################################
class catalog_reader(object):
    """
    This is a class whose job is to read observational catalogs. It is
    a purely abstract class that defines the minimum set entires that
    the read function must return
    """
    def __init__(self):
        """
        The initializer does nothing
        """
        pass

    def read(self, fname):
        """
        Function to read a catalog

        Parameters
           fname : string
              name of the catalog file to read

        Returns
           cat : dict
              a dict containing the data from the catalog; the returned
              data are as follows:
                 path = full path to catalog file
                 basename = base name of catalog with extensions removed
                 cid = array(ncluster), array of cluster ID numbers
                 phot = (ncluster, nfilter) array of photometric values
                 photerr = array of photometric errors; same shape as phot
                 detect = array of bool indicating whether the photometric
                          value listed represents a detection or a 
                          non-detection in that band
                 filters = list of filter names
        """
        raise NotImplementedError(
            "completeness is an abstract class; "
            "implement a derived class")
            
                    
#############################################################
# This is a specialization of the interface to the test mock #
# catalogs                                                   #
##############################################################

class catalog_reader_mock(catalog_reader):
    """
    This is a specialization of the catalog_reader class to the format
    of the test mock catalogs.
    """
    def __init__(self):
        __doc__ == catalog_reader.__init__.__doc__
    def read(self, fname):
        __doc__ == catalog_reader.read.__doc__

        # Get base name
        basename = osp.splitext(osp.basename(fname))[0]

        # Read input photometric catalog; create dummy photerr and detect
        # arrays
        data = np.loadtxt(fname, comments="#")
        cid = data[:,0].astype('int')
        phot = data[:,1:6]
        photerr = data[:,6:11]
        detect = np.ones(phot.shape, dtype='bool')

        # Read names of filters; these are on the last comment line,
        # things not ending in _e
        fp = open(fname, "r")
        for line in fp:
            strline = line.strip()
            if len(strline) == 0: continue
            if strline[0] == "#":
                prevline = strline
                continue
            break
        fp.close()
        filters = prevline.lstrip("#").split()[1:-1]
        filters = [f for f in filters if not f.endswith("_e")]
        

        # Package the output
        out = { "path"       : fname,
                "basename"   : basename,
                "cid"        : cid,
                "phot"       : phot,
                "photerr"    : photerr,
                "detect"     : detect,
                "filters"    : filters }

        # Return
        return out



##############################################################
# This is a specialization of the interface to the LEGUS     #
# catalog format                                             #
##############################################################

class catalog_reader_LEGUS(catalog_reader):
    """
    This is a specialization of the catalog_reader class to the format
    of the test mock catalogs.
    """
    def __init__(self):
        __doc__ == catalog_reader.__init__.__doc__
    def read(self, fname, classcut=[0, 3.5]):
        """
        Function to read a catalog from LEGUS

        Parameters
           fname : string
              name of the catalog file to read; there must be an
              accompanying metadata file, with the same base name and
              the extension .dat, in the same directory; see below
           classcut : listlike (2)
              range of LEGUS classes to include in the returned catalog

        Returns
           cat : dict
              a dict containing the data from the catalog; the returned
              data are as follows:
                 path = full path to catalog file
                 basename = base name of catalog with extensions removed
                 cid = array(ncluster), array of cluster ID numbers
                 phot = (ncluster, nfilter) array of photometric values
                 photerr = array of photometric errors; same shape as phot
                 detect = array of bool indicating whether the photometric
                          value listed represents a detection or a 
                          non-detection in that band
                 filters = list of filter names
                 ra = array(ncluster) of cluster right ascensions
                 dec = array(ncluster) of cluster declinations
                 viscat = bool, True if this is a visually-inspected
                          catalog, false if it is an automated one
                 phot_tab = array(N) of apparent magnitudes at
                                 which the completeness has been
                                 measured
                 comp_tab = array(N, nfilter) of recovery fractions
                            for clusters of that magnitude in each
                            filter

        Notes
           The metadata file format is as follows. The 1st non-comment
           line contains the distance modulus to the target. Lines 2 -
           6 give the names of each filter used in the data
           file. Line 7 is either "visual" for a visually-inspected
           catalog or "auto" for an automatic catalog. Lines 8 to the
           end give the estimated observational completeness; on each
           line the first number is the apparent magnitude, and the
           remaining numbers are the recovery fractions for clusters
           of that apparent magnitude in the each filter.
        """

        # Read the metadata; this lists the distance modulus, filters,
        # construction method (visual or automated), and artifical
        # star test results for the file
        # fmeta = osp.splitext(fname)[0]+'.tab'
        # fp = open(fmeta, 'r')
        # metadata = fp.read().splitlines()
        # fp.close()
        # Extract data
        dmod = 29.98 # Calzetti et al 2015
        if '628c' in fname : 
            filters = ['WFC3_UVIS_F275W', 'WFC3_UVIS_F336W', \
                       'ACS_F435W', 'ACS_F555W', 'ACS_F814W' ]
        elif '628e' in fname :
            filters=  ['WFC3_UVIS_F275W', 'WFC3_UVIS_F336W', \
                       'ACS_F435W', 'WFC3_UVIS_F555W', 'ACS_F814W' ]
        else :
            print('No filters')
        viscat = True
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
            detect[:,i] = np.logical_and(data['col{:d}'.format(2*i+6)]< 99.999,data['col{:d}'.format(2*i+6)] != 66.666) # Flag value
        ra = np.array(data['col4'])
        dec = np.array(data['col5'])
        classification = np.array(data['col34'], dtype='int')

        
        #excluding class zero clusters from legus
        i = np.logical_and(classification > classcut[0],
                             classification < classcut[1])
        cid = cid[i]
        phot = phot[i]
        photerr = photerr[i]
        detect = detect[i]
        ra = ra[i]
        dec = dec[i]
        cla = classification[i]
        # Package the output
        out = { "path"       : fname,
            "basename"   :osp.splitext(osp.basename(fname))[0],
            "cid"        : cid,
            "phot"       : phot,
            "photerr"    : photerr,
            "detect"     : detect,
            "filters"    : filters,
            "dmod"       : dmod,
            "ra"         : ra,
            "dec"        : dec,
            "class"      :cla,
             "viscat"     : viscat
        }

        # Return
        return out
        
            
##############################################################
# This is a specialization of the interface to the LEGUS     #
# catalog with division format                               #
##############################################################

class catalog_reader_LEGUS_div(catalog_reader):
    """
    This is a specialization of the catalog_reader class to the format
    of the test mock catalogs.
    """
    def __init__(self):
        __doc__ == catalog_reader.__init__.__doc__
    def read(self, fname, classcut=[0, 3.5]):
        """
        Function to read a catalog from LEGUS

        Parameters
           fname : string
              name of the catalog file to read; there must be an
              accompanying metadata file, with the same base name and
              the extension .dat, in the same directory; see below
           classcut : listlike (2)
              range of LEGUS classes to include in the returned catalog

        Returns
           cat : dict
              a dict containing the data from the catalog; the returned
              data are as follows:
                 path = full path to catalog file
                 basename = base name of catalog with extensions removed
                 cid = array(ncluster), array of cluster ID numbers
                 phot = (ncluster, nfilter) array of photometric values
                 photerr = array of photometric errors; same shape as phot
                 detect = array of bool indicating whether the photometric
                          value listed represents a detection or a 
                          non-detection in that band
                 filters = list of filter names
                 ra = array(ncluster) of cluster right ascensions
                 dec = array(ncluster) of cluster declinations
                 viscat = bool, True if this is a visually-inspected
                          catalog, false if it is an automated one
                 phot_tab = array(N) of apparent magnitudes at
                                 which the completeness has been
                                 measured
                 comp_tab = array(N, nfilter) of recovery fractions
                            for clusters of that magnitude in each
                            filter

        Notes
           The metadata file format is as follows. The 1st non-comment
           line contains the distance modulus to the target. Lines 2 -
           6 give the names of each filter used in the data
           file. Line 7 is either "visual" for a visually-inspected
           catalog or "auto" for an automatic catalog. Lines 8 to the
           end give the estimated observational completeness; on each
           line the first number is the apparent magnitude, and the
           remaining numbers are the recovery fractions for clusters
           of that apparent magnitude in the each filter.
        """

        # Read the metadata; this lists the distance modulus, filters,
        # construction method (visual or automated), and artifical
        # star test results for the file
        fmeta = osp.splitext(fname)[0]+'.tab'
        fp = open(fmeta, 'r')
        metadata = fp.read().splitlines()
        fp.close()
        # Extract data
        dmod = 29.98 # Calzetti et al 2015
        if '628c' in fname : 
            filters = ['WFC3_UVIS_F275W', 'WFC3_UVIS_F336W', \
                       'ACS_F435W', 'ACS_F555W', 'ACS_F814W' ]
        elif '628e' in fname :
            filters=  ['WFC3_UVIS_F275W', 'WFC3_UVIS_F336W', \
                       'ACS_F435W', 'WFC3_UVIS_F555W', 'ACS_F814W' ]
        else :
            print('No filters')
        viscat = True
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
            detect[:,i] = np.logical_and(data['col{:d}'.format(2*i+6)]< 99.999,data['col{:d}'.format(2*i+6)] != 66.666) # Flag value
        ra = np.array(data['col4'])
        dec = np.array(data['col5'])
        classification = data['col34']


        
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
            "dec"        : dec,
             "viscat"     : viscat
        }

        # Return
        return out
        
##############################################
# Here we register the list of known readers #
##############################################

reader_register = {
    'mock' : catalog_reader_mock(),
    'LEGUS' : catalog_reader_LEGUS()
}
