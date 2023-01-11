#!/usr/bin/env python3
import numpy as np
import scipy as sc 
import numexpr as ne
from astropy.table import Table
import scipy
from astropy.io import fits
from slugpy.cluster_slug import cluster_slug
from catalog_readers import reader_register 
from slugpy import slug_pdf, read_cluster
import emcee


# Parse the inputs
parser = argparse.ArgumentParser(
    description="Function to compute the completenss")
parser.add_argument("-nt", "--ntr", type=int, default=10000,
                    help="number of Monte Carlo trials")
parser.add_argument("--allf", default=True, action="store_true",
                    help="true if we are computing the completeness for all five filters")
args = parser.parse_args()

###############################################################        
# Data for LEGUS fields where artificial star tests have been #
# performed; citations are given below                        #
###############################################################        
        
LEGUS_field_data = {
    
    # Data for NGC 628c from Adamo et al., 2017, ApJ, 841, 131
    'ngc628c' : {
        'filterset' : ['WFC3_UVIS_F275W', 'WFC3_UVIS_F336W', \
                       'ACS_F435W', 'ACS_F555W', 'ACS_F814W' ],
        'phot_tab' : np.array([
            # F275W
            [20.2574257425742, 20.7590759075907, 21.2475247524752, 21.7491749174917,
            22.2508250825082, 22.7524752475247, 23.2541254125412, 23.7557755775577,
            24.2442244224422, 24.7458745874587, 25.2475247524752, 25.7491749174917],
            # F336W
            [20.25742574, 20.75907591, 21.24752475, 21.74917492, 22.25082508,
            22.75247525, 23.25412541, 23.75577558, 24.24422442, 24.74587459,
            25.24752475,25.74917492],
            # F435W
            [20.2574257425742,20.7590759075907,21.2475247524752,21.7491749174917,
            22.2508250825082,22.7524752475247,23.2541254125412,23.7425742574257,
            24.2442244224422,24.7458745874587,25.2475247524752,25.7491749174917],
            # F555W
            [20.2574257425742,20.7590759075907,21.2475247524752,21.7491749174917,
            22.2508250825082,22.7524752475247,23.2541254125412,23.7425742574257,
            24.2442244224422,24.7458745874587,25.2475247524752,25.7491749174917],
            # F814W
            [20.2574257425742,20.7590759075907,21.2475247524752,21.7491749174917,
            22.2508250825082,22.7524752475247,23.2541254125412,23.7425742574257,
            24.2442244224422,24.7458745874587,25.2475247524752,25.7491749174917]
        ]),

       
        'comp_tab' : np.array([
            # F275W
            [100,100,100,100,100,100,95.2727272727272,35.5454545454545,3.9090909090909,0,0,0],
            # F336W
           [100,100,100,100,100,100,100,97.18181818,74,17,0,0],
            # F435W
           [100,100,100,100,100,100,100,100,100,97.4545454545454,77.5454545454545,35.8181818181818],
            # F555W
           [100,100,100,100,100,100,100,100,100,100,82.7272727272727,32.8181818181818],
            # F814W
            [100,100,100,100,100,100,100,100,91.4545454545454,48.9090909090909,10.9999999999999,0.909090909090892]
         ] )
        },
    
    # Data for NGC 628e from Adamo et al., 2017, ApJ, 841, 131
    'ngc628e' : {
        'filterset' : ['WFC3_UVIS_F275W', 'WFC3_UVIS_F336W', \
                       'ACS_F435W', 'WFC3_UVIS_F555W', 'ACS_F814W' ],
        'phot_tab' : np.array([
              # F275W
              [20.2574257425742,20.7590759075907,21.2475247524752,21.7491749174917,
              22.2508250825082,22.7524752475247,23.2541254125412,23.7557755775577,
              24.2442244224422,24.7458745874587,25.2475247524752,25.7491749174917],
              # F336W
              [20.2574257425742,20.7590759075907,21.2475247524752,21.7491749174917,
              22.2508250825082,22.7524752475247,23.2541254125412,23.7425742574257,
              24.2442244224422,24.7458745874587,25.2475247524752,25.7491749174917],
              # F435W
              [20.2574257425742,20.7590759075907,21.2475247524752,21.7491749174917,
              22.2508250825082,22.7524752475247,23.2541254125412,23.7425742574257,
              24.2442244224422,24.7458745874587,25.2475247524752,25.7491749174917],
              # F555W
              [20.2574257425742,20.7590759075907,21.2475247524752,21.7491749174917,
              22.2508250825082,22.7524752475247,23.2541254125412,23.7425742574257,
              24.2442244224422,24.7458745874587,25.2475247524752,25.7491749174917],
              # F814W
              [20.2574257425742,20.7590759075907,21.2475247524752,21.7491749174917,
              22.2508250825082,22.7524752475247,23.2541254125412,23.7425742574257,
              24.2442244224422,24.7458745874587,25.2475247524752,25.7491749174917]

        ]),
        'comp_tab' : np.array([
            # F275W
            [100,100,100,100,100,100,100,32.8181818181818,2.54545454545451,0,0,0],
            # F336W
            [100,100,100,100,100,100,100,69.3636363636363,11.5454545454545,0,0,0],
            # F435W
            [100,100,100,100,100,100,100,100,100,95.2727272727272,91.1818181818181,30.090909090909],
            # F555W
            [100,100,100,100,100,100,100,100,100,100,82.7272727272727,32.8181818181818],     
            # F814W
            [100,100,100,100,100,100,100,100,100,59.2727272727272,15.3636363636363,0]
        ])            
    }
    }
class libcomp(object):
    def __init__(self,catname,allf=True,csdir):
        self.name = catname # this is catalog name needs to be specified (this script only supports "628c" and "628e")
        self.allf = allf # if compute the completness contains only four filters without UV band, set False, else set True
        self.csdir = csdir # csdir indicate the directory of the cluster_slug library data, e.g. tang_phot.fits/tang_phys.fits 
        if self.allf :
            print('computing completenss in all filters')
        else:
            print('computing completenss in LEGUS filters without the UV band')
        if '628c' in self.name:
            self.phot = LEGUS_field_data['ngc628c']['phot_tab']
            self.comp = LEGUS_field_data['ngc628c']['comp_tab']
            self.filterset = LEGUS_field_data['ngc628c']['filterset']
            self.libphot = read_cluster(self.csdir, read_filters = self.filterset).phot_neb_ex
            self.dist = 9.9e6 # 9.9Mpc Calzetti et al 2015 (AJ, 149, 51) 
        elif '628e' in self.name:
            self.phot = LEGUS_field_data['ngc628e']['phot_tab']
            self.comp = LEGUS_field_data['ngc628e']['comp_tab']
            self.filterset = LEGUS_field_data['ngc628e']['filterset']
            self.libphot = read_cluster('./cluster_slug/tang', read_filters = self.filterset).phot_neb_ex
            self.dist = 9.9e6
        else:
            print('completenss cannot be computed without tabulated data')
            exit(0)

    def completeness(self,Nt):
        self.Nt = Nt
        # first convert the data to absolute Magnitude 
        abs_phot = self.phot-5*(np.log10(self.dist)-1) # Formula to calculate the absolute magnitude
        self.comp[self.comp < 0.] = 0.
        self.comp[self.comp > 100.] = 100.
        complist = np.zeros(self.libphot.shape)
        for i in range(len(self.filterset)):
            comp_func = scipy.interpolate.interp1d(abs_phot[i],self.comp[i]/100)
            for j in range(len(self.libphot[:,0])):
                if min(abs_phot[i]) <= self.libphot[j,i] <= max(abs_phot[i]):
                    complist[j,i] = comp_func(self.libphot[j,i])
                elif self.libphot[j,i] < min(abs_phot[i]):
                    complist[j,i] = 1. # detection
                else:
                    complist[j,i] = 0. # non-detection 
        # Set up Markov Chain Monte Carlo experiment  
        prob_detected = np.zeros(complist.shape) # shape is the same (Nc,Nb)
        Nc = len(self.libphot[:,0]) # total number of library clusters 
        if self.allf : # If we are computing the completeness for filterset with all five bands 
            comp_l = np.zeros(Nc)
            for i in range(5):
                prob_detected[:,i] = complist[:,i]
            # start Monte Carlo Trials in series , 1000 as one batch 
            for i in range(self.Nt//1000):
                d_etect = np.zeros((1000, Nc, 5), dtype=bool) # set detect matrix 
                rand = np.random.rand(1000,Nc) # set random matrix 
                for i in range(5):
                    d_etect[:,:,i] = rand<prob_detected[:,i]
                in_v_and_i = np.logical_and( d_etect[:,:,3], d_etect[:,:,4] )     # Check V and I detection
                in_b_and_v = np.logical_and( d_etect[:,:,3], d_etect[:,:,2] )# Check V and B detection
                in_v_and_adjacent = np.logical_or( in_v_and_i, in_b_and_v) # Need to be in V and I or V and B
                in_all_bands = np.sum( d_etect, axis=2) >=4       # Sum over bands to get number of detections
                in_legus = np.logical_and(np.logical_and(self.libphot[:,3]<-6, in_v_and_adjacent),in_all_bands)    # Combine all LEGUS conditions
                comp_l += np.mean(in_legus, axis=0) # This is the completeness values for cluster classes 1, 2 and 3
                del d_etect
                del rand
                del in_legus
                del in_all_bands
            comp_l = comp_l/(self.Nt//1000)
        else : # If we are computing the completeness for filterset with only four filters: U (F336W), B (F435W), V (F555W), IR (F814W)
            comp_l = np.zeros(Nc)
            prob_detected[:,0] = np.zeros(len(prob_detected[:,0]))
            for i in range(4): # Match the column number to the number of filters 
                prob_detected[:,i+1] = complist[:,i+1]
            # start Monte Carlo Trials in series , 1000 as one batch 
            for i in range(self.Nt//1000):
                d_etect = np.zeros((1000, Nc, 5), dtype=bool) # set detect matrix 
                rand = np.random.rand(1000,Nc) # set random matrix 
                for i in range(5):
                    d_etect[:,:,i] = rand<prob_detected[:,i]
                # Imposing the LEGUS criteria 
                in_v_and_i = np.logical_and( d_etect[:,:,3], d_etect[:,:,4] ) 
                in_v_and_b = np.logical_and( d_etect[:,:,3], d_etect[:,:,2] ) 
                in_v_and_adjacent = np.logical_or( in_v_and_i, in_v_and_b) # Need to be in V and I or V and B
                in_four_bands = np.sum( d_etect, axis=2) >=4    # Sum over bands to get number of detections
                in_legus = np.logical_and(np.logical_and(self.libphot[:,3]<-6, in_v_and_adjacent),in_four_bands)    # Combine all LEGUS conditions
                comp_l += np.mean(in_legus, axis=0)# This is the completeness values for cluster classes 1,2 and 3
                del d_etect
                del rand
                del in_legus
                del in_four_bands
            comp_l = comp_l/(self.Nt//1000)
        # save completeness file in .npy format
        if self.allf:
            np.save('lib'+str(args.catalog)+'_comp',comp_l) 
        else :
            np.save('lib'+str(args.catalog)+'noUV_comp',comp_l)
        return None
        
class completeness_LEGUS(object):
    def __init__(self, catname,phot_filterset,Nt = 50000):
        """
        This instantiates a completeness object

        Parameters
           name : string
              the names of the observational catalogs 
          phot_filterset : array
              an array that contains boolean arrays of filterset detected from reading the input catalogs 
           Nf : int 
              number of filters of this filterset 
           Nt     : int 
              number of trials for artificial completeness test 
        """
        self.name = catname 
        self.Nt = Nt 
        self.phot_filterset = phot_filterset

        if '628c' in self.name:
            self.phot = LEGUS_field_data['ngc628c']['phot_tab']
            self.comp = LEGUS_field_data['ngc628c']['comp_tab']
            self.filterset = LEGUS_field_data['ngc628c']['filterset']
            self.dist = 9.9e6 # 9.9Mpc Calzetti et al 2015 (AJ, 149, 51) 
        elif '628e' in self.name:
            self.phot = LEGUS_field_data['ngc628e']['phot_tab']
            self.comp = LEGUS_field_data['ngc628e']['comp_tab']
            self.filterset = LEGUS_field_data['ngc628e']['filterset']
            self.dist = 9.9e6 # 9.9Mpc Calzetti et al 2015 (AJ, 149, 51) 
        else:
            print('unknown observational completenss!!')
            exit(0)

    def comp_LEGUS(self):
        # first convert the data to Absolute Magnitude 
        abs_phot = self.phot-5*(np.log10(self.dist)-1) # Formula to calculate the absolute magnitude
        self.comp[self.comp < 0.] = 0.
        self.comp[self.comp > 100.] = 100.
        comp_filterset = []
        for pf in range(len(self.phot_filterset)):
            # loop through all filtersets 
            allf = len(self.phot_filterset[pf][0]) == 5 # check if the filterset is 5 filters
            complistLEGUS= np.zeros(np.shape(self.phot_filterset[pf])) # 2D complist
            if allf :  
                for i in range(len(self.filterset)):
                    comp_func = scipy.interpolate.interp1d(abs_phot[i],self.comp[i]/100)
                    for j in range(len(self.phot_filterset[pf])):
                        if min(abs_phot[i]) <= self.phot_filterset[pf][j,i] <= max(abs_phot[i]):
                            complistLEGUS[j,i] = comp_func(self.phot_filterset[pf][j,i])
                        elif self.phot_filterset[pf][j,i] < min(abs_phot[i]):
                            complistLEGUS[j,i] = 1.
                        else:
                            complistLEGUS[j,i] = 0.
            else : 
                for i in range(len(self.filterset)-1):
                    comp_func = scipy.interpolate.interp1d(abs_phot[i],self.comp[i]/100)
                    for j in range(len(self.phot_filterset[pf])):
                        if min(abs_phot[i]) <= self.phot_filterset[pf][j,i] <= max(abs_phot[i]):
                            complistLEGUS[j,i] = comp_func(self.phot_filterset[pf][j,i])
                        elif self.phot_filterset[pf][j,i] < min(abs_phot[i]):
                            complistLEGUS[j,i] = 1.
                        else:
                            complistLEGUS[j,i] = 0.
            # next set up a Monte Carlo Simulation base case 
            prob_detected = np.zeros(complistLEGUS.shape) # shape is the same (Nc,Nb)
            Nc = len(self.phot_filterset[pf]) # total number of library clusters 
            if allf:
                comp_l = np.zeros(Nc)
                for i in range(5):
                    prob_detected[:,i] = complistLEGUS[:,i]
                # start Monte Carlo Trials in series , 1000 as one batch 
                for i in range(self.Nt//1000):
                    d_etect = np.zeros((1000, Nc, 5), dtype=bool) # set detect matrix 
                    rand = np.random.rand(1000,Nc) # set random matrix 
                    for i in range(5):
                        d_etect[:,:,i] = rand<prob_detected[:,i]
                    in_v_and_i = np.logical_and( d_etect[:,:,3], d_etect[:,:,4] )     # Check V and I detection
                    in_b_and_v = np.logical_and( d_etect[:,:,3], d_etect[:,:,2] )# Check V and B detection
                    in_v_and_adjacent = np.logical_or( in_v_and_i, in_b_and_v) # Need to be in V and I or V and B
                    in_all_bands = np.sum( d_etect, axis=2) >=4      # Sum over bands to get number of detections
                    in_legus = np.logical_and(np.logical_and(self.phot_filterset[pf][:,3]<-6, in_v_and_adjacent),in_all_bands) # Combine LEGUS conditions
                    comp_l += np.mean(in_legus, axis=0)# This is the comp for classes 1,2,3
                    del d_etect
                    del rand
                    del in_legus
                    del in_all_bands
                comp_l = comp_l/(self.Nt//1000)
                comp_filterset.append(comp_l)
            elif not allf:
                comp_l = np.zeros(Nc)
                for i in range(4):
                    prob_detected[:,i] = complistLEGUS[:,i]
                # start Monte Carlo Trials in series , 1000 as one batch 
                for i in range(self.Nt//1000):
                    d_etect = np.zeros((1000, Nc, 4), dtype=bool) # set detect matrix 
                    rand = np.random.rand(1000,Nc) # set random matrix 
                    for i in range(4):
                        d_etect[:,:,i] = rand<prob_detected[:,i]
                    in_v_and_i = np.logical_and( d_etect[:,:,-2], d_etect[:,:,-1] )     # Check V and I detection
                    in_b_and_v = np.logical_and( d_etect[:,:,-2], d_etect[:,:,-3] )# Check V and B detection
                    in_v_and_adjacent = np.logical_or( in_v_and_i, in_b_and_v) # Need to be in V and I or V and B
                    in_all_bands = np.sum( d_etect, axis=2) >=4        # Sum over bands to get number of detections
                    in_legus = np.logical_and(np.logical_and(self.phot_filterset[pf][:,-2]<-6, in_v_and_adjacent),in_all_bands)    # Combine all LEGUS conditions
                    comp_l += np.mean(in_legus, axis=0)# This is the comp for classes 1,2,3
                    del d_etect
                    del rand
                    del in_legus
                    del in_all_bands
                comp_l = comp_l/(self.Nt//1000)
                comp_filterset.append(comp_l)
            else:
                exit(0) 
        return comp_filterset
        
        

