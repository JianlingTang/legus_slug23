"""
This is a little utility function that makes 2D PDF plots with
flanking histograms for grouped cluster properties comparison. 
"""
    
#####################################################################################
# import functions
from stack_tools import *
import os 
# set colormaps and fonts 
plt.rcParams['font.size'] = 12
plt.rc('font', family='serif')
re = colors.ListedColormap(['red'])
bl = colors.ListedColormap(['blue'])

# indicate filenames 
filename1 = os.path.join('./chains/'+'hlsp_628cmid_1stngc628_div_half_chain.h5')
filename2 = os.path.join('./chains/'+'hlsp_628cmid_2ndngc628_div_half_chain.h5')

# extract MCMC samples from files 
x,a4x,t= samp(filename1,param_i = 0)
y,f,t = samp(filename1, param_i = 1) 
m,a4m,t= samp(filename2, param_i = 0)
n,f,t = samp(filename2, param_i = 1)

# plot datasets using pdfplot function 
pdfplot(a4x,y,a4m,n,nxbin=35, nybin = 35, alphaxy = 0.9,alphamn = 0.45,\
        xyclim = [0,30], mnclim = [0,1.8],xlim = [-3.4,-0.9],ylim=[3.,7],\
        hist_fs=8,log=False,xycmap = re,mncmap = bl,xlabel=r'$\alpha_{M4}$',\
        ylabel=r'$\log (M_{\mathrm{break}}/M_{\odot})$',xystyle='contour',\
        mnstyle = 'imshow', label1=r'{$r \leq R_{50}$}', label2=r'{$r>R_{50}$}')

# save figure 
plt.savefig('Div_r50_am4Mb.pdf',bbox_inches='tight')