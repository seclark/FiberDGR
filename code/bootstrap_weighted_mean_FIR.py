import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import matplotlib
from astropy.io import fits
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from matplotlib.path import Path
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import copy
import time
#import cmocean
#import colorcet as cc
#import colorsys
import os.path
import h5py

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

# local repo imports
import sys
sys.path.insert(0, '../../GalfaCuber/code')
import galfa_vel_helpers as gvh

sys.path.insert(0, '../../FITSHandling/code')
import cutouts

LOCAL = False

def gaussian_umask(data, fwhm=2, zeroed=False):
    """
    fwhm in arcmin aka pixels
    """
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    smoothdata = ndimage.filters.gaussian_filter(data, sigma=sigma)
    
    umask = data - smoothdata
    if zeroed:
        umask[np.where(umask < 0)] = 0
    return umask

def load_lats():
    """
    load a map of all b values in GALFA-HI footprint 
    """
    bees = np.load("../all_galactic_latitudes_galfanhi.npy")
    
    return bees

def load_pointsources():
    """
    load a map of Planck point sources, projected onto GALFA footprint
    """
    if LOCAL:
        pointsourcemask = fits.getdata("/Users/susanclark/Dropbox/Planck/HFI_Mask_PointSrc_2048_R2.00_ONGALFAHI.fits")
    else:
        pointsourcemask = fits.getdata("/data/seclark/Planck/HFI_Mask_PointSrc_2048_R2.00_ONGALFAHI.fits")
    
    return pointsourcemask
    
def load_bootstrap_data(vstart="1024", vstop="1024", bstart=30, NHIcut=True, NHImin=0, NHImax=20, Narrow=False, Nblocks=50, Nsamples=10000000, submonopole=None, bstype="radec"):
    BS_root = "/Users/susanclark/Projects/FiberDGR/data/bootstrap_data/"
    
    if Narrow:
        narrowstr = "Narrow_"
    else:
        narrowstr = ""
        
    bstartstr = "_bstart{}".format(bstart)
        
    if NHIcut:
        NHImaxstr = "_NHImin{}_max{}".format(NHImin, NHImax)
    else:
        NHImaxstr = ""
        
    if submonopole is not None:
        submonopolestr = "_submono{}".format(submonopole)
    else:
        submonopolestr = ""
    
    print( BS_root+'BS_meanNHI_vel{}_to_{}_{}Nblocks{}_Nsamples{}{}{}{}_bstype{}.npy'.format(vstart, vstop, narrowstr, Nblocks, Nsamples, bstartstr, NHImaxstr, submonopolestr, bstype) )
    BS_meanNHI = np.load(BS_root+'BS_meanNHI_vel{}_to_{}_{}Nblocks{}_Nsamples{}{}{}{}_bstype{}.npy'.format(vstart, vstop, narrowstr, Nblocks, Nsamples, bstartstr, NHImaxstr, submonopolestr, bstype))
    BS_meanP857 = np.load(BS_root+'BS_meanP857_vel{}_to_{}_{}Nblocks{}_Nsamples{}{}{}{}_bstype{}.npy'.format(vstart, vstop, narrowstr, Nblocks, Nsamples, bstartstr, NHImaxstr, submonopolestr, bstype))
    BS_weightedmeanNHI = np.load(BS_root+'BS_weightedmeanNHI_vel{}_to_{}_{}Nblocks{}_Nsamples{}{}{}{}_bstype{}.npy'.format(vstart, vstop, narrowstr, Nblocks, Nsamples, bstartstr, NHImaxstr, submonopolestr, bstype))
    BS_weightedmeanP857 = np.load(BS_root+'BS_weightedmeanP857_vel{}_to_{}_{}Nblocks{}_Nsamples{}{}{}{}_bstype{}.npy'.format(vstart, vstop, narrowstr, Nblocks, Nsamples, bstartstr, NHImaxstr, submonopolestr, bstype))
    
    BS_deltaFIR = BS_weightedmeanP857 - BS_meanP857
    perc16 = np.percentile(BS_deltaFIR, 16)
    perc84 = np.percentile(BS_deltaFIR, 84)
    
    print(perc16, perc84)
    
    return perc16, perc84
        
def make_mask_2d(bstart=30, bstop=90, PS=False, NHIcut=False, NHImin=0, NHImax=20, bootstrapchunks=False, bsnum=0, writemap=False, bstype="raonly", negmask=None):
    ny = 2432
    nx = 21600
    mask = np.ones((ny, nx), np.int_)
    
    # latitude cuts
    bees = load_lats()
    if bstart > 0:
        mask[np.where(np.abs(bees) < bstart)] = 0
    if bstop < 90:
        mask[np.where(np.abs(bees) > bstop)] = 0
        
    if PS:
        PSmask = load_pointsources()
        mask[np.where(PSmask < 0.5)] = 0
        
    if negmask is not None:
        mask[np.where(negmask < 0)] = 0
        mask[np.where(np.isnan(negmask) == True)] = 0
        
    if NHIcut:
        if LOCAL:
            nhi90map = fits.getdata("/Users/susanclark/Dropbox/NHImaps/GALFA-HI_NHISRCORR_VLSR-90+90kms.fits")
        else:
            nhi90map = fits.getdata("/data/seclark/GALFADR2/NHImaps/GALFA-HI_NHISRCORR_VLSR-90+90kms.fits")

        nhi90map_div20 = nhi90map/1.E20
        mask[np.where(nhi90map_div20 > NHImax)] = 0
        mask[np.where(nhi90map_div20 < NHImin)] = 0
        mask[np.where(np.isnan(nhi90map_div20) == True)] = 0
        
    if bootstrapchunks is not False:
        # This code splits the data up into roughly even chunks of the pre-bootstrap mask.
        if bstype == "raonly":
            mask1d = np.nansum(mask, axis=0)
            allxcuts = np.zeros(bootstrapchunks+1)
            for _n in np.arange(bootstrapchunks+1):
                allxcuts[_n] = np.argmin(np.abs(np.cumsum(mask1d) - _n*np.nansum(mask)/bootstrapchunks))

            startblockx = np.int(allxcuts[bsnum])
            stopblockx = np.int(allxcuts[bsnum+1])
            if bsnum == bootstrapchunks - 1:
                mask[:, :startblockx] = 0
            else:
                mask[:, :startblockx] = 0
                mask[:, stopblockx:] = 0
        
        # splits into 2 DEC chunks and bootstrapchunks/2 RA chunks
        elif bstype == "radec":
            mask1d_split0 = np.nansum(mask[:np.int(ny/2), :], axis=0)
            mask1d_split1 = np.nansum(mask[np.int(ny/2):, :], axis=0)
            allxcuts = np.zeros(bootstrapchunks+1)
            for _n in np.arange(bootstrapchunks+1):
                if _n < bootstrapchunks/2:
                    #allxcuts[_n] = np.argmin(np.abs(np.cumsum(mask1d_split0) - _n*np.nansum(mask)/bootstrapchunks))
                    allxcuts[_n] = np.argmin(np.abs(np.cumsum(mask1d_split0) - _n*np.nansum(mask[:np.int(ny/2), :])/(bootstrapchunks/2)))
                else:
                    #allxcuts[_n] = np.argmin(np.abs(np.cumsum(mask1d_split1) - (_n - bootstrapchunks/2)*np.nansum(mask)/bootstrapchunks))
                     allxcuts[_n] = np.argmin(np.abs(np.cumsum(mask1d_split1) - (_n - bootstrapchunks/2)*np.nansum(mask[np.int(ny/2):, :])/(bootstrapchunks/2)))
             
            startblockx = np.int(allxcuts[bsnum])
            stopblockx = np.int(allxcuts[bsnum+1])
            if (bsnum == bootstrapchunks - 1) or (bsnum == np.int(bootstrapchunks/2 - 1)):
                mask[:, :startblockx] = 0
            else:
                mask[:, :startblockx] = 0
                mask[:, stopblockx:] = 0
            if bsnum < bootstrapchunks/2:
                mask[np.int(ny/2):, :] = 0
            else:
                mask[:np.int(ny/2), :] = 0
            

    if writemap:
        if LOCAL:
            outhdr = fits.getheader("/Users/susanclark/Dropbox/Planck/HFI_Mask_PointSrc_2048_R2.00_ONGALFAHI.fits")
            if NHIcut:
                outfn = "/Users/susanclark/Dropbox/Planck/mask_GALFA_footprint_absb_gt_{}_lt_{}_NHImin_{}max_{}_HFI_PS_{}.fits".format(bstart, bstop, NHImin, NHImax, PS)
            else:
                outfn = "/Users/susanclark/Dropbox/Planck/mask_GALFA_footprint_absb_gt_{}_lt_{}_HFI_PS_{}.fits".format(bstart, bstop, PS)
                
            fits.writeto(outfn, mask, outhdr)
        else:
            print("Not writing out the file.")
        
    return mask
    
def lat_cut(data, bstart=30, bstop=90):
    bees = load_lats()
    data[np.where(np.abs(bees) < bstart)] = 0 
    data[np.where(np.abs(bees) > bstop)] = 0
    
    return data   
    
def get_USM_slice(vels=["1024"], fwhm=10, zeroed=True, Narrow=False, reverse=False, writemap=False, returnrawslice=False, addnoise=False, noisefactor=1):
    if addnoise:
        noisemap = fits.getdata("/Users/susanclark/Dropbox/DR2 Full Sky Wide Maps/GALFA_HI_W_S1567_V0400.1kms.fits")
        
    if Narrow:
        if LOCAL:
            DR2_Narrow_slice_root = "/Users/susanclark/Dropbox/DR2_Full_Sky_Narrow_Maps/"
        else:
            DR2_Narrow_slice_root = "/data/seclark/GALFADR2/Narrow_maps/"
        DR2_Narrow_vels = np.loadtxt(DR2_Narrow_slice_root+"GALFA-HI_vlsr_Narrow.txt")
        
        vels_Narrow_dict = {"1024": "0000.1"}
        
        vel0kms = vels_Narrow_dict[vels[0]]
        slice_fn = DR2_Narrow_slice_root+"GALFA_HI_N_S{}_V{}kms.fits".format(vels[0], vel0kms)
        slice_data = fits.getdata(slice_fn)
        
        # Add noise to Narrow map: 2*Wide_noisemap because noise goes as sqrt(N) and Narrow = (1/4)Wide
        if addnoise:
            slice_data += noisemap*2*noisefactor
        
        # if longer than one slice, add the rest
        if len(vels) > 1:
            for _vel in vels[1:]:
                velkms = vels_Narrow_dict[_vel]
                slice_fn = DR2_Narrow_slice_root+"GALFA_HI_N_S{}_V{}kms.fits".format(_vel, velkms)
                slice_data += fits.getdata(slice_fn)
            if addnoise:
                print("extra noise for narrow slices not implemented")

    else:
        if vels=="NHI":
            print("USM of total NHI map")
            slice_data = load_2d_data(datatype="NHI90", header=False)
            
        else:
            if LOCAL:
                DR2_Wide_slice_root = "/Users/susanclark/Dropbox/DR2 Full Sky Wide Maps/"
            else:
                DR2_Wide_slice_root = "/data/seclark/GALFADR2/Wide_maps/"
            
            vel0kms = gvh.galfa_name_dict[vels[0]]
            slice_fn = DR2_Wide_slice_root+"GALFA_HI_W_S{}_V{}kms.fits".format(vels[0], vel0kms)
            slice_data = fits.getdata(slice_fn)
            
            # if longer than one slice, add the rest
            if len(vels) > 1:
                print("adding more vels")
                for _vel in vels[1:]:
                    velkms = gvh.galfa_name_dict[_vel]
                    slice_fn = DR2_Wide_slice_root+"GALFA_HI_W_S{}_V{}kms.fits".format(_vel, velkms)
                    slice_data += fits.getdata(slice_fn)
                    
            if addnoise:
                slice_data += noisemap*(1./np.sqrt(len(vels)))*noisefactor
            
    if reverse:
        print("reverse!")
        umask_slice_data = gaussian_umask(slice_data, fwhm=fwhm, zeroed=False)
        umask_slice_data *= -1
        umask_slice_data[np.where(umask_slice_data < 0)] = 0
        
    else:
        umask_slice_data = gaussian_umask(slice_data, fwhm=fwhm, zeroed=zeroed)
        
    umask_slice_data[np.where(np.isnan(umask_slice_data)==True)] = 0 # zero out nans
    
    if writemap:
        if Narrow:
            NWstr = "N"
            DR2_slice_root = "/Users/susanclark/Dropbox/DR2_Full_Sky_Narrow_Maps/"
            narrowstr = "_Narrow"
        else:
            NWstr = "W"
            narrowstr = ""
            if LOCAL:
                DR2_slice_root = "/Users/susanclark/Dropbox/DR2 Full Sky Wide Maps/"
            else:
                DR2_slice_root = "/data/seclark/GALFADR2/Wide_maps/"
        if addnoise:
            noisestr = "_addnoise"
            if noisefactor > 1:
                noisestr = "_addnoise_x{}".format(noisefactor)
        else:
            noisestr = ""
        if reverse:
            print("Not saving because this is a reverse USM")
        else:
            outfn = DR2_slice_root+"GALFA_HI_W_vels{}_to_{}{}_USM{}_zeroed_{}{}.fits".format(vels[0], vels[-1], narrowstr, fwhm, zeroed, noisestr)
            outhdr= fits.getheader(DR2_slice_root+"GALFA_HI_{}_S{}_V{}kms.fits".format(NWstr, vels[0], vel0kms))
            fits.writeto(outfn, umask_slice_data, outhdr)
            
            #outfn = DR2_slice_root+"GALFA_HI_W_vels{}_to_{}{}{}.fits".format(vels[0], vels[-1], narrowstr, noisestr)
            #outhdr= fits.getheader(DR2_slice_root+"GALFA_HI_{}_S{}_V{}kms.fits".format(NWstr, vels[0], vel0kms))
            #fits.writeto(outfn, slice_data, outhdr)
    
    if returnrawslice:
        return umask_slice_data, slice_data
    else:
        return umask_slice_data 
        
def load_USM_stack_data(datatype="weights", fwhm=30, chanstart=1023, chanstop=1025, Narrow=False, bstart=30, bstop=90, randorient=True, cubelen=181):
    if randorient & (datatype != "weights"):
        randorientstr = "_randorient_rot90"
    else:
        randorientstr = ""
    if Narrow:
        narrowstr = "Narrow"
    else:
        narrowstr = ""
    
    stackdata_root = "/Users/susanclark/Projects/FiberDGR/data/"

    fn = "hypercube_{}_USM_{}_{}_{}{}_absb_bstart_{}_bstop_{}_centerw_cubelen{}{}.npy".format(datatype, fwhm, chanstart, chanstop, narrowstr, bstart, bstop, cubelen, randorientstr)
    data = np.load(stackdata_root+fn)
    return data
    
        
def load_channel_map(vels, Narrow=False, addnoise=False, noisefactor=1):
    if addnoise:
        noisestr = "_addnoise"
        if noisefactor > 1:
            noisestr = "_addnoise_x{}".format(noisefactor)
    else:
        noisestr = ""
    
    if Narrow:
        DR2_slice_root = "/Users/susanclark/Dropbox/DR2_Full_Sky_Narrow_Maps/"
        narrowstr = "_Narrow"
        outfn = DR2_slice_root+"GALFA_HI_W_vels{}_to_{}{}{}.fits".format(vels[0], vels[-1], narrowstr, noisestr)
    else:
        DR2_slice_root = "/Users/susanclark/Dropbox/DR2 Full Sky Wide Maps/"
        outfn = DR2_slice_root+"GALFA_HI_W_vels{}_to_{}{}.fits".format(vels[0], vels[-1], noisestr)
    
    slice_map = fits.getdata(outfn)
    return slice_map    
        
def load_USM_map(vels, fwhm=30, zeroed=True, Narrow=False, addnoise=False, noisefactor=1):
    
    if addnoise:
        noisestr = "_addnoise"
        if noisefactor > 1:
            noisestr = "_addnoise_x{}".format(noisefactor)
    else:
        noisestr = ""
    
    if Narrow:
        DR2_slice_root = "/Users/susanclark/Dropbox/DR2_Full_Sky_Narrow_Maps/"
        narrowstr = "_Narrow"
        outfn = DR2_slice_root+"GALFA_HI_W_vels{}_to_{}{}_USM{}_zeroed_{}{}.fits".format(vels[0], vels[-1], narrowstr, fwhm, zeroed, noisestr)
    else:
        narrowstr = ""
        DR2_slice_root = "/Users/susanclark/Dropbox/DR2 Full Sky Wide Maps/"
        outfn = DR2_slice_root+"GALFA_HI_W_vels{}_to_{}{}_USM{}_zeroed_{}{}.fits".format(vels[0], vels[-1], narrowstr, fwhm, zeroed, noisestr)
    
    usm_map = fits.getdata(outfn)
    return usm_map
    
def extended_deltaFIR_plot(centervel=2014, addnoise=False, noisefactor=1):

    nhi90map = fits.getdata("/Users/susanclark/Dropbox/NHImaps/GALFA-HI_NHISRCORR_VLSR-90+90kms.fits")
    p857map = fits.getdata("/Users/susanclark/Dropbox/Planck/HFI_SkyMap_857_2048_R3.01_ONGALFAHI.fits")
    submonopole = 0.64
    if submonopole is not None:
        p857map -= submonopole
        submonopolestr = "_submono{}".format(submonopole)
    else:
        submonopolestr = ""
        
    nhi90map_div1E20 = nhi90map/1.E20

    fwhm = 30
    bstart = 30
    bstop = 90
    NHIcut = True
    NHImin = 0
    NHImax = 8
    maskbs = make_mask_2d(bstart=bstart, bstop=bstop, PS=True, NHIcut=False)  
    
    nvels = 31
    all_meancutnonzeroP857 = np.zeros(nvels+1)
    all_weightedmeancutnonzeroP857 = np.zeros(nvels+1)
    all_meancutnonzeroNHI = np.zeros(nvels+1)
    all_weightedmeancutnonzeroNHI = np.zeros(nvels+1)
    all_meancutwhereP857 = np.zeros(nvels+1)
    all_weightedmeancutwhereP857 = np.zeros(nvels+1)
    all_meancutwhereNHI = np.zeros(nvels+1)
    all_weightedmeancutwhereNHI = np.zeros(nvels+1)
    
    thicknesses = np.zeros(nvels+1)
    mean_USM1_FIR = np.zeros(nvels+1)
    mean_USM1_NHI = np.zeros(nvels+1)
    mean_USM1_FIR_NHIcut = np.zeros(nvels+1)
    mean_USM1_NHI_NHIcut = np.zeros(nvels+1)
    
    #narrow_umask = get_USM_slice(vels=["1024"], fwhm=fwhm, zeroed=True, Narrow=True, reverse=False, writemap=False)
    narrow_umask = load_USM_map(vels=["1024"], fwhm=fwhm, zeroed=True, Narrow=True, addnoise=addnoise, noisefactor=noisefactor)
    all_meancutnonzeroP857[0], all_weightedmeancutnonzeroP857[0], all_meancutnonzeroNHI[0], all_weightedmeancutnonzeroNHI[0], all_meancutwhereP857[0], all_weightedmeancutwhereP857[0], all_meancutwhereNHI[0], all_weightedmeancutwhereNHI[0] = weighted_mean(nhi90map, p857map, weightsmap=narrow_umask, mask=maskbs, nhi_max_fit=NHImax, nhi_min_fit=NHImin)
    thicknesses[0] = 0.184
    
    mean_USM1_FIR[0] = np.nanmean(p857map[np.where((maskbs > 0) & (narrow_umask > 0))])
    mean_USM1_NHI[0] = np.nanmean(nhi90map[np.where((maskbs > 0) & (narrow_umask > 0))])
    mean_USM1_FIR_NHIcut[0] = np.nanmean(p857map[np.where((maskbs > 0) & (narrow_umask > 0) & (nhi90map_div1E20 > NHImin) & (nhi90map_div1E20 < NHImax) )])
    mean_USM1_NHI_NHIcut[0] = np.nanmean(nhi90map[np.where((maskbs > 0) & (narrow_umask > 0) & (nhi90map_div1E20 > NHImin) & (nhi90map_div1E20 < NHImax))])
    
    for _i in np.arange(nvels):
        startvel = centervel - _i
        stopvel = centervel + _i
        vels = [makestring(v) for v in np.arange(startvel, stopvel+1)]
        
        _iplus1 = _i + 1
        thicknesses[_iplus1] = 0.184*4*len(vels)
        
        vels_umask = load_USM_map(vels, fwhm=fwhm, zeroed=True, Narrow=False, addnoise=addnoise, noisefactor=noisefactor)
    
        all_meancutnonzeroP857[_iplus1], all_weightedmeancutnonzeroP857[_iplus1], all_meancutnonzeroNHI[_iplus1], all_weightedmeancutnonzeroNHI[_iplus1], all_meancutwhereP857[_iplus1], all_weightedmeancutwhereP857[_iplus1], all_meancutwhereNHI[_iplus1], all_weightedmeancutwhereNHI[_iplus1] = weighted_mean(nhi90map, p857map, weightsmap=vels_umask, mask=maskbs, nhi_max_fit=NHImax, nhi_min_fit=NHImin)
        
        mean_USM1_FIR[_iplus1] = np.nanmean(p857map[np.where((maskbs > 0) & (vels_umask > 0))])
        mean_USM1_NHI[_iplus1] = np.nanmean(nhi90map[np.where((maskbs > 0) & (vels_umask > 0))])
        mean_USM1_FIR_NHIcut[_iplus1] = np.nanmean(p857map[np.where((maskbs > 0) & (vels_umask > 0) & (nhi90map_div1E20 > NHImin) & (nhi90map_div1E20 < NHImax) )])
        mean_USM1_NHI_NHIcut[_iplus1] = np.nanmean(nhi90map[np.where((maskbs > 0) & (vels_umask > 0) & (nhi90map_div1E20 > NHImin) & (nhi90map_div1E20 < NHImax))])
        
                
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    
    delta857cutnonzero = all_weightedmeancutnonzeroP857 - all_meancutnonzeroP857
    delta857cutwhere = all_weightedmeancutwhereP857 - all_meancutwhereP857
    deltaNHIcutnonzero = all_weightedmeancutnonzeroNHI - all_meancutnonzeroNHI
    deltaNHIcutwhere = all_weightedmeancutwhereNHI - all_meancutwhereNHI
    
    ax.plot(thicknesses, delta857cutnonzero, '.')
    ax.plot(thicknesses, delta857cutwhere, '.')
    ax.plot(thicknesses, deltaNHIcutnonzero, '.')
    ax.plot(thicknesses, deltaNHIcutwhere, '.')
    ax.legend(['delta857cutnonzero', 'delta857cutwhere', 'deltaNHIcutnonzero', 'deltaNHIcutwhere'])
    
    if addnoise:
        noisestr = "_addnoise"
        if noisefactor > 1:
            noisestr = "_addnoise_x{}".format(noisefactor)
    else:
        noisestr = ""
    
    with h5py.File('../data/extended_deltaFIR_NHI_data_USM_centervel{}_fwhm{}_b{}_to_{}{}{}.h5'.format(centervel, fwhm, bstart, bstop, submonopolestr, noisestr), 'w') as hf:
        dset = hf.create_dataset("thicknesses",  data=thicknesses)
        hf.create_dataset("all_meancutnonzeroP857",  data=all_meancutnonzeroP857)
        hf.create_dataset("all_weightedmeancutnonzeroP857",  data=all_weightedmeancutnonzeroP857)
        hf.create_dataset("all_meancutnonzeroNHI",  data=all_meancutnonzeroNHI)
        hf.create_dataset("all_weightedmeancutnonzeroNHI",  data=all_weightedmeancutnonzeroNHI)
        hf.create_dataset("all_meancutwhereP857",  data=all_meancutwhereP857)
        hf.create_dataset("all_weightedmeancutwhereP857",  data=all_weightedmeancutwhereP857)
        hf.create_dataset("all_meancutwhereNHI",  data=all_meancutwhereNHI)
        hf.create_dataset("all_weightedmeancutwhereNHI",  data=all_weightedmeancutwhereNHI)
        hf.create_dataset("mean_USM1_FIR",  data=mean_USM1_FIR)
        hf.create_dataset("mean_USM1_NHI",  data=mean_USM1_NHI)
        hf.create_dataset("mean_USM1_FIR_NHIcut",  data=mean_USM1_FIR_NHIcut)
        hf.create_dataset("mean_USM1_NHI_NHIcut",  data=mean_USM1_NHI_NHIcut)
        dset.attrs['fwhm'] = fwhm
        dset.attrs['bstart'] = bstart
        dset.attrs['bstop'] = bstop
        dset.attrs['submonopole'] = submonopole
        dset.attrs['centervel'] = centervel
        dset.attrs['addnoise'] = addnoise
        dset.attrs['noisefactor'] = noisefactor
        
        
    #return thicknesses, all_meancutnonzeroP857, all_weightedmeancutnonzeroP857, all_meancutnonzeroNHI, all_weightedmeancutnonzeroNHI, all_meancutwhereP857, all_weightedmeancutwhereP857, all_meancutwhereNHI, all_weightedmeancutwhereNHI, mean_USM1_FIR, mean_USM1_NHI, mean_USM1_FIR_NHIcut, mean_USM1_NHI_NHIcut

#thicknesses, noise_all_meancutnonzeroP857, noise_all_weightedmeancutnonzeroP857, noise_all_meancutnonzeroNHI, noise_all_weightedmeancutnonzeroNHI, noise_all_meancutwhereP857, noise_all_weightedmeancutwhereP857, noise_all_meancutwhereNHI, noise_all_weightedmeancutwhereNHI, noise_mean_USM1_FIR, noise_mean_USM1_NHI, noise_mean_USM1_FIR_NHIcut, noise_mean_USM1_NHI_NHIcut

def load_weighted_mean_data(centervel=1024, fwhm=30, bstart=30, bstop=90, submonopole=0.64, addnoise=False, noisefactor=1, returndict=True):
    
    if submonopole is not None:
        submonopolestr = "_submono{}".format(submonopole)
    else:
        submonopolestr = ""
    if addnoise:
        noisestr = "_addnoise"
        if noisefactor > 1:
            noisestr = "_addnoise_x{}".format(noisefactor)
    else:
        noisestr = ""
    if centervel != 1024:
        centervelstr = "_centervel{}".format(centervel)
    else:
        centervelstr = ""
        
    fn = '../data/extended_deltaFIR_NHI_data_USM{}_fwhm{}_b{}_to_{}{}{}.h5'.format(centervelstr, fwhm, bstart, bstop, submonopolestr, noisestr)

    if returndict:
        datadict = {}
    
    with h5py.File(fn, 'r') as f:
        
        if returndict:
            for _key in list(f.keys()):
                datadict[_key] = f[_key][:]
        
        else:
            thicknesses = f['thicknesses'][:]
            all_meancutnonzeroP857 = f['all_meancutnonzeroP857'][:]
            all_weightedmeancutnonzeroP857 = f['all_weightedmeancutnonzeroP857'][:]
            all_meancutnonzeroNHI = f['all_meancutnonzeroNHI'][:]
            all_weightedmeancutnonzeroNHI = f['all_weightedmeancutnonzeroNHI'][:]
            
            all_meancutwhereP857 = f['all_meancutwhereP857'][:]
            all_weightedmeancutwhereP857 = f['all_weightedmeancutwhereP857'][:]
            all_meancutwhereNHI = f['all_meancutwhereNHI'][:]
            all_weightedmeancutwhereNHI = f['all_weightedmeancutwhereNHI'][:]
            
            mean_USM1_FIR = f['mean_USM1_FIR'][:]
            mean_USM1_NHI = f['mean_USM1_NHI'][:]
            mean_USM1_FIR_NHIcut = f['mean_USM1_FIR_NHIcut'][:]
            mean_USM1_NHI_NHIcut = f['mean_USM1_NHI_NHIcut'][:]
        
    if returndict:
        return datadict
    else:
        return thicknesses, all_meancutnonzeroP857, all_weightedmeancutnonzeroP857, all_meancutnonzeroNHI, all_weightedmeancutnonzeroNHI, all_meancutwhereP857, all_weightedmeancutwhereP857, all_meancutwhereNHI, all_weightedmeancutwhereNHI, mean_USM1_FIR, mean_USM1_NHI, mean_USM1_FIR_NHIcut, mean_USM1_NHI_NHIcut

def compute_USM_NHI_weighted_means():
    nhi90map = fits.getdata("/Users/susanclark/Dropbox/NHImaps/GALFA-HI_NHISRCORR_VLSR-90+90kms.fits")
    p857map = fits.getdata("/Users/susanclark/Dropbox/Planck/HFI_SkyMap_857_2048_R3.01_ONGALFAHI.fits")
    submonopole = 0.64
    if submonopole is not None:
        p857map -= submonopole
        submonopolestr = "_submono{}".format(submonopole)
    else:
        submonopolestr = ""
        
    nhi90map_div1E20 = nhi90map/1.E20
    
    fwhm = 30
    bstart = 30
    bstop = 90
    NHIcut = True
    NHImin = 0
    NHImax = 8
    maskbs = make_mask_2d(bstart=bstart, bstop=bstop, PS=True, NHIcut=False) 
    
    nhi_USM = gaussian_umask(nhi90map, fwhm=fwhm, zeroed=True)
    nhi_USM[np.where(np.isnan(nhi_USM) == True)] = 0

    nhi_usm_dict = {}
    nhi_usm_dict["meancutnonzeroP857"], nhi_usm_dict["NHI_weightedmeancutnonzeroP857"], nhi_usm_dict["NHI_meancutnonzeroNHI"], nhi_usm_dict["NHI_weightedmeancutnonzeroNHI"], nhi_usm_dict["NHI_meancutwhereP857"], nhi_usm_dict["NHI_weightedmeancutwhereP857"], nhi_usm_dict["NHI_meancutwhereNHI"], nhi_usm_dict["NHI_weightedmeancutwhereNHI"] = weighted_mean(nhi90map, p857map, weightsmap=nhi_USM, mask=maskbs, nhi_max_fit=NHImax, nhi_min_fit=NHImin)

    nhi_usm_dict["mean_USM1_FIR"] = np.nanmean(p857map[np.where((maskbs > 0) & (nhi_USM > 0))])
    nhi_usm_dict["mean_USM1_NHI"] = np.nanmean(nhi90map[np.where((maskbs > 0) & (nhi_USM > 0))])
    nhi_usm_dict["mean_USM1_FIR_NHIcut"] = np.nanmean(p857map[np.where((maskbs > 0) & (nhi_USM > 0) & (nhi90map_div1E20 > NHImin) & (nhi90map_div1E20 < NHImax) )])
    nhi_usm_dict["mean_USM1_NHI_NHIcut"] = np.nanmean(nhi90map[np.where((maskbs > 0) & (nhi_USM > 0) & (nhi90map_div1E20 > NHImin) & (nhi90map_div1E20 < NHImax))])
    
    print(nhi_usm_dict)
    np.save("../data/USM_NHI_weighted_means_fwhm{}_bstart{}_bstop{}_NHImin{}_NHImax{}.npy".format(fwhm, bstart, bstop, NHImin, NHImax), nhi_usm_dict)

def radial_profile(data):
    ny, nx = data.shape
    y, x = np.indices((ny, nx))
    center = [(ny-1)/2., (nx-1)/2.]
    
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile 
    
def plot_diffcenter_data_and_residuals():

    centervel = 1019 #990#
    fwhm = 30
    bstart = 30
    bstop = 90
    NHImin = 0
    NHImax = 8
    submonopole = 0.64
    datadict = load_weighted_mean_data(fwhm=fwhm, centervel=centervel, bstart=bstart, bstop=bstop, submonopole=submonopole, addnoise=False, returndict=True)

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    
    deltaFIR_cutwhere = datadict['all_weightedmeancutwhereP857'] - datadict['all_meancutwhereP857']
    
    ax1.plot(datadict['thicknesses'], datadict['all_weightedmeancutwhereP857'], '.', color='teal')
    ax1.set_ylabel("857 GHz")
    ax1.set_xlabel("channel width [km/s]")
    ax1.set_title("FIR weighted mean")
    # plot NHI USM values
    #ax1.plot(50, nhiUSMdict['NHI_weightedmeancutwhereP857'] - nhiUSMdict['NHI_meancutwhereP857'], 'd', color='teal')
    #ax1.plot(50, nhiUSMdict['NHI_weightedmeancutwhereP857'] - nhiUSMdict['mean_USM1_FIR_NHIcut'], 'd', color='orange')
    
    ax2.plot(datadict['thicknesses'], datadict['all_meancutwhereP857'], '.', color='teal')
    ax2.plot(datadict['thicknesses'], datadict['mean_USM1_FIR_NHIcut'], '.', color='orange')
    ax2.legend(['unweighted mean of sky', 'unweighted mean of USM$>$0'])
    ax2.set_xlabel("channel width [km/s]")
    ax2.set_title("FIR unweighted mean")
    
    ax3.plot(datadict['thicknesses'], deltaFIR_cutwhere, '.', color='teal')
    ax3.plot(datadict['thicknesses'], datadict['all_weightedmeancutwhereP857'] - datadict['mean_USM1_FIR_NHIcut'], '.', color='orange')
    ax3.set_ylim(0, 0.8)
    ax3.set_xlabel("channel width [km/s]")
    ax3.legend(['weighted mean - sky average', 'weighted mean - mask average'])
    ax3.set_title("weighted mean - unweighted mean")

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(0.3, 45)
        
def plot_noisedata():
    fwhm = 30
    bstart = 30
    bstop = 90
    NHImin = 0
    NHImax = 8
    submonopole = 0.64
    datadict = load_weighted_mean_data(fwhm=fwhm, bstart=bstart, bstop=bstop, submonopole=submonopole, addnoise=False, returndict=True)
    noisedatadict = load_weighted_mean_data(fwhm=fwhm, bstart=bstart, bstop=bstop, submonopole=submonopole, addnoise=True, returndict=True)
    noisedatadictx2 = load_weighted_mean_data(fwhm=fwhm, bstart=bstart, bstop=bstop, submonopole=submonopole, addnoise=True, noisefactor=2, returndict=True)
    nhiUSMdict = np.load("../data/USM_NHI_weighted_means_fwhm{}_bstart{}_bstop{}_NHImin{}_NHImax{}.npy".format(fwhm, bstart, bstop, NHImin, NHImax)).item()
    
    deltaFIR_cutwhere = datadict['all_weightedmeancutwhereP857'] - datadict['all_meancutwhereP857']
    velwidths = datadict['thicknesses']
    
    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot(111)
    
    #ax3.plot(velwidths, datadict['all_weightedmeancutwhereP857'] - noisedatadict['all_weightedmeancutwhereP857'], '.', color='black', alpha=0.8)
    #ax3.plot(velwidths, datadict['all_weightedmeancutwhereP857'] - noisedatadictx2['all_weightedmeancutwhereP857'], '.', color='gray', alpha=0.5)

    #ax.plot(velwidths,  deltaFIR_cutwhere - deltaFIR_cutwhere, '-o', color='teal', alpha=0.5, label='no noise')
    #ax.plot(velwidths,  (noisedatadict['all_weightedmeancutwhereP857'] - datadict['all_meancutwhereP857']) - deltaFIR_cutwhere, '-o', color='black', alpha=0.5, label='+ noise')
    #ax.plot(velwidths,  (noisedatadictx2['all_weightedmeancutwhereP857'] - datadict['all_meancutwhereP857']) - deltaFIR_cutwhere, '-o', color='gray', alpha=0.5, label='+ noise x 2')

    markersize=7
    ax.plot(velwidths,  deltaFIR_cutwhere - deltaFIR_cutwhere, color='teal', ms=markersize, alpha=0.5)
    ax.plot(velwidths,  (noisedatadict['all_weightedmeancutwhereP857'] - datadict['all_meancutwhereP857']) - deltaFIR_cutwhere, color='black', alpha=0.5)
    ax.plot(velwidths,  (noisedatadictx2['all_weightedmeancutwhereP857'] - datadict['all_meancutwhereP857']) - deltaFIR_cutwhere, color='gray', alpha=0.5)
    
    ax.plot(velwidths,  deltaFIR_cutwhere - deltaFIR_cutwhere, 'o', color='white', alpha=1)
    ax.plot(velwidths,  (noisedatadict['all_weightedmeancutwhereP857'] - datadict['all_meancutwhereP857']) - deltaFIR_cutwhere, 'o', color='white', alpha=1, ms=markersize)
    ax.plot(velwidths,  (noisedatadictx2['all_weightedmeancutwhereP857'] - datadict['all_meancutwhereP857']) - deltaFIR_cutwhere, 'o', color='white', alpha=1, ms=markersize)

    ax.plot(velwidths,  deltaFIR_cutwhere - deltaFIR_cutwhere, 'o', color='teal', mec='None', alpha=0.5, ms=markersize, label=r'$\mathrm{no}$ $\mathrm{added}$ $\mathrm{noise}$')
    ax.plot(velwidths,  (noisedatadict['all_weightedmeancutwhereP857'] - datadict['all_meancutwhereP857']) - deltaFIR_cutwhere, 'o', color='black', mec='None', alpha=0.5, ms=markersize, label=r'$\mathrm{data}$ $+$ $\mathrm{noise}$')
    ax.plot(velwidths,  (noisedatadictx2['all_weightedmeancutwhereP857'] - datadict['all_meancutwhereP857']) - deltaFIR_cutwhere, 'o', color='gray', mec='None', alpha=0.5, ms=markersize, label=r'$\mathrm{data}$ $+$ $\mathrm{noise}$ $\times$ $2$')


    ax.legend(fontsize=12)
    
    #ax.set_xlim(-0.5, 12)
    ax.set_xlim(0.0, 11.1504)
    ax.set_ylim(-0.24, 0.04)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    #ax.spines["left"].set_bounds(0, 0.8)
    #ax.spines["bottom"].set_bounds(0, 12)
    ax.spines["left"].set_bounds(-0.22, 0)
    ax.spines["bottom"].set_bounds(0, 11.04)
    
    ax.set_xlabel(r'$\mathrm{Channel}$ $\mathrm{Width}$ $[\mathrm{km/s}]$', size=15)
    ax.set_ylabel(r'$\Delta$ $\mathrm{I}_{857}^{noise}$ $-$ $\Delta$ $\mathrm{I}_{857}$ $[\mathrm{MJy/sr}]$', size=15)
    
    ticklabelsize = 12
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(ticklabelsize) 
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(ticklabelsize) 
    
    plt.tight_layout()
        
def plot_diff_residuals_total_onmask():
    fwhm = 30
    bstart = 30
    bstop = 90
    NHImin = 0
    NHImax = 8
    submonopole = 0.64
    datadict = load_weighted_mean_data(fwhm=fwhm, bstart=bstart, bstop=bstop, submonopole=submonopole, addnoise=False, returndict=True)
    noisedatadict = load_weighted_mean_data(fwhm=fwhm, bstart=bstart, bstop=bstop, submonopole=submonopole, addnoise=True, returndict=True)
    noisedatadictx2 = load_weighted_mean_data(fwhm=fwhm, bstart=bstart, bstop=bstop, submonopole=submonopole, addnoise=True, noisefactor=2, returndict=True)
    nhiUSMdict = np.load("../data/USM_NHI_weighted_means_fwhm{}_bstart{}_bstop{}_NHImin{}_NHImax{}.npy".format(fwhm, bstart, bstop, NHImin, NHImax)).item()
    
    deltaFIR_cutwhere = datadict['all_weightedmeancutwhereP857'] - datadict['all_meancutwhereP857']
    velwidths = datadict['thicknesses']
    
    fig = plt.figure(figsize=(7.,7))
    nrows = 10
    ncolumns = 1
    ax1 = plt.subplot2grid((nrows, ncolumns), (0, 0), rowspan=4, colspan=1)
    ax2 = plt.subplot2grid((nrows, ncolumns), (6, 0), rowspan=4, colspan=1)
    
    #ax1 = fig.add_subplot(311)
    #ax2 = fig.add_subplot(312)
    #ax3 = fig.add_subplot(313)
    
    color1 = 'coral' #'#FF634D' #'palevioletred' #'mediumslateblue'

    # raw
    ax1.plot(velwidths, datadict['all_weightedmeancutwhereP857'], ':', color='teal', label=r'$\bar{I}^{w}_{857}$')
    ax1.plot(velwidths, datadict['all_meancutwhereP857'], '--', color='black', label=r'$\bar{I}_{857}$')
    ax1.plot(velwidths, datadict['mean_USM1_FIR_NHIcut'], color=color1, label=r'$\bar{I}_{857}(w > 0)$')
    
    ax1.legend(bbox_to_anchor=(0.059, 0.9))

    #ax1.fill_between(velwidths, datadict['all_meancutwhereP857'], datadict['all_weightedmeancutwhereP857'], facecolor=color1, hatch="", alpha=0.3)
    #ax1.fill_between(velwidths, datadict['mean_USM1_FIR_NHIcut'], datadict['all_weightedmeancutwhereP857'], facecolor='teal', hatch="", alpha=0.3)
    ax1.fill_between(velwidths, datadict['mean_USM1_FIR_NHIcut'], datadict['all_weightedmeancutwhereP857'], facecolor=color1, hatch="", alpha=0.3)
    
    
    # differences
    ax2.plot(velwidths, deltaFIR_cutwhere, '.', color='teal')
    ax2.plot(velwidths, datadict['all_weightedmeancutwhereP857'] - datadict['mean_USM1_FIR_NHIcut'], '.', color=color1)
    ax2.set_ylim(0, np.max(deltaFIR_cutwhere) + 0.1*np.max(deltaFIR_cutwhere))
    
    #ax3.plot(velwidths, datadict['all_weightedmeancutwhereP857'] - noisedatadict['all_weightedmeancutwhereP857'], '.', color='black', alpha=0.8)
    #ax3.plot(velwidths, datadict['all_weightedmeancutwhereP857'] - noisedatadictx2['all_weightedmeancutwhereP857'], '.', color='gray', alpha=0.5)
    
    #ax3.plot(velwidths,  deltaFIR_cutwhere, '.', color='teal', alpha=0.8)
    #ax3.plot(velwidths,  noisedatadict['all_weightedmeancutwhereP857'] - datadict['all_meancutwhereP857'], '.', color='black', alpha=0.8)
    #ax3.plot(velwidths, noisedatadictx2['all_weightedmeancutwhereP857'] - datadict['all_meancutwhereP857'], '.', color='gray', alpha=0.5)
    

    ax1.set_ylabel(r'$\bar{\mathrm{I}}_{857}$ $[\mathrm{MJy/sr}]$', size=15)
    ax2.set_ylabel(r'$\Delta$ $\mathrm{I}_{857}$ $[\mathrm{MJy/sr}]$', size=15)
    ax1.set_xlabel(r'$\mathrm{Channel}$ $\mathrm{Width}$ $[\mathrm{km/s}]$', size=15)
    ax2.set_xlabel(r'$\mathrm{Channel}$ $\mathrm{Width}$ $[\mathrm{km/s}]$', size=15)
    offhi857 = 54
    maxoffhi857 = 59
    minoffhi857 = 49
    
    ax1.set_ylim(1.3, 2.2)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_bounds(1.3, 2.2)
    ax1.spines["bottom"].set_bounds(0, 45)

    ax2.set_ylim(-0.05, 0.8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_bounds(0, 0.8)
    ax2.spines["bottom"].set_bounds(0, 45)



def plot_data_and_residuals():
    
    fwhm = 30
    bstart = 30
    bstop = 90
    NHImin = 0
    NHImax = 8
    submonopole = 0.64
    datadict = load_weighted_mean_data(fwhm=fwhm, bstart=bstart, bstop=bstop, submonopole=submonopole, addnoise=False, returndict=True)
    noisedatadict = load_weighted_mean_data(fwhm=fwhm, bstart=bstart, bstop=bstop, submonopole=submonopole, addnoise=True, returndict=True)
    noisedatadictx2 = load_weighted_mean_data(fwhm=fwhm, bstart=bstart, bstop=bstop, submonopole=submonopole, addnoise=True, noisefactor=2, returndict=True)
    nhiUSMdict = np.load("../data/USM_NHI_weighted_means_fwhm{}_bstart{}_bstop{}_NHImin{}_NHImax{}.npy".format(fwhm, bstart, bstop, NHImin, NHImax)).item()
    
    deltaFIR_cutwhere = datadict['all_weightedmeancutwhereP857'] - datadict['all_meancutwhereP857']
    
    fig = plt.figure()
    ax1 = fig.add_subplot(411)
    ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413)
    ax4 = fig.add_subplot(414)
    
    ax1.plot(datadict['thicknesses'], datadict['all_weightedmeancutwhereP857'], '.', color='teal')
    ax1.set_ylabel("857 GHz")
    ax1.set_xlabel("channel width [km/s]")
    ax1.set_title("FIR weighted mean")
    # plot NHI USM values
    #ax1.plot(50, nhiUSMdict['NHI_weightedmeancutwhereP857'] - nhiUSMdict['NHI_meancutwhereP857'], 'd', color='teal')
    #ax1.plot(50, nhiUSMdict['NHI_weightedmeancutwhereP857'] - nhiUSMdict['mean_USM1_FIR_NHIcut'], 'd', color='orange')
    
    ax2.plot(datadict['thicknesses'], datadict['all_meancutwhereP857'], '.', color='teal')
    ax2.plot(datadict['thicknesses'], datadict['mean_USM1_FIR_NHIcut'], '.', color='orange')
    ax2.legend(['unweighted mean of sky', 'unweighted mean of USM$>$0'])
    ax2.set_xlabel("channel width [km/s]")
    ax2.set_title("FIR unweighted mean")
    
    ax3.plot(datadict['thicknesses'], deltaFIR_cutwhere, '.', color='teal')
    ax3.plot(datadict['thicknesses'], datadict['all_weightedmeancutwhereP857'] - datadict['mean_USM1_FIR_NHIcut'], '.', color='orange')
    ax3.set_ylim(0, 0.8)
    ax3.set_xlabel("channel width [km/s]")
    ax3.legend(['weighted mean - sky average', 'weighted mean - mask average'])
    ax3.set_title("weighted mean - unweighted mean")
    
    ax4.plot(noisedatadict['thicknesses'], datadict['all_weightedmeancutwhereP857'] - noisedatadict['all_weightedmeancutwhereP857'], '.', alpha=0.5)
    ax4.plot(noisedatadictx2['thicknesses'], datadict['all_weightedmeancutwhereP857'] - noisedatadictx2['all_weightedmeancutwhereP857'], '.', color='red', alpha=0.5)
    ax4.set_xlabel("channel width [km/s]")
    ax4.set_title("weighted mean - weighted mean of (data + noise)")
    ax4.legend(['noise', 'noisex2'])
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(0, 14)

def plot_NHI_data_and_residuals():
    
    fwhm = 30
    bstart = 30
    bstop = 90
    NHImin = 0
    NHImax = 8
    submonopole = 0.64
    datadict = load_weighted_mean_data(fwhm=fwhm, bstart=bstart, bstop=bstop, submonopole=submonopole, addnoise=False, returndict=True)
    noisedatadict = load_weighted_mean_data(fwhm=fwhm, bstart=bstart, bstop=bstop, submonopole=submonopole, addnoise=True, returndict=True)
    nhiUSMdict = np.load("../data/USM_NHI_weighted_means_fwhm{}_bstart{}_bstop{}_NHImin{}_NHImax{}.npy".format(fwhm, bstart, bstop, NHImin, NHImax)).item()
    
    deltaFIR_cutwhere = datadict['all_weightedmeancutwhereNHI'] - datadict['all_meancutwhereNHI']
    
    fig = plt.figure()
    ax1 = fig.add_subplot(411)
    ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413)
    ax4 = fig.add_subplot(414)
    
    ax1.plot(datadict['thicknesses'], datadict['all_weightedmeancutwhereNHI'], '.', color='teal')
    ax1.set_ylabel("NHI")
    ax1.set_xlabel("channel width [km/s]")
    ax1.set_title("NHI weighted mean")
    # plot NHI USM values
    #ax1.plot(50, nhiUSMdict['NHI_weightedmeancutwhereP857'] - nhiUSMdict['NHI_meancutwhereP857'], 'd', color='teal')
    #ax1.plot(50, nhiUSMdict['NHI_weightedmeancutwhereP857'] - nhiUSMdict['mean_USM1_FIR_NHIcut'], 'd', color='orange')
    
    ax2.plot(datadict['thicknesses'], datadict['all_meancutwhereNHI'], '.', color='teal')
    ax2.plot(datadict['thicknesses'], datadict['mean_USM1_NHI_NHIcut']/1.E20, '.', color='orange')
    ax2.legend(['unweighted mean of sky', 'unweighted mean of USM$>$0'])
    ax2.set_xlabel("channel width [km/s]")
    ax2.set_title("NHI unweighted mean")
    
    ax3.plot(datadict['thicknesses'], deltaFIR_cutwhere, '.', color='teal')
    ax3.plot(datadict['thicknesses'], datadict['all_weightedmeancutwhereNHI'] - datadict['mean_USM1_NHI_NHIcut']/1.E20, '.', color='orange')
    #ax3.set_ylim(0, 0.8)
    ax3.set_xlabel("channel width [km/s]")
    ax3.legend(['weighted mean - sky average', 'weighted mean - mask average'])
    ax3.set_title("weighted mean - unweighted mean")
    
    ax4.plot(noisedatadict['thicknesses'], datadict['all_weightedmeancutwhereNHI'] - noisedatadict['all_weightedmeancutwhereNHI'], '.')
    ax4.set_xlabel("channel width [km/s]")
    ax4.set_title("weighted mean - weighted mean of (data + noise)")
    
def RGB_to_HSV_arr(Rarr, Garr, Barr, reverse=False):
    assert Rarr.shape == Garr.shape == Barr.shape
    ny, nx = Rarr.shape
    Rflat = Rarr.flatten()
    Gflat = Garr.flatten()
    Bflat = Barr.flatten()
    Hflat = np.zeros(Rflat.shape)
    Sflat = np.zeros(Rflat.shape)
    Vflat = np.zeros(Rflat.shape)
    for _i in np.arange(len(Hflat)):
        if reverse:
            Hflat[_i], Sflat[_i], Vflat[_i] = colorsys.hsv_to_rgb(Rflat[_i], Gflat[_i], Bflat[_i])
        else:
            Hflat[_i], Sflat[_i], Vflat[_i] = colorsys.rgb_to_hsv(Rflat[_i], Gflat[_i], Bflat[_i])
        
    Harr = Hflat.reshape(ny, nx)
    Sarr = Sflat.reshape(ny, nx)
    Varr = Vflat.reshape(ny, nx)
    return Harr, Sarr, Varr

def plot_RGB_FIR_data():
    DR2_Wide_slice_root = "/Users/susanclark/Dropbox/DR2 Full Sky Wide Maps/"

    nhi90map = fits.getdata("/Users/susanclark/Dropbox/NHImaps/GALFA-HI_NHISRCORR_VLSR-90+90kms.fits")
    p857map = fits.getdata("/Users/susanclark/Dropbox/Planck/HFI_SkyMap_857_2048_R3.01_ONGALFAHI.fits")
    submonopole = 0.64
    if submonopole is not None:
        p857map -= submonopole

    velkms = gvh.galfa_name_dict["1024"]
    slice_fn = DR2_Wide_slice_root+"GALFA_HI_W_S{}_V{}kms.fits".format(1024, velkms)
    slice_data_1024 = fits.getdata(slice_fn)
    big_hdr = fits.getheader(slice_fn)

    velkms = gvh.galfa_name_dict["1025"]
    slice_fn = DR2_Wide_slice_root+"GALFA_HI_W_S{}_V{}kms.fits".format(1025, velkms)
    slice_data_1025 = fits.getdata(slice_fn)

    velkms = gvh.galfa_name_dict["1023"]
    slice_fn = DR2_Wide_slice_root+"GALFA_HI_W_S{}_V{}kms.fits".format(1023, velkms)
    slice_data_1023 = fits.getdata(slice_fn)
    
    umask_1024 = load_USM_map(vels=["1024"], fwhm=30, zeroed=True, Narrow=False, addnoise=False)
    
    bstart = 30
    NHIcut = True
    NHImin = 0
    NHImax = 8
    maskbs = make_mask_2d(bstart=bstart, bstop=90, PS=True, NHIcut=NHIcut, NHImin=NHImin, NHImax=NHImax)  

    #ystart=1498#1500
    #ystop=2200
    ystart=1000
    ystop=1800
    xstart=7000
    xstop=8500
    #xstart=8500
    #xstop=10500
    
    xstartsmaller=8000
    
    cut_hdr, p857cut = cutouts.xycutout_data(p857map, big_hdr, xstart=xstart, xstop=xstop, ystart=ystart, ystop=ystop)
    cut_hdr, nhi90cut = cutouts.xycutout_data(nhi90map, big_hdr, xstart=xstart, xstop=xstop, ystart=ystart, ystop=ystop)

    cut_hdr, slice_data_1024_cut = cutouts.xycutout_data(slice_data_1024, big_hdr, xstart=xstart, xstop=xstop, ystart=ystart, ystop=ystop)
    cut_hdr, slice_data_1025_cut = cutouts.xycutout_data(slice_data_1025, big_hdr, xstart=xstart, xstop=xstop, ystart=ystart, ystop=ystop)
    cut_hdr, slice_data_1023_cut = cutouts.xycutout_data(slice_data_1023, big_hdr, xstart=xstart, xstop=xstop, ystart=ystart, ystop=ystop)

    cut_hdr_smaller, slice_data_1024_cut_smaller = cutouts.xycutout_data(slice_data_1024, big_hdr, xstart=xstartsmaller, xstop=xstop, ystart=ystart, ystop=ystop)
    cut_hdr_smaller, umask_1024_cut_smaller = cutouts.xycutout_data(umask_1024, big_hdr, xstart=xstartsmaller, xstop=xstop, ystart=ystart, ystop=ystop)

    cut_hdr, maskbs_cut = cutouts.xycutout_data(maskbs, big_hdr, xstart=xstart, xstop=xstop, ystart=ystart, ystop=ystop)

    xax, ra_label = cutouts.get_xlabels_ra(cut_hdr, skip = 200.0, origincoord=0)
    yax, dec_label = cutouts.get_ylabels_dec(cut_hdr, skip = 200.0, origincoord=0)
    
    w = cutouts.make_wcs(cut_hdr)
    
    chunky, chunkx = slice_data_1024_cut.shape
    chunkysmaller, chunkxsmaller = umask_1024_cut_smaller.shape
    
    RGBdata = np.zeros((ystop-ystart, xstop-xstart, 3), np.float_)
    RGBdata[:, :, 0]= slice_data_1023[ystart:ystop, xstart:xstop]
    RGBdata[:, :, 1]= slice_data_1024[ystart:ystop, xstart:xstop]
    RGBdata[:, :, 2]= slice_data_1025[ystart:ystop, xstart:xstop]

    RGBdata[np.where(RGBdata <= 0)] = 0
    RGBdata[np.where(np.isnan(RGBdata) == True)] = 0
    
    #for _i in np.arange(3):
    #    RGBdata[:, :, _i] = 

    RGBdata = RGBdata/np.max(RGBdata)
    
    H, S, V = RGB_to_HSV_arr(RGBdata[:, :, 0], RGBdata[:, :, 1], RGBdata[:, :, 2])
    logscaleV = np.log10(V)
    logscaleV = (logscaleV - np.min(logscaleV))
    logscaleV = logscaleV/np.max(logscaleV)

    logscaleS = np.log10(S + 0.001)
    logscaleS = (logscaleS - np.nanmin(logscaleS))
    logscaleS = logscaleS/np.nanmax(logscaleS)

    R,G,B = RGB_to_HSV_arr(H, S, logscaleV, reverse=True)

    RGBscaled = np.zeros(RGBdata.shape)
    RGBscaled[:, :, 0] = R
    RGBscaled[:, :, 1] = G
    RGBscaled[:, :, 2] = B
    RGBscaled = RGBscaled/np.nanmax(RGBscaled)
    
    fig = plt.figure(figsize=(10,9))
    axbackground = fig.add_subplot(111)
    axbackground.axis('off')
    
    left, bottom, width, height = [0.0, 0., 1, 1]
    ax0 = fig.add_axes([left, bottom, width, height])
    ax0.plot([0, 1], [0, 1], alpha=0)
    ax0.axvspan(xmin=0.45, xmax=0.95, ymin=0.075, ymax=0.95, facecolor='gray', alpha=0.5)
    ax0.spines["top"].set_visible(False)
    ax0.spines["bottom"].set_visible(False)
    ax0.spines["left"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    #ax1 = fig.add_subplot(211)
    #ax2 = fig.add_subplot(212)
    
    #ax1 = plt.subplot2grid((30, 50), (0, 0), colspan=49, rowspan=15)
    #ax2 = plt.subplot2grid((30, 50), (15, 0), colspan=49, rowspan=15)
    #ax3 = plt.subplot2grid((30, 50), (25, 49), colspan=1, rowspan=6)
    
    nrows = 25
    ncols = 75
    #ax1 = plt.subplot2grid((nrows, ncols), (0, 0), colspan=49, rowspan=12)
    #ax2 = plt.subplot2grid((nrows, ncols), (15, 0), colspan=49, rowspan=12)
    #ax3 = plt.subplot2grid((nrows, ncols), (24, 49), colspan=1, rowspan=6)
    #ax4 = plt.subplot2grid((nrows, ncols), (0, 50), colspan=24, rowspan=12)
    #ax5 = plt.subplot2grid((nrows, ncols), (15, 50), colspan=24, rowspan=12)
    
    left, bottom, width, height = [0.05, 0.5, 0.6, 0.5]
    ax1 = fig.add_axes([left, bottom, width, height])
    left, bottom, width, height = [0.05, 0.05, 0.6, 0.5]
    ax2 = fig.add_axes([left, bottom, width, height])
    left, bottom, width, height = [0.70, 0.5, 0.2, 0.5]
    ax4 = fig.add_axes([left, bottom, width, height])
    left, bottom, width, height = [0.70, 0.05, 0.2, 0.5]
    ax5 = fig.add_axes([left, bottom, width, height])

    ax4.imshow(slice_data_1024_cut_smaller, cmap=cmocean.cm.tempo)
    ax5.imshow(umask_1024_cut_smaller, cmap=cmocean.cm.tempo)

    labelras = np.array([220., 225., 230., 235., 240.])
    labeldecs = np.array([15., 21., 27.])
    labelras_smaller = np.array([220., 225.])
    
    xax, ras, yax, decs = cutouts.get_xylabels_from_ras_decs(labelras, labeldecs, cut_hdr, origincoord=0)
    xax_smaller, ras_smaller, yax_smaller, decs_smaller = cutouts.get_xylabels_from_ras_decs(labelras_smaller, labeldecs, cut_hdr_smaller, origincoord=0)

    #im1 = ax1.imshow(RGBscaled)
    im1 = ax1.imshow(RGBdata)
    #ax1 = cutouts.set_all_labels(ax1, xax, ra_label, yax, dec_label, roundnum=1)
    ax1 = cutouts.set_all_labels(ax1, xax, labelras, yax, labeldecs, roundnum=0, size=12, degrees=True)
    #plt.colorbar(im1)
    
    #for _b in [50, 60]:
    #    xs_blines, ys_blines = cutouts.add_b_lines(_b, w)
    #    ax1.plot(xs_blines, ys_blines, color="gray", alpha=0.2, lw=2)
    #    ax2.plot(xs_blines, ys_blines, color="gray", alpha=0.2, lw=2)
        
    for ax in [ax1, ax2]:
        ax.set_ylim(0, chunky)
        ax.set_xlim(0, chunkx)
    for ax in [ax4, ax5]:
        ax.set_ylim(0, chunkysmaller)
        ax.set_xlim(0, chunkxsmaller)
    
    p857vmin = np.percentile(p857cut[np.where(maskbs_cut > 0.5)], 0.5)
    p857vmax = np.percentile(p857cut[np.where(maskbs_cut > 0.5)], 99.5)
    #im2 = ax2.imshow(p857cut, cmap="Greys_r", norm=LogNorm(vmin=np.min(p857cut), vmax=np.max(p857cut) ))
    im2 = ax2.imshow(p857cut, cmap="afmhot", vmin=p857vmin, vmax=p857vmax)
    #ax2 = cutouts.set_all_labels(ax2, xax, ra_label, yax, dec_label, roundnum=1)
    cutouts.set_all_labels(ax2, xax, labelras, yax, labeldecs, roundnum=0, size=12, degrees=True)
    
    ax4 = cutouts.set_all_labels(ax4, xax_smaller, labelras_smaller, yax_smaller, labeldecs, roundnum=0, size=12, degrees=True)
    ax5 = cutouts.set_all_labels(ax5, xax_smaller, labelras_smaller, yax_smaller, labeldecs, roundnum=0, size=12, degrees=True)
    #plt.colorbar(im2, cax=ax3)
    ax4.yaxis.set_label_position("right")
    ax5.yaxis.set_label_position("right")

    ax1.set_xticks([])

    ny, nx, nz = RGBscaled.shape
    #for ax in [ax1, ax2, ax4, ax5]:
    #    ax.set_ylim(0, ny)

def make_grid():
    shiftright = 7
    vwhitespace = 2
    hwhitespace = 1
    #ax3 = plt.subplot2grid((nrows, ncolumns), (0, _i*11 + hwhitespace + shiftright + 1), rowspan=5, colspan=8)
    #ax1 = plt.subplot2grid((nrows, ncolumns), (5 + 1*vwhitespace, _i*11+ hwhitespace+ shiftright), rowspan=10, colspan=10)
    #ax2 = plt.subplot2grid((nrows, ncolumns), (15 + 2*vwhitespace, _i*11+ hwhitespace+ shiftright), rowspan=10, colspan=10)
    #ax4 = plt.subplot2grid((nrows, ncolumns), (25 + 3*vwhitespace, _i*11+ hwhitespace+ shiftright), rowspan=10, colspan=10)
    
    n_slices = 6
    axwidth = 0.9/(n_slices+1)
    axheight = 0.2 #0.8/3.5
    axspacing = 0.05
    
    axtop = axspacing*2 + axheight
    
    fig = plt.figure(figsize=(10, 6))
    
    left, bottom, width, height = [0.1, .8, axwidth, 0.15]
    ax3 = fig.add_axes([left, bottom, width, height])
    left, bottom, width, height = [0.1, .55, axwidth, axheight]
    ax1 = fig.add_axes([left, bottom, width, height])
    left, bottom, width, height = [0.1, .3, axwidth, axheight]
    ax2 = fig.add_axes([left, bottom, width, height])
    left, bottom, width, height = [0.1, .05, axwidth, axheight]
    ax4 = fig.add_axes([left, bottom, width, height])
    
def make_grid2():

    fig = plt.figure(figsize=(19, 4))
    gs = gridspec.GridSpec(4, 6,
                       width_ratios=[1, 1, 1, 1, 1, 1],
                       height_ratios=[1, 2, 2, 2]
                       )
    for _i in np.arange(6*4):
        ax = plt.subplot(gs[_i])

def plot_one_USM_stack():
    """ talk figure """
    DR2_Wide_slice_root = "/Users/susanclark/Dropbox/DR2 Full Sky Wide Maps/"
    submonopole = 0.64
    
    cmap = cmocean.cm.deep
    
    starty=1000
    stopy=1800
    startx=7800
    stopx=8600
    
    centervel = 1024
    _i = 1
    startvel = centervel - _i
    stopvel = centervel + _i
    
    velkms = gvh.galfa_name_dict["1024"]
    slice_fn = DR2_Wide_slice_root+"GALFA_HI_W_S{}_V{}kms.fits".format(1024, velkms)
    big_hdr = fits.getheader(slice_fn)
    

    
    #cut_hdr, narrowvelchunk = cutouts.xycutout_data(narrowvelslice, big_hdr, xstart=startx, xstop=stopx, ystart=starty, ystop=stopy)
    #xax, ra_label = cutouts.get_xlabels_ra(cut_hdr, skip = 200.0, origincoord=0)
    #yax, dec_label = cutouts.get_ylabels_dec(cut_hdr, skip = 200.0, origincoord=0)
    
    vels = [makestring(v) for v in np.arange(startvel, stopvel+1)]
    USMslice = load_USM_map(vels=vels, fwhm=30, zeroed=True, Narrow=False, addnoise=False)
    velslice = load_channel_map(vels, Narrow=False, addnoise=False, noisefactor=1)
    stackdata_P857 = load_USM_stack_data(datatype="P3_857", fwhm=30, chanstart=vels[0], chanstop=vels[-1], Narrow=False, bstart=30, bstop=90, randorient=True, cubelen=181)
    stackdata_weights = load_USM_stack_data(datatype="weights", fwhm=30, chanstart=vels[0], chanstop=vels[-1], Narrow=False, bstart=30, bstop=90, randorient=True, cubelen=181)
    
    stackdata_adj = (stackdata_P857 + stackdata_weights*submonopole)/stackdata_weights
    
    print(vels)
    print(len(vels)*0.72)

    USMchunk = USMslice[starty:stopy, startx:stopx]
    velchunk = velslice[starty:stopy, startx:stopx]
        
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    print(np.nanmin(USMchunk), np.nanmax(USMchunk))
    imvel = ax1.imshow(velchunk, cmap=cmap)
    imUSM = ax2.imshow(USMchunk, cmap=cmap)
    imstack = ax3.imshow(stackdata_adj, cmap="magma")
    
    for _ax, _im in zip([ax1, ax2, ax3], [imvel, imUSM, imstack]):
        plt.colorbar(_im, ax= _ax, orientation = 'horizontal')
        _ax.set_xticks([])
        _ax.set_yticks([])
    
    
def plot_postage_stamps_stacks():
    DR2_Wide_slice_root = "/Users/susanclark/Dropbox/DR2 Full Sky Wide Maps/"

    nhi90map = fits.getdata("/Users/susanclark/Dropbox/NHImaps/GALFA-HI_NHISRCORR_VLSR-90+90kms.fits")
    p857map = fits.getdata("/Users/susanclark/Dropbox/Planck/HFI_SkyMap_857_2048_R3.01_ONGALFAHI.fits")
    submonopole = 0.64
    if submonopole is not None:
        p857map -= submonopole
        
    all_galfa_vels_sort, all_mask_I_v_sort = get_I_v_mask()
    all_mask_I_v_sort = all_mask_I_v_sort/np.max(all_mask_I_v_sort)
    
    alphaslice=0.5

    halfvel = 0.72/2
        
    cmap = cmocean.cm.deep
    
    centervel = 1024
    n_slices = 5#6
    
    starty=400
    stopy=2000
    startx= 8000
    stopx = 9600
    #starty=400
    #stopy=1800
    #startx= 9800
    #stopx = 11400
    starty=1000
    stopy=1800
    startx=7000
    stopx=8500    
    #starty=1000
    #stopy=1800
    #startx=6700
    #stopx=7500
    starty=1000
    stopy=1800
    startx=7800
    stopx=8600
    #startx=7000
    #stopx=7800
    
    # super straight fiber-y region
    #starty=1400
    #stopy=2000
    #startx=8600
    #stopx=9600
    
    bstart = 30
    NHIcut = True
    NHImin = 0
    NHImax = 8
    maskbs = make_mask_2d(bstart=bstart, bstop=90, PS=True, NHIcut=NHIcut, NHImin=NHImin, NHImax=NHImax) 
    
    narrowUSMslice = load_USM_map(vels=["1024"], fwhm=30, zeroed=True, Narrow=True, addnoise=False)
    narrowvelslice = load_channel_map(["1024"], Narrow=True, addnoise=False, noisefactor=1)
    narrowUSMslice = narrowUSMslice/np.nanmax(narrowUSMslice[np.where(maskbs > 0)])
    narrowUSMchunk = narrowUSMslice[starty:stopy, startx:stopx]
    narrowvelchunk = narrowvelslice[starty:stopy, startx:stopx]
    narrowstackdata_P857 = load_USM_stack_data(datatype="P3_857", fwhm=30, chanstart="1024", chanstop="1024", Narrow=True, bstart=30, bstop=90, randorient=True, cubelen=181)
    narrowstackdata_weights = load_USM_stack_data(datatype="weights", fwhm=30, chanstart="1024", chanstop="1024", Narrow=True, bstart=30, bstop=90, randorient=True, cubelen=181)
    narrowstackadj = (narrowstackdata_P857 + submonopole*narrowstackdata_weights)/narrowstackdata_weights
    
    chunktrim=40
    narrowstackcorner = narrowstackadj[:chunktrim, :chunktrim]
    narrowstackchunk = narrowstackadj[chunktrim:-chunktrim, chunktrim:-chunktrim]
    
    velkms = gvh.galfa_name_dict["1024"]
    slice_fn = DR2_Wide_slice_root+"GALFA_HI_W_S{}_V{}kms.fits".format(1024, velkms)
    big_hdr = fits.getheader(slice_fn)
    
    cut_hdr, narrowvelchunk = cutouts.xycutout_data(narrowvelslice, big_hdr, xstart=startx, xstop=stopx, ystart=starty, ystop=stopy)
    xax, ra_label = cutouts.get_xlabels_ra(cut_hdr, skip = 200.0, origincoord=0)
    yax, dec_label = cutouts.get_ylabels_dec(cut_hdr, skip = 200.0, origincoord=0)
    
    startvel_narrow_chan = 0.1 - 0.18/2
    stopvel_narrow_chan = 0.1 + 0.18/2
    
    fig = plt.figure(figsize=(17, 8))
    #ax0 = fig.add_subplot(3, n_slices+1, 1)
    
    nrows = 41
    ncolumns = 76 #(n_slices+1)*10 + 8 + 6
    
    # find min and max of data
    allmin_vels = np.zeros(n_slices+1) 
    allmax_vels = np.zeros(n_slices+1) 
    
    allmin_USMs = np.zeros(n_slices+1) 
    allmax_USMs = np.zeros(n_slices+1)

    allmin_stacks = np.zeros(n_slices+1) 
    allmax_stacks = np.zeros(n_slices+1)
    
    allmin_stackoutside = np.zeros(n_slices+1)
    
    allmin_vels[0] = np.min(narrowvelchunk)
    allmax_vels[0] = np.max(narrowvelchunk)
    allmin_USMs[0] = np.min(narrowUSMchunk)
    allmax_USMs[0] = np.max(narrowUSMchunk)
    allmin_stacks[0] = np.min(narrowstackchunk)
    allmax_stacks[0] = np.max(narrowstackchunk)
    
    #allmin_stackoutside[0] = np.median(narrowstackcorner)
    allmin_stackoutside[0] = np.nanmedian(narrowstackadj[chunktrim, :])
    
    for _i in np.arange(n_slices):
        startvel = centervel - _i
        stopvel = centervel + _i
        vels = [makestring(v) for v in np.arange(startvel, stopvel+1)]
        USMslice = load_USM_map(vels=vels, fwhm=30, zeroed=True, Narrow=False, addnoise=False)
        velslice = load_channel_map(vels, Narrow=False, addnoise=False, noisefactor=1)
        stackdata_P857 = load_USM_stack_data(datatype="P3_857", fwhm=30, chanstart=vels[0], chanstop=vels[-1], Narrow=False, bstart=30, bstop=90, randorient=True, cubelen=181)
        stackdata_weights = load_USM_stack_data(datatype="weights", fwhm=30, chanstart=vels[0], chanstop=vels[-1], Narrow=False, bstart=30, bstop=90, randorient=True, cubelen=181)
        
        # need to correct for number of channels added together
        print("len vels", len(vels))
        velslice = velslice/len(vels)
        
        stackdata_adj = (stackdata_P857 + stackdata_weights*submonopole)/stackdata_weights
        
        USMslice = USMslice/np.nanmax(USMslice[np.where(maskbs > 0)])
        
        allmin_USMs[_i+1] = np.min(USMslice[starty:stopy, startx:stopx])
        allmax_USMs[_i+1] = np.max(USMslice[starty:stopy, startx:stopx])
            
        allmin_vels[_i+1] = np.min(velslice[starty:stopy, startx:stopx])
        allmax_vels[_i+1] = np.max(velslice[starty:stopy, startx:stopx])
        
        allmin_stacks[_i+1] = np.min(stackdata_adj[chunktrim:-chunktrim, chunktrim:-chunktrim])
        allmax_stacks[_i+1] = np.max(stackdata_adj[chunktrim:-chunktrim, chunktrim:-chunktrim])
        
        #allmin_stackoutside[_i+1] = np.nanmedian(stackdata_adj[:chunktrim, :chunktrim])
        allmin_stackoutside[_i+1] = np.nanmedian(stackdata_adj[chunktrim, :])
        
    vmin_USM = np.min(allmin_USMs)
    vmax_USM = np.max(allmax_USMs)
    vmin_vel = np.min(allmin_vels)
    vmax_vel = np.max(allmax_vels)
    vmin_stack = np.min(allmin_stacks - allmin_stackoutside)
    vmax_stack = np.max(allmax_stacks - allmin_stackoutside)
        
    print(allmin_vels, allmax_vels)
    print(allmin_USMs, allmax_USMs)
    print(allmin_stacks, allmax_stacks)
    print("diff stacks:", allmax_stacks - allmin_stacks)
    print("diff stacks:", allmax_stacks - allmin_stackoutside)
    
    for _i in np.arange(n_slices+1):
        startvel = centervel - _i
        stopvel = centervel + _i
        vels = [makestring(v) for v in np.arange(startvel, stopvel+1)]
        print("vel 0 = {}".format(vels[0]))
        
        startvel = np.float(gvh.galfa_name_dict[vels[0]])
        startvel_indx = np.where(all_galfa_vels_sort == startvel)[0][0]
        stopvel = np.float(gvh.galfa_name_dict[vels[-1]])
        stopvel_indx = np.where(all_galfa_vels_sort == stopvel)[0][0]
        
        startvel_chan = startvel - halfvel
        stopvel_chan = stopvel + halfvel
        
        USMslice = load_USM_map(vels=vels, fwhm=30, zeroed=True, Narrow=False, addnoise=False)
        USMslice = USMslice/np.nanmax(USMslice[np.where(maskbs > 0)])
        velslice = load_channel_map(vels, Narrow=False, addnoise=False, noisefactor=1)
        stackdata_P857 = load_USM_stack_data(datatype="P3_857", fwhm=30, chanstart=vels[0], chanstop=vels[-1], Narrow=False, bstart=30, bstop=90, randorient=True, cubelen=181)
        stackdata_weights = load_USM_stack_data(datatype="weights", fwhm=30, chanstart=vels[0], chanstop=vels[-1], Narrow=False, bstart=30, bstop=90, randorient=True, cubelen=181)
        stackchunk = (stackdata_P857 + submonopole*stackdata_weights)/stackdata_weights
        
        stackchunk = stackchunk[chunktrim:-chunktrim, chunktrim:-chunktrim]
        nystack, nxstack = stackchunk.shape
        
        velslice = velslice/len(vels)
        
        USMchunk = USMslice[starty:stopy, startx:stopx]
        velchunk = velslice[starty:stopy, startx:stopx]
        NHIchunk = nhi90map[starty:stopy, startx:stopx]
        
        
        shiftright = 7
        vwhitespace = 2
        hwhitespace = 0
        ax3 = plt.subplot2grid((nrows, ncolumns), (0, _i*11 + hwhitespace + shiftright + 1 + 3), rowspan=5, colspan=8)
        ax1 = plt.subplot2grid((nrows, ncolumns), (5 + 1*vwhitespace, _i*11+ hwhitespace+ shiftright+ 3), rowspan=10, colspan=10)
        ax2 = plt.subplot2grid((nrows, ncolumns), (15 + 2*vwhitespace, _i*11+ hwhitespace+ shiftright+ 3), rowspan=10, colspan=10)
        ax4 = plt.subplot2grid((nrows, ncolumns), (25 + 3*vwhitespace, _i*11+ hwhitespace+ shiftright+ 3), rowspan=10, colspan=10)
        
        mid1 = 5 + 1*vwhitespace + 5
        mid2 = 15 + 2*vwhitespace + 5
        mid3 = 25 + 3*vwhitespace + 5
        
        print(0, _i*11 + hwhitespace + shiftright)
        print(25 + 3*vwhitespace, _i*10+ hwhitespace+ shiftright)
        
        axwidth = 0.9/(n_slices+1)
        axheight = 0.9/4
        
        left, bottom, width, height = [0.1, .7, axwidth, axheight]
        #ax1 = fig.add_axes([left, bottom, width, height])
        
        # leftmost, Narrow slice
        if _i == 0:
            ax1.imshow(narrowvelchunk, cmap=cmap, vmin=vmin_vel, vmax=0.9*vmax_vel)
            ax2.imshow(narrowUSMchunk, cmap=cmap, vmin=vmin_USM, vmax=vmax_USM)
            ax4.imshow(narrowstackchunk - allmin_stackoutside[0], cmap="magma", vmin=vmin_stack, vmax=vmax_stack)
            
            nxchunk, nychunk = narrowvelchunk.shape
            
            for _ax in [ax1, ax2]:
                _ax.set_aspect('equal')
                
                _ax.set_ylim(0, nychunk)
                
                labelras = np.array([226., 220.])
                labeldecs = np.array([16., 21., 26.])
                
                xax, ras, yax, decs = cutouts.get_xylabels_from_ras_decs(labelras, labeldecs, cut_hdr, origincoord=0)

                _ax = cutouts.set_all_labels(_ax, xax, ra_label, yax, dec_label, roundnum=0, size=10, degrees=True)

            ylabelrot=360
            ylabelpad=35
            #ax1.set_ylabel(r"$\mathrm{Channel}$ $\mathrm{Map}$", rotation=ylabelrot, labelpad=ylabelpad)
            #ax2.set_ylabel(r"$\mathrm{Unsharp}$ $\mathrm{Mask}$", rotation=ylabelrot, labelpad=ylabelpad)
            #ax4.set_ylabel(r"$\mathrm{Stacked}$ $I_{857}$", rotation=ylabelrot, labelpad=ylabelpad)

            ax4.set_yticks([0, nystack/2., nystack-1])
            ax4.set_xticks([0, nxstack/2., nxstack-1])
            ax4.set_yticklabels([r"$+50'$", r"$0'$", r"$-50'$"])
            ax4.set_xticklabels([r"$-50'$", r"$0'$", r"$+50'$"])
            
            N_galfa_vels = len(all_galfa_vels_sort)
            all_galfa_vels_sort_interp = np.zeros(N_galfa_vels*2)
            all_galfa_vels_sort_interp[:N_galfa_vels] = all_galfa_vels_sort
            all_galfa_vels_sort_interp[N_galfa_vels:] = all_galfa_vels_sort + (all_galfa_vels_sort[1] - all_galfa_vels_sort[0])/2.0
            all_galfa_vels_sort_interp = np.sort(all_galfa_vels_sort_interp)
            
            all_mask_I_v_sort_interp = np.interp(all_galfa_vels_sort_interp, all_galfa_vels_sort, all_mask_I_v_sort)

            all_mask_I_v_sort = copy.copy(all_mask_I_v_sort_interp)
            all_galfa_vels_sort = copy.copy(all_galfa_vels_sort_interp)
            ax3.fill_between(all_galfa_vels_sort, all_mask_I_v_sort, facecolor='gray', alpha=0.3, interpolate=True)
            ax3.fill_between(all_galfa_vels_sort, all_mask_I_v_sort, where=(all_galfa_vels_sort >= startvel_narrow_chan) , facecolor='white', alpha=1, interpolate=True)
            ax3.fill_between(all_galfa_vels_sort, all_mask_I_v_sort, where=(all_galfa_vels_sort >= startvel_narrow_chan) , facecolor='teal', alpha=alphaslice, interpolate=True)
            ax3.fill_between(all_galfa_vels_sort, all_mask_I_v_sort, where=(all_galfa_vels_sort > stopvel_narrow_chan) , facecolor='white', alpha=1, interpolate=True)
            ax3.fill_between(all_galfa_vels_sort, all_mask_I_v_sort, where=(all_galfa_vels_sort >= stopvel_narrow_chan) , facecolor='gray', alpha=0.3, interpolate=True)


            ax3.set_xlim(-25, 25)
            ax3.spines["top"].set_visible(False)
            ax3.spines["bottom"].set_visible(False)
            ax3.spines["left"].set_visible(False)
            ax3.spines["right"].set_visible(False)
            #ax3.set_ylim(0, 2.5)
            ax3.set_yticks([])    
            ax3.tick_params(axis='both', which='both',length=0)
            ax3.xaxis.set_tick_params(pad=-1)
            #ax3.text(-35,-1,r"$v_{lsr}$ $\mathrm{[km/s]}$")
            plt.gcf().text(0.186, 0.773, r"$v_{lsr}$ $\mathrm{[km/s]}$", fontsize=12)

        #if (_i > 0) & (_i < n_slices):  
        if _i > 0:  
            print("min max USM", vmin_USM, vmax_USM, "minmax chunk", np.nanmin(USMchunk), np.nanmax(USMchunk))
            print("NEW minvel, max vel: ", vmin_vel, vmax_vel)
            print("NEW min, max velchunk: ", np.min(velchunk), np.max(velchunk))
            imvel = ax1.imshow(velchunk, cmap=cmap, vmin=vmin_vel, vmax=0.9*vmax_vel)
            imUSM = ax2.imshow(USMchunk, cmap=cmap, vmin=vmin_USM, vmax=vmax_USM)
            imstack = ax4.imshow(stackchunk - allmin_stackoutside[_i], cmap="magma", vmin=vmin_stack, vmax=vmax_stack)
            #ax4.plot(radprof_stack)
            
            print("stack max = {}".format(np.max(stackchunk)))
            
            for _ax in [ax1, ax2, ax4]:
                _ax.set_aspect('equal')
                _ax.set_xticks([])
                _ax.set_yticks([])
            
            for _ax in [ax1, ax2]:
                _ax.set_ylim(0, nychunk) # fixed

            ax3.fill_between(all_galfa_vels_sort, all_mask_I_v_sort, facecolor='gray', alpha=0.3, interpolate=True)
            ax3.fill_between(all_galfa_vels_sort, all_mask_I_v_sort, where=(all_galfa_vels_sort >= startvel_chan) , facecolor='white', alpha=1, interpolate=True)
            ax3.fill_between(all_galfa_vels_sort, all_mask_I_v_sort, where=(all_galfa_vels_sort >= startvel_chan) , facecolor='teal', alpha=alphaslice, interpolate=True)
            ax3.fill_between(all_galfa_vels_sort, all_mask_I_v_sort, where=(all_galfa_vels_sort > stopvel_chan) , facecolor='white', alpha=1, interpolate=True)
            ax3.fill_between(all_galfa_vels_sort, all_mask_I_v_sort, where=(all_galfa_vels_sort >= stopvel_chan) , facecolor='gray', alpha=0.3, interpolate=True)

            ax3.set_xlim(-25, 25)
            ax3.spines["top"].set_visible(False)
            ax3.spines["bottom"].set_visible(False)
            ax3.spines["left"].set_visible(False)
            ax3.spines["right"].set_visible(False)
            #ax3.set_ylim(0, 2.5)
            ax3.set_yticks([])
            ax3.set_xticks([])
            
        #for _ax in [ax1, ax2, ax4]:
        #    _ax.set_ylim(0, nystack)
        
        # Show total column
        #if _i == n_slices:
        #    ax1.imshow(NHIchunk, cmap=cmap)
    #plt.subplots_adjust(left=0.1, right=0.95, top=0.99, bottom=0.1, wspace=0.1, hspace=0.3)
    cbarax1 = plt.subplot2grid((nrows, ncolumns), (mid1, 0), rowspan=2, colspan=6)
    cbarax2 = plt.subplot2grid((nrows, ncolumns), (mid2, 0), rowspan=2, colspan=6)
    cbarax4 = plt.subplot2grid((nrows, ncolumns), (mid3, 0), rowspan=2, colspan=6)
    
    cbarvel = plt.colorbar(imvel, cax=cbarax1, orientation='horizontal')
    cbarUSM = plt.colorbar(imUSM, cax=cbarax2, orientation='horizontal')
    cbarstack = plt.colorbar(imstack, cax=cbarax4, orientation='horizontal')
    
    cbarlabelsize = 12
    cbarvel.ax.set_xlabel(r"$T_b$ $\mathrm{[K]}$", size=cbarlabelsize)
    cbarUSM.ax.set_xlabel(r"$w^{\delta v}$", size=cbarlabelsize)
    cbarstack.ax.set_xlabel(r"$\Delta I_{857}$", size=cbarlabelsize)
    
    cbarax1.set_title(r"$\mathrm{Channel}$ $\mathrm{Map}$")
    cbarax2.set_title(r"$\mathrm{Unsharp}$ $\mathrm{Mask}$")
    cbarax4.set_title(r"$\mathrm{Stacked}$ $I_{857}$")
    
    for _cbar in [cbarvel, cbarUSM, cbarstack]:
        _cbar.outline.set_visible(False)
        #_cbarax.spines["top"].set_visible(False)
        #_cbarax.spines["bottom"].set_visible(False)
        #_cbarax.spines["left"].set_visible(False)
        #_cbarax.spines["right"].set_visible(False)
        
def get_sim_data(dtype="tb", v=True, n=True, t=True, mach=0.5, returnfn=False):
    vstr = nstr = tstr = ""
    
    if v:
        vstr = "_v"
    if n:
        nstr = "_n"
    if t:
        tstr = "_t"
    if mach == 0.5:
        machstr = "05"
    else:
        machstr = str(np.int(mach))
    
    sims_root = "/Users/susanclark/Projects/FiberDGR/data/transfer_1258637_files_029b63a8/sims_isothermal_ramses/"
    sim_fn = sims_root+"{}{}{}{}_mach{}.fits".format(dtype, vstr, nstr, tstr, machstr)
    sim_data = fits.getdata(sim_fn)
    
    if returnfn:
        return sim_fn
    else:
        return sim_data
        
def plot_sims_slices():
    
    tb_n_v_t_5 = get_sim_data(dtype="tb", v=True, n=True, t=True, mach=5)
    tb_n_v_t_05 = get_sim_data(dtype="tb", v=True, n=True, t=True, mach=0.5)
    tb_n_v_t_1 = get_sim_data(dtype="tb", v=True, n=True, t=True, mach=1)
    
    fig = plt.figure(figsize=(10, 6))
    
    block_widths = 0.28
    block_heights = 0.4
    
    left, bottom, width, height = [0.2, 0.5, block_widths, block_heights]
    ax1 = fig.add_axes([left, bottom, width, height])
    left, bottom, width, height = [0.2, 0.05, block_widths, block_heights]
    ax2 = fig.add_axes([left, bottom, width, height])
    left, bottom, width, height = [0.55, 0.5, block_widths, block_heights]
    ax3 = fig.add_axes([left, bottom, width, height])
    left, bottom, width, height = [0.55, 0.05, block_widths, block_heights]
    ax4 = fig.add_axes([left, bottom, width, height])
    
    left, bottom, width, height = [0.05, 0.5, 0.1, 0.1]
    ax1_sub = fig.add_axes([left, bottom, width, height])
    left, bottom, width, height = [0.05, 0.05, 0.1, 0.1]
    ax2_sub = fig.add_axes([left, bottom, width, height])
    left, bottom, width, height = [0.85, 0.5, 0.1, 0.1]
    ax3_sub = fig.add_axes([left, bottom, width, height])
    left, bottom, width, height = [0.85, 0.05, 0.1, 0.1]
    ax4_sub = fig.add_axes([left, bottom, width, height])
    
    nz_5, ny_5, nx_5 = tb_n_v_t_5.shape
    nz_05, ny_05, nx_05 = tb_n_v_t_05.shape
    
    int_tb_n_v_t_5 = np.sum(np.sum(tb_n_v_t_5, axis=1), axis=1)
    int_tb_n_v_t_05 = np.sum(np.sum(tb_n_v_t_05, axis=1), axis=1)
    
    ax1_sub.fill(int_tb_n_v_t_5, facecolor='gray', alpha=0.5)
    ax1_sub.vlines(np.int(nz_5/2), 0, np.max(int_tb_n_v_t_5))
    ax2_sub.fill(int_tb_n_v_t_5, facecolor='black')
    
    ax3_sub.fill(int_tb_n_v_t_05, facecolor='gray', alpha=0.5)
    ax3_sub.vlines(np.int(nz_05/2), 0, np.max(int_tb_n_v_t_05))
    ax4_sub.fill(int_tb_n_v_t_05, facecolor='black')
    
    cmap = cmocean.cm.matter
    
    ax1.imshow(np.sum(tb_n_v_t_5, axis=0), cmap=cmap)
    ax2.imshow(tb_n_v_t_5[np.int(nz_5/2), :, :], cmap=cmap)
    ax3.imshow(np.sum(tb_n_v_t_05, axis=0), cmap=cmap)
    ax4.imshow(tb_n_v_t_05[np.int(nz_05/2), :, :], cmap=cmap)
    
    titlesize = 18
    ax1.set_title(r"$\mathcal{M}_s = 5$", size=titlesize)
    ax3.set_title(r"$\mathcal{M}_s = 0.5$", size=titlesize)
    
    for _ax in [ax1_sub, ax2_sub, ax3_sub, ax4_sub]:
        _ax.spines["top"].set_visible(False)
        _ax.spines["bottom"].set_visible(False)
        _ax.spines["left"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        _ax.set_xticks([])
        _ax.set_yticks([])
        _ax.set_xlim(50, 150)
        
    for _ax in [ax1, ax2, ax3, ax4]:
        _ax.set_xticks([])
        _ax.set_yticks([])
        
        
def plot_3sims():
    
    #cmap = cm.coolwarm #m.Spectral #cm.RdYlBu #cmocean.cm.balance
    #cmap = LinearSegmentedColormap.from_list(
    #    'custom_BGR', ['blue', 'green', 'red'], N=256)
    cmap = LinearSegmentedColormap.from_list(
        'custom_BGR', [cm.Blues(0.8), cm.Purples(0.8), cm.Reds(0.8)], N=256)
    im0 =  plt.imshow(np.array([[0,1]]), cmap=cmap)
    im0.set_visible(False)
    
    fig = plt.figure(figsize=(12, 6))
    nrows = 25
    ncols = 75
    slicerowspan = 10
    slicecolspan = 20
    ax0 = plt.subplot2grid((nrows, ncols), (0, 0), colspan=ncols, rowspan=2)
    ax1 = plt.subplot2grid((nrows, ncols), (5, 5), colspan=slicecolspan, rowspan=slicerowspan)
    ax2 = plt.subplot2grid((nrows, ncols), (15, 5), colspan=slicecolspan, rowspan=slicerowspan)
    ax3 = plt.subplot2grid((nrows, ncols), (5, 30), colspan=slicecolspan, rowspan=slicerowspan)
    ax4 = plt.subplot2grid((nrows, ncols), (15, 30), colspan=slicecolspan, rowspan=slicerowspan)
    ax5 = plt.subplot2grid((nrows, ncols), (5, 55), colspan=slicecolspan, rowspan=slicerowspan)
    ax6 = plt.subplot2grid((nrows, ncols), (15, 55), colspan=slicecolspan, rowspan=slicerowspan)
    
    subrowspan=2
    subcolspan=5
    subcolnudge = 0
    ax1_sub = plt.subplot2grid((nrows, ncols), (5, 0 + subcolnudge), colspan=subcolspan, rowspan=subrowspan)
    ax2_sub = plt.subplot2grid((nrows, ncols), (15, 0 + subcolnudge), colspan=subcolspan, rowspan=subrowspan)
    ax3_sub = plt.subplot2grid((nrows, ncols), (5, 25 + subcolnudge), colspan=subcolspan, rowspan=subrowspan)
    ax4_sub = plt.subplot2grid((nrows, ncols), (15, 25 + subcolnudge), colspan=subcolspan, rowspan=subrowspan)
    ax5_sub = plt.subplot2grid((nrows, ncols), (5, 50 + subcolnudge), colspan=subcolspan, rowspan=subrowspan)
    ax6_sub = plt.subplot2grid((nrows, ncols), (15, 50 + subcolnudge), colspan=subcolspan, rowspan=subrowspan)
    
    for _ax in [ax1_sub, ax2_sub, ax3_sub, ax4_sub, ax5_sub, ax6_sub]:
        pos1 = _ax.get_position() # get the original position 
        pos2 = [pos1.x0 + 0.02, pos1.y0,  pos1.width, pos1.height] 
        _ax.set_position(pos2) # set a new position
        
    titlesize = 18
    ax1.set_title(r"$\mathcal{M}_s = 5$", size=titlesize)
    ax3.set_title(r"$\mathcal{M}_s = 1$", size=titlesize)
    ax5.set_title(r"$\mathcal{M}_s = 0.5$", size=titlesize)
    
    #ax0.text(0.5, 0.5, "Test", size=30, va="center", ha="center",
    #    bbox=dict(boxstyle=custom_box_style(0.5, 0.5, 10, 10, 1), alpha=0.2))
    
    leftpt = 0.
    rightpt = 1.
    patches = []
    polygon = Polygon([[leftpt, 0.5], [leftpt+0.02, 1.], [rightpt-0.01, 1.], [rightpt, 0.5],
                       [rightpt-0.01, 0.], [leftpt+0.02, 0.], [leftpt, 0.5 ]], closed=True)
    patches.append(polygon)
    p = PatchCollection(patches, alpha=0.4, color='orange')
    ax0.add_collection(p)
    
    #from matplotlib.path import Path
    #from matplotlib.patches import PathPatch

    #path = Path([[leftpt, 0.5], [leftpt+0.02, 1.], [rightpt-0.01, 1.], [rightpt, 0.5],
    #                   [rightpt-0.01, 0.], [leftpt+0.02, 0.], [leftpt, 0.5 ]])
    #patch = PathPatch(path, facecolor='none')
    #ax0.add_patch(patch) 
    #Z, Z2 = np.meshgrid(np.linspace(0,1), np.linspace(0,1))
    #im = ax0.imshow(Z-Z2, interpolation='bilinear', cmap=plt.cm.RdYlGn,
    #                origin='lower', extent=[0, 1, 0, 1],
    #                clip_path=patch, clip_on=True)
    #im.set_clip_path(patch)
    

    cbar = plt.colorbar(im0, cax=ax0, extend='both', extendfrac=0.03, orientation='horizontal')
    cbar.outline.set_visible(False)
    cbar.set_ticks([])
    #cbar.solids.set_rasterized(False)
    
    textcolor='white'
    textsize=13
    ax0.text(s=r"$\mathrm{\bf dynamically}$ $\mathrm{\bf correlated}$", x=0.1, y=0.5, horizontalalignment='center', verticalalignment='center', color=textcolor, size=textsize)
    ax0.text(s=r"$\mathrm{\bf thermodynamically}$ $\mathrm{\bf correlated}$", x=0.86, y=0.5, horizontalalignment='center', verticalalignment='center', color=textcolor, size=textsize)
    
    tb_n_v_t_5 = get_sim_data(dtype="tb", v=True, n=True, t=True, mach=5)
    tb_n_v_t_05 = get_sim_data(dtype="tb", v=True, n=True, t=True, mach=0.5)
    tb_n_v_t_1 = get_sim_data(dtype="tb", v=True, n=True, t=True, mach=1)
    
    nz_5, ny_5, nx_5 = tb_n_v_t_5.shape
    nz_1, ny_1, nx_1 = tb_n_v_t_1.shape
    nz_05, ny_05, nx_05 = tb_n_v_t_05.shape
    
    int_tb_n_v_t_5 = np.sum(np.sum(tb_n_v_t_5, axis=1), axis=1)
    int_tb_n_v_t_1 = np.sum(np.sum(tb_n_v_t_1, axis=1), axis=1)
    int_tb_n_v_t_05 = np.sum(np.sum(tb_n_v_t_05, axis=1), axis=1)
    
    #clip_cmap1 = cmocean.tools.crop_by_percent(cmap, 50, which='max')
    #clip_cmap2 = cmocean.tools.crop_by_percent(cmap, 33.3, which='both')
    #clip_cmap3 = cmocean.tools.crop_by_percent(cmap, 66.6, which='min')
    
    # Remove the middle 40% of the RdBu_r colormap
    #interval_1 = np.hstack([np.linspace(0, 0.33), np.linspace(0.6, 1)])
    interval_1 = np.linspace(0, 0.4)
    colors1 = cmap(interval_1)
    clip_cmap1 = LinearSegmentedColormap.from_list('name', colors1)
    
    interval_2 = np.linspace(0.3, 0.7)
    colors2 = cmap(interval_2)
    clip_cmap2 = LinearSegmentedColormap.from_list('name', colors2)
    
    interval_3 = np.linspace(0.6, 1.)
    colors3 = cmap(interval_3)
    clip_cmap3 = LinearSegmentedColormap.from_list('name', colors3, N=256)
    
    clip_cmap1 = "Blues"
    clip_cmap2 = "Purples"
    clip_cmap3 = "Reds"

    ax2.imshow(np.sum(tb_n_v_t_5, axis=0), cmap=clip_cmap1)
    ax1.imshow(tb_n_v_t_5[np.int(nz_5/2), :, :], cmap=clip_cmap1)
    ax4.imshow(np.sum(tb_n_v_t_1, axis=0), cmap=clip_cmap2)
    ax3.imshow(tb_n_v_t_1[np.int(nz_05/2), :, :], cmap=clip_cmap2)
    ax6.imshow(np.sum(tb_n_v_t_05, axis=0), cmap=clip_cmap3)
    ax5.imshow(tb_n_v_t_05[np.int(nz_05/2), :, :], cmap=clip_cmap3)
    
    ax1_sub.fill(int_tb_n_v_t_5, facecolor='gray', alpha=0.5)
    ax1_sub.vlines(np.int(nz_5/2), 0, np.max(int_tb_n_v_t_5))
    ax2_sub.fill(int_tb_n_v_t_5, facecolor='black')
    
    ax3_sub.fill(int_tb_n_v_t_1, facecolor='gray', alpha=0.5)
    ax3_sub.vlines(np.int(nz_1/2), 0, np.max(int_tb_n_v_t_1))
    ax4_sub.fill(int_tb_n_v_t_1, facecolor='black')
    
    ax5_sub.fill(int_tb_n_v_t_05, facecolor='gray', alpha=0.5)
    ax5_sub.vlines(np.int(nz_05/2), 0, np.max(int_tb_n_v_t_05))
    ax6_sub.fill(int_tb_n_v_t_05, facecolor='black')
    
    for _ax in [ax1_sub, ax2_sub, ax3_sub, ax4_sub, ax5_sub, ax6_sub]:
        _ax.spines["top"].set_visible(False)
        _ax.spines["bottom"].set_visible(False)
        _ax.spines["left"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        _ax.patch.set_alpha(0.0)
        _ax.set_xticks([])
        _ax.set_yticks([])
        #_ax.set_xlim(50, 150)
        _ax.set_xlim(50, 150)
    for _ax in [ax3_sub, ax4_sub, ax5_sub, ax6_sub]:
        #_ax.set_xlim(60, 120)
        #_ax.set_xlim(55, 125)
        _ax.set_xlim(75, 125)
        
    ax0.spines["top"].set_visible(False)
    ax0.spines["bottom"].set_visible(False)
    ax0.spines["left"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.set_xticks([])
    ax0.set_yticks([])
        
    for _ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        _ax.set_xticks([])
        _ax.set_yticks([])
    
def custom_box_style(x0, y0, width, height, mutation_size, mutation_aspect=1):
    """
    Given the location and size of the box, return the path of
    the box around it.
    
     - *x0*, *y0*, *width*, *height* : location and size of the box
     - *mutation_size* : a reference scale for the mutation.
     - *aspect_ratio* : aspect-ration for the mutation.
    """

    # note that we are ignoring mutation_aspect. This is okay in general.

    # padding
    mypad = 0.3
    pad = mutation_size * mypad

    # width and height with padding added.
    width, height = width + 2.*pad, \
                    height + 2.*pad,

    # boundary of the padded box
    x0, y0 = x0-pad, y0-pad,
    x1, y1 = x0+width, y0 + height

    cp = [(x0, y0),
          (x1, y0), (x1, y1), (x0, y1),
          (x0-pad, (y0+y1)/2.), (x0, y0),
          (x0, y0)]

    com = [Path.MOVETO,
           Path.LINETO, Path.LINETO, Path.LINETO,
           Path.LINETO, Path.LINETO,
           Path.CLOSEPOLY]

    path = Path(cp, com)

    return path
    
def get_I_v_mask():
    Ivdict_fn = '/Users/susanclark/Projects/FiberDGR/data/GALFA_HI_Iv_dict_mask_GALFA_footprint_absb_gt_30_lt_90_NHImin_0max_8_HFI_PS_True.npy'
    Ivdict = np.load(Ivdict_fn, encoding='bytes').item()
    all_galfa_vels = np.zeros(len(Ivdict.keys()))
    all_mask_I_v = np.zeros(len(Ivdict.keys()))
    for _i, _key in enumerate(Ivdict.keys()):
        str_key = _key.decode('utf-8')
        
        all_galfa_vels[_i] = gvh.galfa_name_dict[str_key]
        all_mask_I_v[_i] = Ivdict[_key]
        
    sortindx = np.argsort(all_galfa_vels)
    all_galfa_vels_sort = all_galfa_vels[sortindx]
    all_mask_I_v_sort = all_mask_I_v[sortindx]
    
    return all_galfa_vels_sort, all_mask_I_v_sort

def plot_2panel_RGB_FIR_data():
    DR2_Wide_slice_root = "/Users/susanclark/Dropbox/DR2 Full Sky Wide Maps/"

    nhi90map = fits.getdata("/Users/susanclark/Dropbox/NHImaps/GALFA-HI_NHISRCORR_VLSR-90+90kms.fits")
    p857map = fits.getdata("/Users/susanclark/Dropbox/Planck/HFI_SkyMap_857_2048_R3.01_ONGALFAHI.fits")
    submonopole = 0.64
    if submonopole is not None:
        p857map -= submonopole
        
    planck_cmap = ListedColormap(np.loadtxt("/Users/susanclark/Dropbox/ACTPol/code/Planck_Parchment_RGB.txt")/255.)
    planck_cmap.set_bad("gray") # color of missing pixels
    planck_cmap.set_under("white") # color of background, necessary if you want to use with mollview
    clip_planck_cmap = cmocean.tools.crop_by_percent(planck_cmap.reversed(), 50, which='max')

    velkms = gvh.galfa_name_dict["1024"]
    slice_fn = DR2_Wide_slice_root+"GALFA_HI_W_S{}_V{}kms.fits".format(1024, velkms)
    slice_data_1024 = fits.getdata(slice_fn)
    big_hdr = fits.getheader(slice_fn)

    velkms = gvh.galfa_name_dict["1025"]
    slice_fn = DR2_Wide_slice_root+"GALFA_HI_W_S{}_V{}kms.fits".format(1025, velkms)
    slice_data_1025 = fits.getdata(slice_fn)

    velkms = gvh.galfa_name_dict["1023"]
    slice_fn = DR2_Wide_slice_root+"GALFA_HI_W_S{}_V{}kms.fits".format(1023, velkms)
    slice_data_1023 = fits.getdata(slice_fn)
    
    umask_1024 = load_USM_map(vels=["1024"], fwhm=30, zeroed=True, Narrow=False, addnoise=False)
    
    bstart = 30
    NHIcut = True
    NHImin = 0
    NHImax = 8
    maskbs = make_mask_2d(bstart=bstart, bstop=90, PS=True, NHIcut=NHIcut, NHImin=NHImin, NHImax=NHImax)  

    #ystart=1498#1500
    #ystop=2200
    ystart=1000
    ystop=1800
    xstart=7000
    xstop=8500
    #xstart=8500
    #xstop=10500
    
    xstartsmaller=8000
    
    cut_hdr, p857cut = cutouts.xycutout_data(p857map, big_hdr, xstart=xstart, xstop=xstop, ystart=ystart, ystop=ystop)
    cut_hdr, nhi90cut = cutouts.xycutout_data(nhi90map, big_hdr, xstart=xstart, xstop=xstop, ystart=ystart, ystop=ystop)

    cut_hdr, slice_data_1024_cut = cutouts.xycutout_data(slice_data_1024, big_hdr, xstart=xstart, xstop=xstop, ystart=ystart, ystop=ystop)
    cut_hdr, slice_data_1025_cut = cutouts.xycutout_data(slice_data_1025, big_hdr, xstart=xstart, xstop=xstop, ystart=ystart, ystop=ystop)
    cut_hdr, slice_data_1023_cut = cutouts.xycutout_data(slice_data_1023, big_hdr, xstart=xstart, xstop=xstop, ystart=ystart, ystop=ystop)

    cut_hdr_smaller, slice_data_1024_cut_smaller = cutouts.xycutout_data(slice_data_1024, big_hdr, xstart=xstartsmaller, xstop=xstop, ystart=ystart, ystop=ystop)
    cut_hdr_smaller, umask_1024_cut_smaller = cutouts.xycutout_data(umask_1024, big_hdr, xstart=xstartsmaller, xstop=xstop, ystart=ystart, ystop=ystop)

    cut_hdr, maskbs_cut = cutouts.xycutout_data(maskbs, big_hdr, xstart=xstart, xstop=xstop, ystart=ystart, ystop=ystop)

    xax, ra_label = cutouts.get_xlabels_ra(cut_hdr, skip = 200.0, origincoord=0)
    yax, dec_label = cutouts.get_ylabels_dec(cut_hdr, skip = 200.0, origincoord=0)
    
    w = cutouts.make_wcs(cut_hdr)
    
    chunky, chunkx = slice_data_1024_cut.shape
    chunkysmaller, chunkxsmaller = umask_1024_cut_smaller.shape
    
    RGBdata = np.zeros((ystop-ystart, xstop-xstart, 3), np.float_)
    RGBdata[:, :, 0]= slice_data_1023[ystart:ystop, xstart:xstop]
    RGBdata[:, :, 1]= slice_data_1024[ystart:ystop, xstart:xstop]
    RGBdata[:, :, 2]= slice_data_1025[ystart:ystop, xstart:xstop]

    RGBdata[np.where(RGBdata <= 0)] = 0
    RGBdata[np.where(np.isnan(RGBdata) == True)] = 0
    
    #for _i in np.arange(3):
    #    RGBdata[:, :, _i] = 

    RGBdata = RGBdata/np.max(RGBdata)
    
    H, S, V = RGB_to_HSV_arr(RGBdata[:, :, 0], RGBdata[:, :, 1], RGBdata[:, :, 2])
    logscaleV = np.log10(V)
    logscaleV = (logscaleV - np.min(logscaleV))
    logscaleV = logscaleV/np.max(logscaleV)

    logscaleS = np.log10(S + 0.001)
    logscaleS = (logscaleS - np.nanmin(logscaleS))
    logscaleS = logscaleS/np.nanmax(logscaleS)

    R,G,B = RGB_to_HSV_arr(H, S, logscaleV, reverse=True)

    RGBscaled = np.zeros(RGBdata.shape)
    RGBscaled[:, :, 0] = R
    RGBscaled[:, :, 1] = G
    RGBscaled[:, :, 2] = B
    RGBscaled = RGBscaled/np.nanmax(RGBscaled)
    
    fig = plt.figure(figsize=(12,9))
    axbackground = fig.add_subplot(111)
    axbackground.axis('off')

    left, bottom, width, height = [0.05, 0.52, 0.8, 0.45]
    ax1 = fig.add_axes([left, bottom, width, height])
    left, bottom, width, height = [0.05, 0.07, 0.8, 0.45]
    ax2 = fig.add_axes([left, bottom, width, height])
    left, bottom, width, height = [0.78, 0.52, 0.15, 0.15]
    ax3 = fig.add_axes([left, bottom, width, height])
    #left, bottom, width, height = [0.8, 0.07, 0.03, 0.2]
    left, bottom, width, height = [0.78, 0.07, 0.15, 0.03]
    ax4 = fig.add_axes([left, bottom, width, height])

    labelras = np.array([220., 225., 230., 235., 240.])
    labeldecs = np.array([15., 21., 27.])
    
    xax, ras, yax, decs = cutouts.get_xylabels_from_ras_decs(labelras, labeldecs, cut_hdr, origincoord=0)
    
    #im1 = ax1.imshow(RGBscaled)
    im1 = ax1.imshow(RGBdata)
    #ax1 = cutouts.set_all_labels(ax1, xax, ra_label, yax, dec_label, roundnum=1)
    ax1 = cutouts.set_all_labels(ax1, xax, labelras, yax, labeldecs, roundnum=0, size=12, degrees=True)
    #plt.colorbar(im1)
    
    #for _b in [50, 60]:
    #    xs_blines, ys_blines = cutouts.add_b_lines(_b, w)
    #    ax1.plot(xs_blines, ys_blines, color="gray", alpha=0.2, lw=2)
    #    ax2.plot(xs_blines, ys_blines, color="gray", alpha=0.2, lw=2)
        
    for ax in [ax1, ax2]:
        ax.set_ylim(0, chunky)
        ax.set_xlim(0, chunkx)
        
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
    
    p857vmin = np.percentile(p857cut[np.where(maskbs_cut > 0.5)], 0.5)
    p857vmax = np.percentile(p857cut[np.where(maskbs_cut > 0.5)], 99.5)
    #im2 = ax2.imshow(p857cut, cmap="Greys_r", norm=LogNorm(vmin=np.min(p857cut), vmax=np.max(p857cut) ))
    im2 = ax2.imshow(p857cut, cmap=cc.m_linear_kryw_5_100_c67, vmin=p857vmin, vmax=p857vmax)
    #im2 = ax2.imshow(p857cut, cmap=clip_planck_cmap, vmin=p857vmin, vmax=p857vmax)
    #im2 = ax2.imshow(p857cut, cmap=cmocean.cm.gray, vmin=p857vmin, vmax=p857vmax)
    #ax2 = cutouts.set_all_labels(ax2, xax, ra_label, yax, dec_label, roundnum=1)
    cutouts.set_all_labels(ax2, xax, labelras, yax, labeldecs, roundnum=0, size=12, degrees=True)

    cbar = plt.colorbar(im2, cax=ax4, orientation='horizontal')
    
    cbar.outline.set_visible(False)
    cbar.ax.set_title(r"$\mathrm{MJy/sr}$", size=15)
    cbar.ax.set_xticks([1, 2, 3])
    cbar.ax.set_xticklabels([r"$1$", r"$2$", r"$3$"], size=12)

    ax1.set_xticks([])
    
    all_galfa_vels_sort, all_mask_I_v_sort = get_I_v_mask()
    
    v1023arg = np.where(all_galfa_vels_sort == -0.4)[0][0]
    v1024arg = np.where(all_galfa_vels_sort == 0.4)[0][0]
    v1025arg = np.where(all_galfa_vels_sort == 1.1)[0][0]
    
    #ax3.plot(all_galfa_vels_sort, all_mask_I_v_sort, color='black')
    alphaslice=0.5
    #ax3.vlines(all_galfa_vels_sort[v1023arg], ymin=0, ymax=all_mask_I_v_sort[v1023arg], color='red', alpha=alphaslice)
    #ax3.vlines(all_galfa_vels_sort[v1024arg], ymin=0, ymax=all_mask_I_v_sort[v1024arg], color='green', alpha=alphaslice)
    #ax3.vlines(all_galfa_vels_sort[v1025arg], ymin=0, ymax=all_mask_I_v_sort[v1025arg], color='blue', alpha=alphaslice)
    
    halfvel = 0.72/2
    start_v1023 = all_galfa_vels_sort[v1023arg] - halfvel
    stop_v1023 = all_galfa_vels_sort[v1023arg] + halfvel
    start_v1024 = all_galfa_vels_sort[v1024arg] - halfvel
    stop_v1024 = all_galfa_vels_sort[v1024arg] + halfvel
    start_v1025 = all_galfa_vels_sort[v1025arg] - halfvel
    stop_v1025 = all_galfa_vels_sort[v1025arg] + halfvel
        
    #ax3.fill_between(all_galfa_vels_sort, np.zeros(len(all_galfa_vels_sort)), all_mask_I_v_sort, where=( (all_galfa_vels_sort >= start_v1023) & (all_galfa_vels_sort <= stop_v1023) ), facecolor='red', alpha=alphaslice, interpolate=True)
    
    ax3.fill_between(all_galfa_vels_sort, all_mask_I_v_sort, facecolor='gray', alpha=0.3, interpolate=True)
    ax3.fill_between(all_galfa_vels_sort, all_mask_I_v_sort, where=(all_galfa_vels_sort >= start_v1023) , facecolor='white', alpha=1, interpolate=True)

    #ax3.fill_between(all_galfa_vels_sort, all_mask_I_v_sort, where=(all_galfa_vels_sort > start_v1023) & (all_galfa_vels_sort <= stop_v1023), facecolor='red', alpha=alphaslice, interpolate=True)
    ax3.fill_between(all_galfa_vels_sort, all_mask_I_v_sort, where=(all_galfa_vels_sort >= start_v1023) , facecolor='red', alpha=alphaslice, interpolate=True)
    ax3.fill_between(all_galfa_vels_sort, all_mask_I_v_sort, where=(all_galfa_vels_sort > stop_v1023) , facecolor='white', alpha=1, interpolate=True)
    
    ax3.fill_between(all_galfa_vels_sort, all_mask_I_v_sort, where=(all_galfa_vels_sort > start_v1024) , facecolor='green', alpha=alphaslice, interpolate=True)
    ax3.fill_between(all_galfa_vels_sort, all_mask_I_v_sort, where=(all_galfa_vels_sort > stop_v1024) , facecolor='white', alpha=1, interpolate=True)

    ax3.fill_between(all_galfa_vels_sort, all_mask_I_v_sort, where=(all_galfa_vels_sort >= start_v1025) , facecolor='blue', alpha=alphaslice, interpolate=True)
    ax3.fill_between(all_galfa_vels_sort, all_mask_I_v_sort, where=(all_galfa_vels_sort > stop_v1025) , facecolor='white', alpha=1, interpolate=True)
    
    ax3.fill_between(all_galfa_vels_sort, all_mask_I_v_sort, where=(all_galfa_vels_sort >= stop_v1025) , facecolor='gray', alpha=0.3, interpolate=True)
    
    #ax3.fill_betweenx(all_mask_I_v_sort, 0, 0, where=(all_galfa_vels_sort > start_v1023) & (all_galfa_vels_sort <= stop_v1023), interpolate=True)
    
    #ax3.fill_betweenx(all_galfa_vels_sort, x1=-1, x2=4, interpolate=True)
    #ax3.fill_betweenx(all_mask_I_v_sort, all_galfa_vels_sort, x2=start_v1023, where=all_galfa_vels_sort <= stop_v1023, facecolor='red', alpha=alphaslice, interpolate=True)
    ax3.set_xlim(-20, 20)
    
    ax1.set_ylabel(r"$\mathrm{DEC}$", size=20)
    ax2.set_ylabel(r"$\mathrm{DEC}$", size=20)
    ax2.set_xlabel(r"$\mathrm{RA}$", size=20)
    
    for _ax in [ax3, ax4]:
        _ax.spines["top"].set_visible(False)
        _ax.spines["bottom"].set_visible(False)
        _ax.spines["left"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        
    ax3.set_yticks([])
    ax3.set_xlabel(r"$v_{lsr}$ $[\mathrm{km}/s]$", size=15)
    ax3.set_xticks([-20, 20])
    ax3.tick_params(axis='both', which='both',length=0)
    ax3.set_xticklabels([r"$-20$", r"$+20$"], size=12)
    ax3.xaxis.set_tick_params(pad=-2)
    
def plot_USM_thin_thick():

    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    
    USMroot = "/Users/susanclark/Dropbox/DR2 Full Sky Wide Maps/"
    USMfn1 = "GALFA_HI_W_vels1023_to_1025_USM30_zeroed_True.fits" # 2.16 km/s
    #USMfn2 = "GALFA_HI_W_vels1007_to_1041_USM30_zeroed_True.fits" # 25.2 km/s
    USMfn2 = "GALFA_HI_W_vels1011_to_1037_USM30_zeroed_True.fits" # 19.44 km/s
    USMfn3 = "GALFA_HI_W_vels0994_to_1054_USM30_zeroed_True.fits" # 43.92 km/s
    
    all_axs = [ax1, ax2, ax3]
    all_fns = [USMfn1, USMfn2, USMfn3]
    all_widths = [2.16, 19.44, 43.92]
    
    starty=400
    stopy=2000#1800
    startx= 8000#9800
    stopx = 9800#11400
    
    for _i, (ax, fn, width) in enumerate(zip(all_axs, all_fns, all_widths)):
        USMdata = fits.getdata(USMroot+fn)
        USMchunk = USMdata[starty:stopy, startx:stopx]
        ax.imshow(USMchunk)
        ax.set_title("{} km/s".format(width))
    

def makestring(v):
    if v > 999:
        return str(v)
    else:
        return '0'+str(v)

def make_USM_maps(addnoise=False, noisefactor=1, plainchannelmap=False):
    fwhm = 30
    zeroed = True
    
    if addnoise:
        noisestr = "_addnoise"
        if noisefactor > 1:
            noisestr = "_addnoise_x{}".format(noisefactor)
    else:
        noisestr = ""
    
    NWstr = "N"
    DR2_slice_root = "/Users/susanclark/Dropbox/DR2_Full_Sky_Narrow_Maps/"
    narrowstr = "_Narrow"
    if plainchannelmap:
        outfn = DR2_slice_root+"GALFA_HI_W_vels{}_to_{}{}{}.fits".format("1024", "1024", narrowstr, noisestr)
    else:
        outfn = DR2_slice_root+"GALFA_HI_W_vels{}_to_{}{}_USM{}_zeroed_{}{}.fits".format("1024", "1024", narrowstr, fwhm, zeroed, noisestr)
    if os.path.isfile(outfn):
        print("Done")
    else:
        narrow_slice_data = get_USM_slice(vels=["1024"], fwhm=fwhm, zeroed=True, Narrow=True, reverse=False, writemap=True, returnrawslice=False, addnoise=addnoise, noisefactor=noisefactor)
    
    NWstr = "W"
    narrowstr = ""
    DR2_slice_root = "/Users/susanclark/Dropbox/DR2 Full Sky Wide Maps/"
    centervel = 990#1024
    for _i in np.arange(31):
        startvel = centervel - _i
        stopvel = centervel + _i
        vels = [makestring(v) for v in np.arange(startvel, stopvel+1)]
        print("vel 0 = {}".format(vels[0]))
        
        if plainchannelmap:
            outfn = DR2_slice_root+"GALFA_HI_W_vels{}_to_{}{}{}.fits".format(vels[0], vels[-1], narrowstr, noisestr)
        else:
            outfn = DR2_slice_root+"GALFA_HI_W_vels{}_to_{}{}_USM{}_zeroed_{}{}.fits".format(vels[0], vels[-1], narrowstr, fwhm, zeroed, noisestr)
        if os.path.isfile(outfn):
            print("Done with vels {}".format(vels))
        else:
            slice_data = get_USM_slice(vels=vels, fwhm=fwhm, zeroed=True, Narrow=False, reverse=False, writemap=True, returnrawslice=False, addnoise=addnoise, noisefactor=noisefactor)
    
def weighted_mean(NHImap, P857map, weightsmap=None, mask=None, nhi_max_fit=8, nhi_min_fit=0):
    
    NHImap_div1E20 = NHImap/1.E20

    # could make this 25th and 75th weighted percentiles or something
    #nhi_max_fit = 8.
    #nhi_min_fit = 2.

    if mask is not None:
        print("Including mask")
        cutwhere = np.where((P857map > 0) & (NHImap > 0) & (NHImap_div1E20 < nhi_max_fit) & (NHImap_div1E20 > nhi_min_fit) & (mask > 0))
        cutnonzero = np.where((P857map > 0) & (NHImap > 0) & (np.isnan(P857map) == False) & (np.isnan(NHImap) == False) & (mask > 0) )
        maskcutwhere = np.zeros((2432, 21600), np.int_)
        maskcutnonzero = np.zeros((2432, 21600), np.int_)
        maskcutwhere[cutwhere] = 1
        maskcutnonzero[cutnonzero] = 1

    else:
        cutwhere = np.where((P857map > 0) & (NHImap > 0) &(NHImap_div1E20 < nhi_max_fit) & (NHImap_div1E20 > nhi_min_fit))
        cutnonzero = np.where((P857map > 0) & (NHImap > 0) & (np.isnan(P857map) == False) & (np.isnan(NHImap) == False))
        
    NHImap_div1E20_cut = NHImap_div1E20[cutwhere]
    P857map_cut = P857map[cutwhere]

    meancutwhereNHI = np.mean(NHImap_div1E20_cut)
    meancutwhereP857 = np.mean(P857map_cut)
    print('mean NHI cutwhere = {}'.format(meancutwhereNHI))
    print('mean 857 cutwhere = {}'.format(meancutwhereP857))
    meancutnonzeroNHI = np.average(NHImap_div1E20[cutnonzero])
    meancutnonzeroP857 = np.average(P857map[cutnonzero])
    print('mean NHI cutnonzero = {}'.format(meancutnonzeroNHI))
    print('mean 857 cutnonzero = {}'.format(meancutnonzeroP857))
    if weightsmap is not None:
        weightedmeancutwhereNHI = np.average(NHImap_div1E20_cut, weights=weightsmap[cutwhere])
        weightedmeancutwhereP857 = np.average(P857map_cut, weights=weightsmap[cutwhere])
        print('weighted mean NHI cutwhere = {}'.format(weightedmeancutwhereNHI))
        print('weighted mean 857 cutwhere = {}'.format(weightedmeancutwhereP857))
        weightedmeancutnonzeroNHI = np.average(NHImap_div1E20[cutnonzero], weights=weightsmap[cutnonzero])
        weightedmeancutnonzeroP857 = np.average(P857map[cutnonzero], weights=weightsmap[cutnonzero])
        print('weighted mean NHI cutnonzero = {}'.format(weightedmeancutnonzeroNHI))
        print('weighted mean 857 cutnonzero = {}'.format(weightedmeancutnonzeroP857))
        
        print('delta cutnonzero FIR = {}'.format(weightedmeancutnonzeroP857 - meancutnonzeroP857))
        print('delta cutnonzero NHI = {}'.format(weightedmeancutnonzeroNHI - meancutnonzeroNHI))
        print('delta FIR/delta NHI = {}'.format((weightedmeancutnonzeroP857 - meancutnonzeroP857)/(weightedmeancutnonzeroNHI - meancutnonzeroNHI)))
        
    print("Len cutnonzero {}, cutwhere {}".format(len(np.nonzero(cutnonzero)[0]), len(np.nonzero(cutwhere)[0])))
    
    if weightsmap is not None:
        return meancutnonzeroP857, weightedmeancutnonzeroP857, meancutnonzeroNHI, weightedmeancutnonzeroNHI, meancutwhereP857, weightedmeancutwhereP857, meancutwhereNHI, weightedmeancutwhereNHI
    else:
        return meancutnonzeroP857, meancutnonzeroNHI, meancutwhereP857, meancutwhereNHI

def weighted_mean_flatarr(flatNHI, flat857, flatweights):
    
    wherenonzero = np.where( (flatNHI > 0) & (flat857 > 0) & (np.isnan(flatNHI) == False) & (np.isnan(flat857) == False) & (np.isnan(flatweights) == False))
    
    unweighted_NHI_mean = np.nanmean(flatNHI[wherenonzero])
    unweighted_857_mean = np.nanmean(flat857[wherenonzero])
    
    weighted_NHI_mean = np.average(flatNHI[wherenonzero], weights=flatweights[wherenonzero])
    weighted_857_mean = np.average(flat857[wherenonzero], weights=flatweights[wherenonzero])
    
    return unweighted_NHI_mean, unweighted_857_mean, weighted_NHI_mean, weighted_857_mean
    
def weighted_mean_arr(NHIarr, P857arr, weightsarr):
    
    unweighted_NHI_mean = np.nanmean(NHIarr)
    unweighted_857_mean = np.nanmean(P857arr)
    
    weighted_NHI_mean = np.average(NHIarr, weights=weightsarr)
    weighted_857_mean = np.average(P857arr, weights=weightsarr)
    
    return unweighted_NHI_mean, unweighted_857_mean, weighted_NHI_mean, weighted_857_mean
    
def plot_hists():
    NHImap = fits.getdata("/Users/susanclark/Dropbox/NHImaps/GALFA-HI_NHISRCORR_VLSR-90+90kms.fits")
    P857map = fits.getdata("/Users/susanclark/Dropbox/Planck/HFI_SkyMap_857_2048_R3.01_ONGALFAHI.fits")
    
    submonopole = 0.64
    if submonopole is not None:
        P857map -= submonopole
        submonopolestr = "_submono{}".format(submonopole)
    else:
        submonopolestr = ""
    
    bstart = 30
    NHIcut = True
    NHImin = 0
    NHImax = 8
    
    maskbs = make_mask_2d(bstart=bstart, bstop=90, PS=True, NHIcut=NHIcut, NHImin=NHImin, NHImax=NHImax)  
    
    NHImap_div1E20 = NHImap/1.E20

    print("Including mask in cuts")
    cutwhere = np.where((P857map > 0) & (NHImap > 0) & (NHImap_div1E20 < NHImax) & (NHImap_div1E20 > NHImin) & (maskbs > 0))
    cutnonzero = np.where((P857map > 0) & (NHImap > 0) & (np.isnan(P857map) == False) & (np.isnan(NHImap) == False) & (maskbs > 0))
                
    vels=["1023", "1024", "1025"]
    #vels0=["1024"]
    
    usewhere = True
    
    # unsharp masked data
    vels_umask = get_USM_slice(vels=vels, fwhm=30, zeroed=True, Narrow=False, reverse=False, writemap=False)
    
    inv_vels_umask = get_USM_slice(vels=vels, fwhm=30, zeroed=False, Narrow=False, reverse=False, writemap=False)
    inv_vels_umask *= -1
    inv_vels_umask[inv_vels_umask < 0] = 0
    
    # histogram by umask data
    topbin = np.percentile(vels_umask, 99)
    cuthist = np.where((P857map > 0) & (NHImap > 0) & (NHImap_div1E20 < NHImax) & (NHImap_div1E20 > NHImin) & (maskbs > 0) & (vels_umask > 0) )
    umask_hist, umask_bin_edges = np.histogram(vels_umask[cuthist], range=(0, topbin))
    
    # histogram by inverse umask data
    topinvbin = np.percentile(inv_vels_umask, 99)
    cutinvhist = np.where((P857map > 0) & (NHImap > 0) & (NHImap_div1E20 < NHImax) & (NHImap_div1E20 > NHImin) & (maskbs > 0) & (inv_vels_umask > 0) )
    inv_umask_hist, inv_umask_bin_edges = np.histogram(inv_vels_umask[cutinvhist], range=(0, topinvbin))
    
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    
    for _i in np.arange(len(umask_hist)):
        cutdata = np.where( (vels_umask > umask_bin_edges[_i]) & (vels_umask < umask_bin_edges[_i+1]) & (P857map > 0) & (NHImap > 0) & (NHImap_div1E20 < NHImax) & (NHImap_div1E20 > NHImin) & (maskbs > 0) & (vels_umask > 0) )
        ax1.hist(P857map[cutdata]/NHImap_div1E20[cutdata], range=(0, 2), histtype="step", bins=100, normed=True, label="USM {} to {}".format(np.round(umask_bin_edges[_i], 2), np.round(umask_bin_edges[_i+1], 2)));
    
    ax1.legend()
    ax1.set_title('P857/NHI by USM value')
    
    ax2 = fig.add_subplot(122)
    
    for _i in np.arange(len(inv_umask_hist)):
        cutdata = np.where( (inv_vels_umask > inv_umask_bin_edges[_i]) & (inv_vels_umask < inv_umask_bin_edges[_i+1]) & (P857map > 0) & (NHImap > 0) & (NHImap_div1E20 < NHImax) & (NHImap_div1E20 > NHImin) & (maskbs > 0) & (inv_vels_umask > 0) )
        ax2.hist(P857map[cutdata]/NHImap_div1E20[cutdata], range=(0, 2), histtype="step", bins=100, normed=True, label="inv USM {} to {}".format(np.round(inv_umask_bin_edges[_i]), np.round(inv_umask_bin_edges[_i+1], 2) ));
    
    ax2.legend()
    ax2.set_title('P857/NHI by inverse USM value')

def plot_DGR_hist_by_gradient():
    NHImap = fits.getdata("/Users/susanclark/Dropbox/NHImaps/GALFA-HI_NHISRCORR_VLSR-90+90kms.fits")
    P857map = fits.getdata("/Users/susanclark/Dropbox/Planck/HFI_SkyMap_857_2048_R3.01_ONGALFAHI.fits")
    
    submonopole = 0.64
    if submonopole is not None:
        P857map -= submonopole
        submonopolestr = "_submono{}".format(submonopole)
    else:
        submonopolestr = ""
    
    bstart = 30
    NHIcut = True
    NHImin = 0
    NHImax = 8
    
    maskbs = make_mask_2d(bstart=bstart, bstop=90, PS=True, NHIcut=NHIcut, NHImin=NHImin, NHImax=NHImax)  
    
    NHImap_div1E20 = NHImap/1.E20

    print("Including mask in cuts")
    cutwhere = np.where((P857map > 0) & (NHImap > 0) & (NHImap_div1E20 < NHImax) & (NHImap_div1E20 > NHImin) & (np.isnan(P857map) == False) & (np.isnan(NHImap) == False) & (maskbs > 0))
    cutnonzero = np.where((P857map > 0) & (NHImap > 0) & (np.isnan(P857map) == False) & (np.isnan(NHImap) == False) & (maskbs > 0))
                
    vels=["1023", "1024", "1025"]
    #vels0=["1024"]
    
    usewhere = True
    
    # unsharp masked data
    vels_umask, vels_raw = get_USM_slice(vels=vels, fwhm=30, zeroed=True, Narrow=False, reverse=False, writemap=False, returnrawslice=True)
    gradient_vels = np.gradient(vels_raw)
    gradient_mag = np.sqrt(gradient_vels[0]**2 + gradient_vels[1]**2) # replace USM with gradient magnitude
        
    # histogram by umask data
    topbin = np.nanpercentile(gradient_mag, 99)
    cuthist = np.where((P857map > 0) & (NHImap > 0) & (NHImap_div1E20 < NHImax) & (NHImap_div1E20 > NHImin) & (maskbs > 0) & (np.isnan(gradient_mag)==False) & (np.isnan(NHImap_div1E20)==False) )
    umask_hist, umask_bin_edges = np.histogram(gradient_mag[cuthist], range=(0, topbin), bins=10)
    
    fig = plt.figure(figsize=(16,8))
    axbackground = fig.add_subplot(111)
    axbackground.axis('off')
    
    left, bottom, width, height = [0.05, 0.1, 0.6, 0.9]
    ax1 = fig.add_axes([left, bottom, width, height])
    
    cmap = cmocean.cm.matter
    
    # gray background hist of all data
    ax1.hist(P857map[cutwhere]/NHImap_div1E20[cutwhere], range=(0, 2), bins=100, density=True, edgecolor='None', fc=(0.5, 0.5, 0.5, 0.5), alpha=0.5, histtype = 'step', fill=True, lw=0)
    
    for _i in np.arange(len(umask_hist)):
        cutdata = np.where( (gradient_mag > umask_bin_edges[_i]) & (gradient_mag < umask_bin_edges[_i+1]) & (P857map > 0) & (NHImap > 0) & (NHImap_div1E20 < NHImax) & (NHImap_div1E20 > NHImin) & (maskbs > 0) & (np.isnan(gradient_mag) == False) )
        ax1.hist(P857map[cutdata]/NHImap_div1E20[cutdata], range=(0, 2), histtype="step", bins=100, density=True, lw=1.5, color=cmap(_i/len(umask_hist)), label="USM {} to {}".format(np.round(umask_bin_edges[_i], 2), np.round(umask_bin_edges[_i+1], 2)));
    
    #ax1.legend()
    #ax1.set_title('P857/NHI by USM value')
    
    ax1.set_xlim(0, 1.5)
    
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    
    ax1.spines["left"].set_bounds(0, 5.5)
    ax1.spines["bottom"].set_bounds(0, 1.5)
    
    ax1.tick_params(labelsize=15)
    ax1.set_xlabel(r"$I_{857}/\mathrm{N}_\mathrm{\sc HI}$", size=18)
    
    #ax1.axis('off')
    #[400:1800, 9800:11400]
    
    #plt.savefig('../figures/DGR_hist_matter_notitle_nohist_bins12.eps')
    
            
    DR2_Wide_slice_root = "/Users/susanclark/Dropbox/DR2 Full Sky Wide Maps/"
    vel0kms = gvh.galfa_name_dict["1024"]
    slice_fn = DR2_Wide_slice_root+"GALFA_HI_W_S{}_V{}kms.fits".format("1024", vel0kms)
    
    allsky_hdr = fits.getheader(slice_fn)
    allsky_wcs = cutouts.make_wcs(allsky_hdr)

    starty=400
    stopy=1800
    startx= 9800
    stopx = 11400

    # make properly formatted chunks
    xycut_hdr, xycut_vels_umask = cutouts.xycutout_data(gradient_mag, allsky_hdr, xstart=startx, xstop=stopx, ystart=starty, ystop=stopy)
    xycut_hdr, xycut_vels_raw = cutouts.xycutout_data(vels_raw, allsky_hdr, xstart=startx, xstop=stopx, ystart=starty, ystop=stopy)
    
    xax, ra_label = cutouts.get_xlabels_ra(xycut_hdr, skip = 400.0)
    yax, dec_label = cutouts.get_ylabels_dec(xycut_hdr, skip = 400.0)

    xax, ra_label, yax, dec_label = cutouts.get_xylabels_from_ras_decs([173, 183, 193], [6, 16, 26], xycut_hdr, origincoord=0)
    
    nycut, nxcut = xycut_vels_umask.shape
    
    #fig = plt.figure()
    #ax1 = fig.add_subplot(111)
    left, bottom, width, height = [0.5, 0.17, 0.5, 0.77]
    ax0 = fig.add_axes([left, bottom, width, height])
    #ax0.axis('off')
    
    ax0.set_xlabel(r"$\mathrm{RA}$", size=15)
    ax0.set_ylabel(r"$\mathrm{DEC}$", size=15)

    im1 = ax0.contourf(xycut_vels_umask, cmap = cmocean.cm.matter, levels=umask_bin_edges, extend="max")
    ax0 = cutouts.set_all_labels(ax0, xax, ra_label, yax, dec_label, roundnum=1, degrees=True)
    ax0.tick_params(labelsize=15)
    cbar = plt.colorbar(im1, ticks=umask_bin_edges, ax=ax0)
    cbar.ax.tick_params(labelsize=12) 
    #cbar.ax.set_yticks(umask_bin_edges)
    cbar.ax.set_yticklabels(np.round(umask_bin_edges, 1))
    cbar.outline.set_visible(False)
    #cbar.set_label(r"$w^{\delta v = 2.2 \mathrm{km/s}}$", size=18, rotation=270)
    cbar.ax.set_title(r"$\mathrm{gradient}$ $\mathrm{magnitude}$", size=18)       
    
    #plt.savefig('../figures/DGR_hist_contourplot_gradient_magnitude_vels1023_1025_matter_bins10.pdf')     
                
def plot_DGR_hist_by_USM():
    NHImap = fits.getdata("/Users/susanclark/Dropbox/NHImaps/GALFA-HI_NHISRCORR_VLSR-90+90kms.fits")
    P857map = fits.getdata("/Users/susanclark/Dropbox/Planck/HFI_SkyMap_857_2048_R3.01_ONGALFAHI.fits")
    
    #I353map = fits.getdata("/Users/susanclark/Dropbox/Planck/HFI_SkyMap_353_2048_R2.02_full_T_projected_GALFAallsky.fits")
    #Q353map = fits.getdata("/Users/susanclark/Dropbox/Planck/HFI_SkyMap_353_2048_R2.02_full_Q_projected_GALFAallsky.fits")
    #U353map = fits.getdata("/Users/susanclark/Dropbox/Planck/HFI_SkyMap_353_2048_R2.02_full_U_projected_GALFAallsky.fits")
    #P857map = np.sqrt(Q353map**2 + U353map**2)/I353map
    
    submonopole = 0.64
    if submonopole is not None:
        P857map -= submonopole
        submonopolestr = "_submono{}".format(submonopole)
    else:
        submonopolestr = ""
    
    bstart = 30
    NHIcut = True
    NHImin = 0
    NHImax = 8
    
    maskbs = make_mask_2d(bstart=bstart, bstop=90, PS=True, NHIcut=NHIcut, NHImin=NHImin, NHImax=NHImax)  
    
    #HACK
    #starty=400
    #stopy=1800
    #startx= 9800
    #stopx = 11400
    starty=1000
    stopy=1800
    startx=7800
    stopx=8600
    #maskbs[:starty, :] = 0
    #maskbs[stopy:, :] = 0
    #maskbs[:, stopx:] = 0
    #maskbs[:, :startx] = 0
    
    NHImap_div1E20 = NHImap/1.E20

    print("Including mask in cuts")
    cutwhere = np.where((P857map > 0) & (NHImap > 0) & (NHImap_div1E20 < NHImax) & (NHImap_div1E20 > NHImin) & (np.isnan(P857map) == False) & (np.isnan(NHImap) == False) & (maskbs > 0))
    cutnonzero = np.where((P857map > 0) & (NHImap > 0) & (np.isnan(P857map) == False) & (np.isnan(NHImap) == False) & (maskbs > 0))
                
    vels=["1023", "1024", "1025"]
    #vels=["1010", "1011", "1012"]
    #vels0=["1024"]
    
    usewhere = True
    
    # unsharp masked data
    vels_umask, vels_raw = get_USM_slice(vels=vels, fwhm=30, zeroed=True, Narrow=False, reverse=False, writemap=False, returnrawslice=True)
    vels_umask = vels_umask/np.nanmax(vels_umask[maskbs > 0])
    #vels_raw[np.where(np.isnan(vels_raw) == True)] = 0 #HACK
    #vels_raw[np.where(vels_raw < 0)] = 0 #HACK
    #vels_umask = vels_raw/np.nanmax(vels_raw[maskbs > 0]) #HACK
    
        
    # histogram by umask data
    topbin = np.percentile(vels_umask, 99)
    cuthist = np.where((P857map > 0) & (NHImap > 0) & (NHImap_div1E20 < NHImax) & (NHImap_div1E20 > NHImin) & (maskbs > 0) & (vels_umask > 0) )
    umask_hist, umask_bin_edges = np.histogram(vels_umask[cuthist], range=(0, topbin), bins=10)
    
    fig = plt.figure(figsize=(16,8))
    axbackground = fig.add_subplot(111)
    axbackground.axis('off')
    
    left, bottom, width, height = [0.05, 0.1, 0.6, 0.9]
    ax1 = fig.add_axes([left, bottom, width, height])
    
    cmap = cmocean.cm.matter
    
    # gray background hist of all data
    ax1.hist(P857map[cutwhere]/NHImap_div1E20[cutwhere], range=(0, 2), bins=100, density=True, edgecolor='None', fc=(0.5, 0.5, 0.5, 0.5), alpha=0.5, histtype = 'step', fill=True, lw=0)
    
    for _i in np.arange(len(umask_hist)):
        cutdata = np.where( (vels_umask > umask_bin_edges[_i]) & (vels_umask < umask_bin_edges[_i+1]) & (P857map > 0) & (NHImap > 0) & (NHImap_div1E20 < NHImax) & (NHImap_div1E20 > NHImin) & (maskbs > 0) & (vels_umask > 0) )
        ax1.hist(P857map[cutdata]/NHImap_div1E20[cutdata], range=(0, 2), histtype="step", bins=100, density=True, lw=2, color=cmap(_i/len(umask_hist)), label="USM {} to {}".format(np.round(umask_bin_edges[_i], 2), np.round(umask_bin_edges[_i+1], 2)));
    
    #ax1.legend()
    #ax1.set_title('P857/NHI by USM value')
    
    ax1.set_xlim(0, 1.5)
    #ax1.set_xlim(0, .2)
    
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    
    ax1.spines["left"].set_bounds(0, 5.5)
    ax1.spines["bottom"].set_bounds(0, 1.5)
    
    ax1.tick_params(labelsize=18)
    ax1.set_xlabel(r"$I_{857}/\mathrm{N}_\mathrm{\sc HI}$ $[\mathrm{MJy} \, \mathrm{sr}^{-1} / \mathrm{cm}^{-2}]$", size=24, labelpad=10)
    
    #ax1.axis('off')
    #[400:1800, 9800:11400]
    
    #plt.savefig('../figures/DGR_hist_matter_notitle_nohist_bins12.eps')
    
            
    DR2_Wide_slice_root = "/Users/susanclark/Dropbox/DR2 Full Sky Wide Maps/"
    vel0kms = gvh.galfa_name_dict["1024"]
    slice_fn = DR2_Wide_slice_root+"GALFA_HI_W_S{}_V{}kms.fits".format("1024", vel0kms)
    
    allsky_hdr = fits.getheader(slice_fn)
    allsky_wcs = cutouts.make_wcs(allsky_hdr)

    starty=400
    stopy=1800
    startx= 9800
    stopx = 11400

    # make properly formatted chunks
    xycut_hdr, xycut_vels_umask = cutouts.xycutout_data(vels_umask, allsky_hdr, xstart=startx, xstop=stopx, ystart=starty, ystop=stopy)
    xycut_hdr, xycut_vels_raw = cutouts.xycutout_data(vels_raw, allsky_hdr, xstart=startx, xstop=stopx, ystart=starty, ystop=stopy)
    
    xax, ra_label = cutouts.get_xlabels_ra(xycut_hdr, skip = 400.0)
    yax, dec_label = cutouts.get_ylabels_dec(xycut_hdr, skip = 400.0)

    xax, ra_label, yax, dec_label = cutouts.get_xylabels_from_ras_decs([173, 183, 193], [6, 16, 26], xycut_hdr, origincoord=0)
    
    nycut, nxcut = xycut_vels_umask.shape
    
    #fig = plt.figure()
    #ax1 = fig.add_subplot(111)
    left, bottom, width, height = [0.5, 0.17, 0.5, 0.77]
    ax0 = fig.add_axes([left, bottom, width, height])
    #ax0.axis('off')
    
    textsize = 18
    
    ax0.set_xlabel(r"$\mathrm{RA}$", size=textsize)
    ax0.set_ylabel(r"$\mathrm{DEC}$", size=textsize)

    im1 = ax0.contourf(xycut_vels_umask, cmap = cmocean.cm.matter, levels=umask_bin_edges, extend="max")
    ax0 = cutouts.set_all_labels(ax0, xax, ra_label, yax, dec_label, roundnum=1, degrees=True)
    ax0.tick_params(labelsize=18)
    cbar = plt.colorbar(im1, ticks=umask_bin_edges, ax=ax0)
    cbar.ax.tick_params(labelsize=15) 
    #cbar.ax.set_yticks(umask_bin_edges)
    cbar.ax.set_yticklabels(np.round(umask_bin_edges, 2))
    cbar.outline.set_visible(False)
    #cbar.set_label(r"$w^{\delta v = 2.2 \mathrm{km/s}}$", size=18, rotation=270)
    cbar.ax.set_title(r"$w^{\delta v}$", size=20)
    
    #plt.subplots_adjust()
    #plt.savefig('../figures/DGR_hist_contourplot_umask_vels1023_1025_matter_bins10_bigtext_units.png')
    
    #fig = plt.figure()
    #ax2 = fig.add_subplot(111)
    #im2 = ax2.imshow(xycut_vels_raw, cmap = cmocean.cm.gray_r, vmin=0, vmax=np.percentile(xycut_vels_raw, 99.5))
    #ax2 = cutouts.set_all_labels(ax2, xax, ra_label, yax, dec_label, roundnum=1, degrees=True)
    #ax2.set_ylim(0, nycut)
    #cbar = plt.colorbar(im2)
    #cbar.outline.set_visible(False)

    
def plot_pos_neg_USM_FIR_vs_NHI():

    NHImap = fits.getdata("/Users/susanclark/Dropbox/NHImaps/GALFA-HI_NHISRCORR_VLSR-90+90kms.fits")
    P857map = fits.getdata("/Users/susanclark/Dropbox/Planck/HFI_SkyMap_857_2048_R3.01_ONGALFAHI.fits")
    
    submonopole = 0.64
    if submonopole is not None:
        P857map -= submonopole
        submonopolestr = "_submono{}".format(submonopole)
    else:
        submonopolestr = ""
    
    bstart = 30
    NHIcut = True
    NHImin = 0
    NHImax = 8
    
    maskbs = make_mask_2d(bstart=bstart, bstop=90, PS=True, NHIcut=NHIcut, NHImin=NHImin, NHImax=NHImax)  
    
    NHImap_div1E20 = NHImap/1.E20

    print("Including mask in cuts")
    cutwhere = np.where((P857map > 0) & (NHImap > 0) & (NHImap_div1E20 < NHImax) & (NHImap_div1E20 > NHImin) & (maskbs > 0))
    cutnonzero = np.where((P857map > 0) & (NHImap > 0) & (np.isnan(P857map) == False) & (np.isnan(NHImap) == False) & (maskbs > 0))
            
    cmap=copy.copy(matplotlib.cm.gist_gray_r)
    cmap._init()
    cmap._lut[0, :] = (1.0,1.0,1.0,0.05)#(0.1,0.1,0.1,0.05)


    H_bcut, xedges, yedges = np.histogram2d(NHImap_div1E20[cutnonzero], P857map[cutnonzero], bins=np.linspace(0, 20, 100))

    fig = plt.figure(figsize=(10, 5))
    X, Y = np.meshgrid(xedges, yedges)
    plt.pcolormesh(X, Y, H_bcut.T, norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=H_bcut.min(), vmax=H_bcut.max()),
                       cmap=cmap)
    #plt.pcolormesh(X, Y, H_bcut.T, vmin=0.05*np.median(H_bcut), vmax=0.95*np.median(H_bcut), cmap=cmap)
    plt.colorbar()

    xlimmax = 6
    ylimmax = 4
    plt.xlim(0, xlimmax)
    plt.ylim(0, ylimmax)

    plt.xlabel('NHI / 1E20')
    plt.ylabel('857 GHz')
    
    # unweighted means
    meancutnonzeroP857, meancutnonzeroNHI, meancutwhereP85764, meancutwhereNHI = weighted_mean(NHImap, P857map, weightsmap=None, mask=maskbs)
    
    vels=["1023", "1024", "1025"]
    #vels0=["1024"]
    
    usewhere = True
    
    # unsharp masked data
    vels_umask = get_USM_slice(vels=vels, fwhm=30, zeroed=True, Narrow=False, reverse=False, writemap=False)
    
    # weighted means
    meancutnonzeroP857, weightedmeancutnonzeroP857, meancutnonzeroNHI, weightedmeancutnonzeroNHI, meancutwhereP857, weightedmeancutwhereP857, meancutwhereNHI, weightedmeancutwhereNHI = weighted_mean(NHImap, P857map, weightsmap=vels_umask, mask=maskbs)
    if usewhere:
        mwmmuw_P857 = weightedmeancutwhereP857 - meancutwhereP857
        mwmmuw_NHI = weightedmeancutwhereNHI - meancutwhereNHI
    else:
        mwmmuw_P857 = weightedmeancutnonzeroP857 - meancutnonzeroP857
        mwmmuw_NHI = weightedmeancutnonzeroNHI - meancutnonzeroNHI
    
    inv_vels_umask = get_USM_slice(vels=vels, fwhm=30, zeroed=False, Narrow=False, reverse=False, writemap=False)
    inv_vels_umask *= -1
    inv_vels_umask[inv_vels_umask < 0] = 0
    invmeancutnonzeroP857, invweightedmeancutnonzeroP857, invmeancutnonzeroNHI, invweightedmeancutnonzeroNHI, invmeancutwhereP857, invweightedmeancutwhereP857, invmeancutwhereNHI, invweightedmeancutwhereNHI = weighted_mean(NHImap, P857map, weightsmap=inv_vels_umask, mask=maskbs)
    
    if usewhere:
        inv_mwmmuw_P857 = invweightedmeancutwhereP857 - invmeancutwhereP857
        inv_mwmmuw_NHI = invweightedmeancutwhereNHI - invmeancutwhereNHI
    else:
        inv_mwmmuw_P857 = invweightedmeancutnonzeroP857 - invmeancutnonzeroP857
        inv_mwmmuw_NHI = invweightedmeancutnonzeroNHI - invmeancutnonzeroNHI
        
    print("Inverse ratio of means: {}".format(invmeancutnonzeroP857/invmeancutnonzeroNHI))
    print("Inverse ratio of deltas: {}".format(inv_mwmmuw_P857/inv_mwmmuw_NHI))
    
    print("{} should equal {}".format(invmeancutnonzeroP857, meancutnonzeroP857))
    
    if usewhere:
        plt.plot(weightedmeancutwhereNHI, weightedmeancutwhereP857, 'o', color='tomato', label="weighted mean")  
        plt.plot(meancutwhereNHI, meancutwhereP857, 'o', color='cornflowerblue', label="unweighted mean")
        plt.plot(invweightedmeancutwhereNHI, invweightedmeancutwhereP857, 'o', color='yellow', label="inverse weighted mean")
        
        plt.hlines(y=weightedmeancutwhereP857, xmin=0, xmax=weightedmeancutwhereNHI, color='tomato', linestyle='dashed')
        plt.hlines(y=invweightedmeancutwhereP857, xmin=0, xmax=invweightedmeancutwhereNHI, color='yellow', linestyle='dashed')
        plt.hlines(y=meancutwhereP857, xmin=0, xmax=meancutwhereNHI, color='cornflowerblue', linestyle='dashed')
        plt.vlines(x=weightedmeancutwhereNHI, ymin=0, ymax=weightedmeancutwhereP857, color='tomato', linestyle='dashed')
        plt.vlines(x=invweightedmeancutwhereNHI, ymin=0, ymax=invweightedmeancutwhereP857, color='yellow', linestyle='dashed')
        plt.vlines(x=meancutwhereNHI, ymin=0, ymax=meancutwhereP857, color='cornflowerblue', linestyle='dashed')
        
        plt.arrow(meancutwhereNHI, meancutwhereP857, mwmmuw_NHI, mwmmuw_P857, color="tomato", width=0.001, head_width=0.005)
        plt.arrow(0, 0, meancutwhereNHI, meancutwhereP857, color="cornflowerblue", width=0.001, head_width=0.005)
        plt.arrow(invmeancutwhereNHI, invmeancutwhereP857, inv_mwmmuw_NHI, inv_mwmmuw_P857, color="yellow", width=0.001, head_width=0.005)
        
    else:
        plt.plot(weightedmeancutnonzeroNHI, weightedmeancutnonzeroP857, 'o', color='tomato', label="weighted mean") 
        plt.plot(meancutnonzeroNHI, meancutnonzeroP857, 'o', color='cornflowerblue', label="unweighted mean")
        plt.plot(invweightedmeancutnonzeroNHI, invweightedmeancutnonzeroP857, 'o', color='yellow', label="inverse weighted mean")
        
        plt.hlines(y=weightedmeancutnonzeroP857, xmin=0, xmax=weightedmeancutnonzeroNHI, color='tomato', linestyle='dashed') 
        plt.hlines(y=meancutnonzeroP857, xmin=0, xmax=meancutnonzeroNHI, color='cornflowerblue', linestyle='dashed')
        plt.hlines(y=invweightedmeancutnonzeroP857, xmin=0, xmax=invweightedmeancutnonzeroNHI, color='yellow', linestyle='dashed')
        plt.vlines(x=weightedmeancutnonzeroNHI, ymin=0, ymax=weightedmeancutnonzeroP857, color='tomato', linestyle='dashed')
        plt.vlines(x=meancutnonzeroNHI, ymin=0, ymax=meancutnonzeroP857, color='cornflowerblue', linestyle='dashed')
        plt.vlines(x=invweightedmeancutnonzeroNHI, ymin=0, ymax=invweightedmeancutnonzeroP857, color='yellow', linestyle='dashed')
        
        
        plt.arrow(meancutnonzeroNHI, meancutnonzeroP857, mwmmuw_NHI, mwmmuw_P857, color="tomato", width=0.001, head_width=0.005)
        plt.arrow(0, 0, meancutnonzeroNHI, meancutnonzeroP857, color="cornflowerblue", width=0.001, head_width=0.005)
        plt.arrow(invmeancutnonzeroNHI, invmeancutnonzeroP857, inv_mwmmuw_NHI, inv_mwmmuw_P857, color="yellow", width=0.001, head_width=0.005)
        
    plt.legend()
    
    print("Ratio of means, nonzero: {}".format(meancutnonzeroP857/meancutnonzeroNHI))
    print("Ratio of means, where: {}".format(meancutwhereP857/meancutwhereNHI))
    print("Ratio of deltas: {}".format(mwmmuw_P857/mwmmuw_NHI))
    print("Ratio of inv deltas: {}".format(inv_mwmmuw_P857/inv_mwmmuw_NHI))
    
def plot_FIR_vs_NHI():

    NHImap = fits.getdata("/Users/susanclark/Dropbox/NHImaps/GALFA-HI_NHISRCORR_VLSR-90+90kms.fits")
    P857map = fits.getdata("/Users/susanclark/Dropbox/Planck/HFI_SkyMap_857_2048_R3.01_ONGALFAHI.fits")
    
    submonopole = 0.64
    if submonopole is not None:
        P857map -= submonopole
        submonopolestr = "_submono{}".format(submonopole)
    else:
        submonopolestr = ""
    
    bstart = 30
    NHIcut = True
    NHImin = 0
    NHImax = 8
    
    maskbs = make_mask_2d(bstart=bstart, bstop=90, PS=True, NHIcut=NHIcut, NHImin=NHImin, NHImax=NHImax) 
    maskPS = make_mask_2d(bstart=bstart, bstop=90, PS=True, NHIcut=False)  
    
    NHImap_div1E20 = NHImap/1.E20

    print("Including mask in cuts")
    cutwhere = np.where((P857map > 0) & (NHImap > 0) & (NHImap_div1E20 < NHImax) & (NHImap_div1E20 > NHImin) & (maskbs > 0))
    cutnonzero = np.where((P857map > 0) & (NHImap > 0) & (np.isnan(P857map) == False) & (np.isnan(NHImap) == False) & (maskPS > 0))
            
    cmap=copy.copy(matplotlib.cm.gist_gray_r)
    #cmap._init()
    #cmap._lut[0, :] = (1.0,1.0,1.0,0.05)#(0.1,0.1,0.1,0.05)

    NHI_div1E20_max = 10

    H_bcut, xedges, yedges = np.histogram2d(NHImap_div1E20[cutnonzero], P857map[cutnonzero], bins=np.linspace(0, NHI_div1E20_max, 1000))
    maxnhist = np.nanmax(H_bcut)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    X, Y = np.meshgrid(xedges, yedges)
    #plt.pcolormesh(X, Y, H_bcut.T, norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=0, vmax=H_bcut.max()), cmap=cmap)
    #plt.pcolormesh(X, Y, H_bcut.T, norm=colors.LogNorm(vmin=1., vmax=H_bcut.max()), cmap=cmap)
    #plt.pcolormesh(X, Y, H_bcut.T, cmap=cmap)
    im = ax.pcolormesh(X, Y, H_bcut.T, norm=colors.LogNorm(vmin=0.001*maxnhist, vmax=0.999*maxnhist), cmap=cmap)
    #plt.scatter(NHImap_div1E20[cutnonzero], P857map[cutnonzero], s=1, color='black', alpha=0.1)
    
    left, bottom, width, height = [0.65, 0.81, 0.15, 0.03]
    cbarax = fig.add_axes([left, bottom, width, height])
    #plt.pcolormesh(X, Y, H_bcut.T, vmin=0.05*np.median(H_bcut), vmax=0.95*np.median(H_bcut), cmap=cmap)
    cbar = plt.colorbar(im, cax=cbarax, orientation='horizontal')
    
    #plt.axes().set_aspect('equal')#, 'datalim')
    ax.set_aspect('equal')

    xlimmax = NHI_div1E20_max
    ylimmax = NHI_div1E20_max
    ax.set_xlim(0, xlimmax)
    ax.set_ylim(0, ylimmax)

    textlabelsize = 20
    ax.set_xlabel(r'$\mathrm{N}_\mathrm{HI}$ / $10^{20}$ $[\mathrm{cm}^{-2}]$', size=textlabelsize, labelpad=10)
    ax.set_ylabel(r'$I_{857}$ $[\mathrm{MJy/sr}]$', size=textlabelsize, labelpad=8)
    
    #ax.xaxis.set_label_params(pad=2)
    
    #cbar.outline.set_visible(False)
    
    #ax.set_xticks([1, 3, 5, 7, 9])
    ax.set_xticks([0, 1, 2, 5, 6, 7, 8, 9])
    ax.set_xticklabels([r"$0$", r"$1$", r"$2$", r"$5$", r"$6$", r"$7$", r"$8$", r"$9$"], size=12)
    ax.set_yticks([0, 1, 3, 4, 5, 6, 7, 8, 9])
    ax.set_yticklabels([r"$0$", r"$1$", r"$3$", r"$4$", r"$5$", r"$6$", r"$7$", r"$8$", r"$9$"], size=12)
    
    # unweighted means
    meancutnonzeroP857, meancutnonzeroNHI, meancutwhereP85764, meancutwhereNHI = weighted_mean(NHImap, P857map, weightsmap=None, mask=maskbs)
    
    vels=["1023", "1024", "1025"]
    #vels0=["1024"]
    
    usewhere = True
    
    # unsharp masked data
    #vels_umask = get_USM_slice(vels=vels, fwhm=30, zeroed=True, Narrow=False, reverse=False, writemap=False)
    vels_umask = load_USM_map(vels=vels, fwhm=30, zeroed=True, Narrow=False, addnoise=False, noisefactor=0)
    
    # weighted means
    meancutnonzeroP857, weightedmeancutnonzeroP857, meancutnonzeroNHI, weightedmeancutnonzeroNHI, meancutwhereP857, weightedmeancutwhereP857, meancutwhereNHI, weightedmeancutwhereNHI = weighted_mean(NHImap, P857map, weightsmap=vels_umask, mask=maskPS, nhi_max_fit=8, nhi_min_fit=0)
    #def weighted_mean(NHImap, P857map, weightsmap=None, mask=None, nhi_max_fit=8, nhi_min_fit=0)
    if usewhere:
        mwmmuw_P857 = weightedmeancutwhereP857 - meancutwhereP857
        mwmmuw_NHI = weightedmeancutwhereNHI - meancutwhereNHI
    else:
        mwmmuw_P857 = weightedmeancutnonzeroP857 - meancutnonzeroP857
        mwmmuw_NHI = weightedmeancutnonzeroNHI - meancutnonzeroNHI
        
    weightedcolor='tomato'
    unweightedcolor='cornflowerblue'
    
    if usewhere:
        ax.plot(weightedmeancutwhereNHI, weightedmeancutwhereP857, 'o', color=weightedcolor, label="weighted mean")  
        ax.plot(meancutwhereNHI, meancutwhereP857, 'o', color='cornflowerblue', label="unweighted mean")
        
        ax.hlines(y=weightedmeancutwhereP857, xmin=0, xmax=weightedmeancutwhereNHI, color=weightedcolor, linestyle='dashed')
        ax.hlines(y=meancutwhereP857, xmin=0, xmax=meancutwhereNHI, color='cornflowerblue', linestyle='dashed')
        ax.vlines(x=weightedmeancutwhereNHI, ymin=0, ymax=weightedmeancutwhereP857, color=weightedcolor, linestyle='dashed')
        ax.vlines(x=meancutwhereNHI, ymin=0, ymax=meancutwhereP857, color='cornflowerblue', linestyle='dashed')
        
        ax.arrow(meancutwhereNHI, meancutwhereP857, mwmmuw_NHI, mwmmuw_P857, color=weightedcolor, width=0.001, head_width=0.005)
        ax.arrow(0, 0, meancutwhereNHI, meancutwhereP857, color="cornflowerblue", width=0.001, head_width=0.005)
        
    else:
        ax.plot(weightedmeancutnonzeroNHI, weightedmeancutnonzeroP857, 'o', color=weightedcolor, label="weighted mean") 
        ax.plot(meancutnonzeroNHI, meancutnonzeroP857, 'o', color='cornflowerblue', label="unweighted mean")
        
        ax.hlines(y=weightedmeancutnonzeroP857, xmin=0, xmax=weightedmeancutnonzeroNHI, color=weightedcolor, linestyle='dashed') 
        ax.hlines(y=meancutnonzeroP857, xmin=0, xmax=meancutnonzeroNHI, color='cornflowerblue', linestyle='dashed')
        ax.vlines(x=weightedmeancutnonzeroNHI, ymin=0, ymax=weightedmeancutnonzeroP857, color=weightedcolor, linestyle='dashed')
        ax.vlines(x=meancutnonzeroNHI, ymin=0, ymax=meancutnonzeroP857, color='cornflowerblue', linestyle='dashed')
        
        ax.arrow(meancutnonzeroNHI, meancutnonzeroP857, mwmmuw_NHI, mwmmuw_P857, color=weightedcolor, width=0.001, head_width=0.005)
        ax.arrow(0, 0, meancutnonzeroNHI, meancutnonzeroP857, color="cornflowerblue", width=0.001, head_width=0.005)
        
    #plt.legend()
    
    print("Ratio of means, nonzero: {}".format(meancutnonzeroP857/meancutnonzeroNHI))
    print("Ratio of means, where: {}".format(meancutwhereP857/meancutwhereNHI))
    print("Ratio of deltas: {}".format(mwmmuw_P857/mwmmuw_NHI))
    
    ypos = -0.5
    xpos = -0.5
    textsize = 15
    if usewhere:
        ax.text(weightedmeancutwhereNHI, ypos, r"$\bar{N}_\mathrm{HI}^w$", color=weightedcolor, horizontalalignment='center', size=textsize)
        ax.text(meancutwhereNHI, ypos, r"$\bar{N}_\mathrm{HI}$", color=unweightedcolor, horizontalalignment='center', size=textsize)
        ax.text(xpos, weightedmeancutwhereP857, r"$\bar{I}_{857}^w$", color=weightedcolor, horizontalalignment='center', size=textsize)
        ax.text(xpos, meancutwhereP857, r"$\bar{I}_{857}$", color=unweightedcolor, horizontalalignment='center', size=textsize)
    else:
        ax.text(weightedmeancutnonzeroNHI, ypos, r"$\bar{N}_\mathrm{HI}^w$", color=weightedcolor, horizontalalignment='center', size=textsize)
        ax.text(meancutnonzeroNHI, ypos, r"$\bar{N}_\mathrm{HI}$", color=unweightedcolor, horizontalalignment='center', size=textsize)
                    
    
def plotscatter():
    NHImap = fits.getdata("/Users/susanclark/Dropbox/NHImaps/GALFA-HI_NHISRCORR_VLSR-90+90kms.fits")
    P857map = fits.getdata("/Users/susanclark/Dropbox/Planck/HFI_SkyMap_857_2048_R3.01_ONGALFAHI.fits")
    
    submonopole = 0.64
    if submonopole is not None:
        P857map -= submonopole
        submonopolestr = "_submono{}".format(submonopole)
    else:
        submonopolestr = ""
    
    bstart = 30
    NHIcut = True
    NHImin = 0
    NHImax = 8
    
    maskbs = make_mask_2d(bstart=bstart, bstop=90, PS=True, NHIcut=NHIcut, NHImin=NHImin, NHImax=NHImax)  
    
    NHImap_div1E20 = NHImap/1.E20
    
    vels=["1023", "1024", "1025"]
    #vels0=["1024"]
    
    # unsharp masked data
    vels_umask = get_USM_slice(vels=vels, fwhm=30, zeroed=True, Narrow=False, reverse=False, writemap=False)
    inv_vels_umask = get_USM_slice(vels=vels, fwhm=30, zeroed=False, Narrow=False, reverse=False, writemap=False)
    inv_vels_umask *= -1
    inv_vels_umask[inv_vels_umask < 0] = 0
    
    cutwhere = np.where((P857map > 0) & (NHImap > 0) & (NHImap_div1E20 < NHImax) & (NHImap_div1E20 > NHImin) & (maskbs > 0))
    cutnonzero = np.where((P857map > 0) & (NHImap > 0) & (np.isnan(P857map) == False) & (np.isnan(NHImap) == False) & (maskbs > 0))
    
    cutwhere_umask = np.where((P857map > 0) & (NHImap > 0) & (NHImap_div1E20 < NHImax) & (NHImap_div1E20 > NHImin) & (maskbs > 0) & (vels_umask > 0))
    cutwhere_invumask = np.where((P857map > 0) & (NHImap > 0) & (NHImap_div1E20 < NHImax) & (NHImap_div1E20 > NHImin) & (maskbs > 0) & (inv_vels_umask > 0))
    
    plt.scatter(NHImap[cutwhere_umask], P857map[cutwhere_umask], alpha=0.1, color="red", s=0.1)
    plt.scatter(NHImap[cutwhere_invumask], P857map[cutwhere_invumask], alpha=0.1, color="blue", s=0.1)

def plot_USM_chunks():
    vels7=["1017", "1018", "1019", "1020", "1021", "1022", "1023", "1024", "1025", "1026", "1027", "1028", "1029", "1030", "1031"]
    vels6=["1018", "1019", "1020", "1021", "1022", "1023", "1024", "1025", "1026", "1027", "1028", "1029", "1030"]
    vels5=["1019", "1020", "1021", "1022", "1023", "1024", "1025", "1026", "1027", "1028", "1029"]
    vels4=["1020", "1021", "1022", "1023", "1024", "1025", "1026", "1027", "1028"]
    vels3=["1021", "1022", "1023", "1024", "1025", "1026", "1027"]
    vels2=["1022", "1023", "1024", "1025", "1026"]
    vels1=["1023", "1024", "1025"]
    vels0=["1024"]
    allvels = [vels0, vels1, vels2, vels3, vels4, vels5, vels6, vels7]
    
    starty=400
    stopy=1800
    startx=9800
    stopx=11400
    
    fig = plt.figure()
    
    ax0 = fig.add_subplot(3, 3, 1)
    narrow_umask = get_USM_slice(vels=["1024"], fwhm=30, zeroed=True, Narrow=True, reverse=False, writemap=False)
    im = ax0.imshow(narrow_umask[starty:stopy, startx:stopx], cmap="viridis")
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="5%", pad=0.05) 
    plt.colorbar(im, cax=cax)
    ax0.set_title("0.18 km/s")
    ax0.set_xticks([])
    ax0.set_yticks([])
    
    for _i, _vels in enumerate(allvels):
        _i += 1
        vels_umask = get_USM_slice(vels=_vels, fwhm=30, zeroed=True, Narrow=False, reverse=False, writemap=False)
        ax = fig.add_subplot(3, 3, _i+1)
        im = ax.imshow(vels_umask[starty:stopy, startx:stopx], cmap="viridis")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05) 
        plt.colorbar(im, cax=cax)
        ax.set_title("{} km/s".format(np.round(0.184*4*len(_vels), 2) ))
        ax.set_xticks([])
        ax.set_yticks([])
    
def plot_wedge_figure(deltaNHI=False):
    if LOCAL:
        nhi90map = fits.getdata("/Users/susanclark/Dropbox/NHImaps/GALFA-HI_NHISRCORR_VLSR-90+90kms.fits")
        p857map = fits.getdata("/Users/susanclark/Dropbox/Planck/HFI_SkyMap_857_2048_R3.01_ONGALFAHI.fits")
    else:
        nhi90map = fits.getdata("/data/seclark/GALFADR2/NHImaps/GALFA-HI_NHISRCORR_VLSR-90+90kms.fits")
        p857map = fits.getdata("/data/seclark/Planck/HFI_SkyMap_857_2048_R3.01_ONGALFAHI.fits")    
    
    submonopole = 0.64
    if submonopole is not None:
        p857map -= submonopole
        submonopolestr = "_submono{}".format(submonopole)
    else:
        submonopolestr = ""
    
    bstart = 30
    NHIcut = True
    NHImin = 0
    NHImax = 8
    bstype = "radec"
    maskbs = make_mask_2d(bstart=bstart, bstop=90, PS=True, NHIcut=NHIcut, NHImin=NHImin, NHImax=NHImax)  
    
    vels7=["1017", "1018", "1019", "1020", "1021", "1022", "1023", "1024", "1025", "1026", "1027", "1028", "1029", "1030", "1031"]
    vels6=["1018", "1019", "1020", "1021", "1022", "1023", "1024", "1025", "1026", "1027", "1028", "1029", "1030"]
    vels5=["1019", "1020", "1021", "1022", "1023", "1024", "1025", "1026", "1027", "1028", "1029"]
    vels4=["1020", "1021", "1022", "1023", "1024", "1025", "1026", "1027", "1028"]
    vels3=["1021", "1022", "1023", "1024", "1025", "1026", "1027"]
    vels2=["1022", "1023", "1024", "1025", "1026"]
    vels1=["1023", "1024", "1025"]
    vels0=["1024"]
    allvels = [vels0, vels1, vels2, vels3, vels4, vels5, vels6, vels7]

    all_meancutnonzeroP857 = np.zeros(len(allvels)+1)
    all_weightedmeancutnonzeroP857 = np.zeros(len(allvels)+1)
    all_meancutnonzeroNHI = np.zeros(len(allvels)+1)
    all_weightedmeancutnonzeroNHI = np.zeros(len(allvels)+1)
    all_meancutwhereP857 = np.zeros(len(allvels)+1)
    all_weightedmeancutwhereP857 = np.zeros(len(allvels)+1)
    all_meancutwhereNHI = np.zeros(len(allvels)+1)
    all_weightedmeancutwhereNHI = np.zeros(len(allvels)+1)

    loaddata=True
    if loaddata:
        all_meancutnonzeroP857 = np.load('../data/all_meancutnonzeroP857_bstart{}_NHImin{}_max{}{}.npy'.format(bstart, NHImin, NHImax, submonopolestr))
        all_weightedmeancutnonzeroP857 = np.load('../data/all_weightedmeancutnonzeroP857_bstart{}_NHImin{}_max{}{}.npy'.format(bstart, NHImin, NHImax, submonopolestr))
        all_meancutnonzeroNHI = np.load('../data/all_meancutnonzeroNHI_bstart{}_NHImin{}_max{}{}.npy'.format(bstart, NHImin, NHImax, submonopolestr))
        all_weightedmeancutnonzeroNHI = np.load('../data/all_weightedmeancutnonzeroNHI_bstart{}_NHImin{}_max{}{}.npy'.format(bstart, NHImin, NHImax, submonopolestr))
        
    else:
        narrow_umask = get_USM_slice(vels=["1024"], fwhm=30, zeroed=True, Narrow=True, reverse=False, writemap=False)
        all_meancutnonzeroP857[0], all_weightedmeancutnonzeroP857[0], all_meancutnonzeroNHI[0], all_weightedmeancutnonzeroNHI[0], all_meancutwhereP857[0], all_weightedmeancutwhereP857[0], all_meancutwhereNHI[0], all_weightedmeancutwhereNHI[0] = weighted_mean(nhi90map, p857map, weightsmap=narrow_umask, mask=maskbs)

        for _i, _vels in enumerate(allvels):
            _i += 1
            vels_umask = get_USM_slice(vels=_vels, fwhm=30, zeroed=True, Narrow=False, reverse=False, writemap=False)
            all_meancutnonzeroP857[_i], all_weightedmeancutnonzeroP857[_i], all_meancutnonzeroNHI[_i], all_weightedmeancutnonzeroNHI[_i], all_meancutwhereP857[_i], all_weightedmeancutwhereP857[_i], all_meancutwhereNHI[_i], all_weightedmeancutwhereNHI[_i] = weighted_mean(nhi90map, p857map, weightsmap=vels_umask, mask=maskbs)
        
        np.save('../data/all_meancutnonzeroP857_bstart{}_NHImin{}_max{}{}.npy'.format(bstart, NHImin, NHImax, submonopolestr), all_meancutnonzeroP857)
        np.save('../data/all_weightedmeancutnonzeroP857_bstart{}_NHImin{}_max{}{}.npy'.format(bstart, NHImin, NHImax, submonopolestr), all_weightedmeancutnonzeroP857)
        np.save('../data/all_meancutnonzeroNHI_bstart{}_NHImin{}_max{}{}.npy'.format(bstart, NHImin, NHImax, submonopolestr), all_meancutnonzeroNHI)
        np.save('../data/all_weightedmeancutnonzeroNHI_bstart{}_NHImin{}_max{}{}.npy'.format(bstart, NHImin, NHImax, submonopolestr), all_weightedmeancutnonzeroNHI)

    Nblocks = 40#50
    #Nsamples = 10000000
    Nsamples = 1000000
    onesigmaerrs = np.zeros((2, len(allvels)+1), np.float_)
    onesigmaBS = np.zeros((2, len(allvels)+1), np.float_)
    try:
        onesigmaBS[0, 0], onesigmaBS[1, 0] = load_bootstrap_data(vstart="1024", vstop="1024", bstart=bstart, NHIcut=NHIcut, NHImin=NHImin, NHImax=NHImax, Narrow=True, Nblocks=Nblocks, Nsamples=Nsamples, submonopole=submonopole, bstype=bstype)
        print("NARROW values: {}, {}".format(onesigmaBS[0, 0], onesigmaBS[1, 0]))
    except:
        print("Could not load narrow.")
        onesigmaBS[0] = 1.0
        
    for _i in np.arange(len(allvels)):
        startvel = allvels[_i][0]
        stopvel = allvels[_i][-1]
        print(startvel, stopvel)
        try:
            # perc16, perc84
            onesigmaBS[0, _i+1], onesigmaBS[1, _i+1] = load_bootstrap_data(vstart=startvel, vstop=stopvel, bstart=bstart, NHIcut=NHIcut, NHImin=NHImin, NHImax=NHImax, Narrow=False, Nblocks=Nblocks, Nsamples=Nsamples, submonopole=submonopole, bstype=bstype)

        except:
            print("Could not load")
            onesigmaBS[0, _i+1] = 1.0

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    thicknesses = np.zeros(len(allvels) + 1)
    velwidths = [0.184*4*len(vel) for vel in allvels]
    thicknesses[0] = 0.184
    thicknesses[1:] = velwidths

    delta857 = all_weightedmeancutnonzeroP857 - all_meancutnonzeroP857
    #onesigmaerrs[0, :] -= delta857
    #onesigmaerrs[1, :] = delta857 - onesigmaerrs[1, :]
    onesigmaerrs[1, :] = onesigmaBS[1, :] - delta857
    onesigmaerrs[0, :] = delta857 - onesigmaBS[0, :]
    
    print("delta857: {}".format(delta857))
    print("onesigmaBS: {}".format(onesigmaBS))
    
    #plt.plot(thicknesses, delta857, 'o', color='teal')
    ax.errorbar(thicknesses, delta857, yerr=onesigmaerrs, fmt='o', color='teal', ecolor='lightgray', elinewidth=3, capsize=0);
    ax.set_xlabel(r'$\mathrm{Channel}$ $\mathrm{Width}$ $[\mathrm{km/s}]$', size=15)
    ax.set_ylabel(r'$\Delta$ $\mathrm{I}_{857}$ $[\mathrm{MJy/sr}]$', size=15)
    offhi857 = 54
    maxoffhi857 = 59
    minoffhi857 = 49
    #plt.axhspan(minoffhi857, maxoffhi857, alpha=0.1, color='C1')

    thinlim = 0.184*4*4 # width used in PRL and in the data Lazarian used

    maxx=np.max(thicknesses)
    minx= 0.0


    ax.fill([thinlim, maxx,maxx], [0.01, 0.01, np.max(delta857)], color='C2', alpha=0.1, linewidth=0.0)
    #plt.ylim(-0.1, 2)
    ax.set_ylim(-0.05, 0.75)
    ax.set_xlim(minx, maxx+0.01*maxx)
    print(minx, maxx, maxx+0.01*maxx)
    
    ax.hlines(0., xmin=0, xmax=maxx, linestyle='solid', color='C2', alpha=0.1, lw=7.5)
    ax.vlines(thinlim, ymin=-0.05, ymax=0.7, linestyle='dashed', color="gray")
    
    l2 = np.array((thinlim, 0))
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    ax.spines["left"].set_bounds(0, 0.7)
    ax.spines["bottom"].set_bounds(0, maxx)
    
    ticklabelsize = 12
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(ticklabelsize) 
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(ticklabelsize) 

    # Rotate angle
    angle = (np.max(delta857))/(maxx-thinlim)
    trans_angle = plt.gca().transData.transform_angles(np.array((angle,)),
                                                       l2.reshape((1, 2)))[0]

    # Plot text
    #plt.text(l2[0], l2[1], 'text rotated correctly', fontsize=16,
    #               rotation=trans_angle, rotation_mode='anchor')
    
    #plt.savefig('../figures/deltaFIR_by_channel_vel_BSerrs_bstart{}_NHImin{}_max{}{}_bsstype{}_Nblocks{}_Nsamples{}_texlabels.pdf'.format(bstart, NHImin, NHImax, submonopolestr, bstype, Nblocks, Nsamples))

"""
if __name__ == "__main__":
    if LOCAL:
        nhi90map = fits.getdata("/Users/susanclark/Dropbox/NHImaps/GALFA-HI_NHISRCORR_VLSR-90+90kms.fits")
        p857map = fits.getdata("/Users/susanclark/Dropbox/Planck/HFI_SkyMap_857_2048_R3.01_ONGALFAHI.fits")
    else:
        nhi90map = fits.getdata("/data/seclark/GALFADR2/NHImaps/GALFA-HI_NHISRCORR_VLSR-90+90kms.fits")
        p857map = fits.getdata("/data/seclark/Planck/HFI_SkyMap_857_2048_R3.01_ONGALFAHI.fits")

    submonopole = 0.64
    if submonopole is not None:
        p857map -= submonopole

    Nblocks = 40#50
    bstart = 30
    NHIcut=True
    NHImax=8
    NHImin=0
    bstype="radec"
    allblocks = []
    for _i in np.arange(Nblocks):
        _mtest = make_mask_2d(bstart=bstart, bstop=90, NHIcut=NHIcut, NHImin=NHImin, NHImax=NHImax, PS=True, bootstrapchunks=Nblocks, bsnum=_i, bstype=bstype, negmask=p857map)
        allblocks.append(_mtest)
        
    #vels=["1017", "1018", "1019", "1020", "1021", "1022", "1023", "1024", "1025", "1026", "1027", "1028", "1029", "1030", "1031"]
    #vels=["1018", "1019", "1020", "1021", "1022", "1023", "1024", "1025", "1026", "1027", "1028", "1029", "1030"]
    #vels=["1019", "1020", "1021", "1022", "1023", "1024", "1025", "1026", "1027", "1028", "1029"]
    #vels=["1020", "1021", "1022", "1023", "1024", "1025", "1026", "1027", "1028"]
    #vels=["1021", "1022", "1023", "1024", "1025", "1026", "1027"]
    #vels=["1022", "1023", "1024", "1025", "1026"]
    #vels=["1023", "1024", "1025"]
    vels=["1024"]
    
    Narrow = True
    umask = get_USM_slice(vels, fwhm=30, zeroed=True, Narrow=Narrow, reverse=False, writemap=False)

    allP857blocks = []
    allNHIblocks = []
    allumaskblocks = []
    for _i in np.arange(Nblocks):
        selectpix = np.where( (allblocks[_i] > 0) & (p857map > 0) & (nhi90map > 0) & (np.isnan(p857map)==False) & (np.isnan(nhi90map)==False) & (np.isnan(umask)==False) )
        allP857blocks.append(p857map[selectpix])
        allNHIblocks.append(nhi90map[selectpix])
        allumaskblocks.append(umask[selectpix])
        
    all_block_lens = [len(allP857blocks[_i]) for _i in np.arange(Nblocks)]
    print(all_block_lens)
    all_blockP857_arr = np.zeros((Nblocks, np.max(all_block_lens)), np.float_) 
    all_blockNHI_arr = np.zeros((Nblocks, np.max(all_block_lens)), np.float_)
    all_blockumask_arr = np.zeros((Nblocks, np.max(all_block_lens)), np.float_)
    for _i in np.arange(Nblocks):
        all_blockP857_arr[_i, :all_block_lens[_i]] = allP857blocks[_i]
        all_blockNHI_arr[_i, :all_block_lens[_i]] = allNHIblocks[_i]
        all_blockumask_arr[_i, :all_block_lens[_i]] = allumaskblocks[_i]
    
    all_block_lens = np.array(all_block_lens)
    
    print("all_blockNHI_arr shape: {}".format(all_blockNHI_arr.shape))
    
    block_unweighted_NHI = np.nansum(all_blockNHI_arr, axis=1)
    block_unweighted_P857 = np.nansum(all_blockP857_arr, axis=1)
    block_weighted_NHI = np.nansum(all_blockNHI_arr*all_blockumask_arr, axis=1)
    block_weighted_P857 = np.nansum(all_blockP857_arr*all_blockumask_arr, axis=1)
    block_weightsums = np.nansum(all_blockumask_arr, axis=1)
    
    print("block_unweighted_NHI shape: {}".format(block_unweighted_NHI.shape))
        
    Nsamples = 1000000#00
    BS_meanP857 = np.zeros(Nsamples)
    BS_weightedmeanP857 = np.zeros(Nsamples)
    BS_meanNHI = np.zeros(Nsamples)
    BS_weightedmeanNHI = np.zeros(Nsamples)
    
    time0 = time.time()
    
    for _i in np.arange(Nsamples):
        randints = np.random.randint(Nblocks, size=Nblocks)
        #randints = np.arange(Nblocks)
        
        BS_meanNHI[_i] = np.nansum(block_unweighted_NHI[randints])/np.sum(all_block_lens[randints])
        BS_meanP857[_i] = np.nansum(block_unweighted_P857[randints])/np.sum(all_block_lens[randints])
        BS_weightedmeanNHI[_i] = np.nansum(block_weighted_NHI[randints])/np.sum(block_weightsums[randints])
        BS_weightedmeanP857[_i] = np.nansum(block_weighted_P857[randints])/np.sum(block_weightsums[randints])
        
    orderedints = np.arange(Nblocks)
    meanNHI = np.nansum(block_unweighted_NHI[orderedints])/np.sum(all_block_lens[orderedints])
    meanP857 = np.nansum(block_unweighted_P857[orderedints])/np.sum(all_block_lens[orderedints])
    weightedmeanNHI = np.nansum(block_weighted_NHI[orderedints])/np.sum(block_weightsums[orderedints])
    weightedmeanP857 = np.nansum(block_weighted_P857[orderedints])/np.sum(block_weightsums[orderedints])
    ordered_deltaFIR = weightedmeanP857 - meanP857
    print("ordered_deltaFIR = {}".format(ordered_deltaFIR))
            
    time1 = time.time()
    print("Took {} minutes".format((time1-time0)/60.))

    if Narrow:
        narrowstr = "Narrow_"
    else:
        narrowstr = ""
        
    if NHIcut:
        NHImaxstr = "_NHImin{}_max{}".format(NHImin, NHImax)
    else:
        NHImaxstr = ""
        
    if submonopole is not None:
        submonopolestr = "_submono{}".format(submonopole)
    else:
        submonopolestr = ""

    np.save('../data/bootstrap_data/BS_meanNHI_vel{}_to_{}_{}Nblocks{}_Nsamples{}_bstart{}{}{}_bstype{}.npy'.format(vels[0], vels[-1], narrowstr, Nblocks, Nsamples, bstart, NHImaxstr, submonopolestr, bstype), BS_meanNHI)
    np.save('../data/bootstrap_data/BS_meanP857_vel{}_to_{}_{}Nblocks{}_Nsamples{}_bstart{}{}{}_bstype{}.npy'.format(vels[0], vels[-1], narrowstr, Nblocks, Nsamples, bstart, NHImaxstr, submonopolestr, bstype), BS_meanP857)
    np.save('../data/bootstrap_data/BS_weightedmeanNHI_vel{}_to_{}_{}Nblocks{}_Nsamples{}_bstart{}{}{}_bstype{}.npy'.format(vels[0], vels[-1], narrowstr, Nblocks, Nsamples, bstart, NHImaxstr, submonopolestr, bstype), BS_weightedmeanNHI)
    np.save('../data/bootstrap_data/BS_weightedmeanP857_vel{}_to_{}_{}Nblocks{}_Nsamples{}_bstart{}{}{}_bstype{}.npy'.format(vels[0], vels[-1], narrowstr, Nblocks, Nsamples, bstart, NHImaxstr, submonopolestr, bstype), BS_weightedmeanP857)

    BS_deltaNHI = BS_weightedmeanNHI - BS_meanNHI

    BS_deltaFIR = BS_weightedmeanP857 - BS_meanP857
    perc16 = np.percentile(BS_deltaFIR, 16)
    perc84 = np.percentile(BS_deltaFIR, 84)
    print("delta FIR 1 sigma error bars from {} to {}".format(perc16, perc84))

    maskwhole = make_mask_2d(bstart=bstart, bstop=90, NHIcut=NHIcut, NHImin=NHImin, NHImax=NHImax, PS=True, negmask=p857map)
    print("maskwhole sum = {}".format(np.nansum(maskwhole)))
    maskwhole[np.where(p857map < 0)] = 0
    maskwhole[np.where(nhi90map < 0)] = 0
    maskwhole[np.where(np.isnan(p857map)==True)] = 0
    maskwhole[np.where(np.isnan(nhi90map)==True)] = 0
    maskwhole[np.where(np.isnan(umask)==True)] = 0
    
    print("{} should equal {}".format(np.nansum(maskwhole), np.nansum(allblocks) ))
    
    meancutnonzeroP857, weightedmeancutnonzeroP857, meancutnonzeroNHI, weightedmeancutnonzeroNHI, meancutwhereP857, weightedmeancutwhereP857, meancutwhereNHI, weightedmeancutwhereNHI = weighted_mean(nhi90map, p857map, weightsmap=umask, mask=maskwhole)

    wholesky_deltaFIR = weightedmeancutnonzeroP857 - meancutnonzeroP857 
    print("wholesky deltaFIR = {}".format(wholesky_deltaFIR))
    
    print("am i lopsided? {} vs {}".format(wholesky_deltaFIR-perc16, perc84-wholesky_deltaFIR))
    #plt.figure()
    #plt.hist(BS_deltaFIR, bins=100);
    #plt.vlines(wholesky_deltaFIR, ymin=0, ymax=500)
    #plt.vlines(ordered_deltaFIR, ymin=0, ymax=400, color='red')
"""
if __name__ == "__main__":

    for _vel in np.arange(955, 1093):#1092):
        if _vel < 1000:
            _velstr = str(0)+str(_vel)
        else:
            _velstr = str(_vel)
        uslice = get_USM_slice(vels=[_velstr], fwhm=30, zeroed=True, Narrow=False, reverse=False, writemap=True, returnrawslice=False, addnoise=False, noisefactor=1)
        
        
    
        