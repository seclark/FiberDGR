import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import matplotlib
from astropy.io import fits
import matplotlib.colors as colors
import copy
import time

# local repo imports
import sys
sys.path.insert(0, '../../GalfaCuber/code')
import galfa_vel_helpers as gvh

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
    
def get_USM_slice(vels=["1024"], fwhm=10, zeroed=True, Narrow=False, reverse=False, writemap=False):
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
        # if longer than one slice, add the rest
        if len(vels) > 1:
            for _vel in vels[1:]:
                velkms = vels_Narrow_dict[_vel]
                slice_fn = DR2_Narrow_slice_root+"GALFA_HI_N_S{}_V{}kms.fits".format(_vel, velkms)
                slice_data += fits.getdata(slice_fn)
        
        #Narrow_slice_fn = "/disks/jansky/a/users/goldston/susan/Wide_maps/Allsky_Narrow/GALFA-HI_VLSR_460.1m_per_s.fits"
        #slice_data = fits.getdata(Narrow_slice_fn)
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
                    
    if reverse:
        print("reverse!")
        umask_slice_data = gaussian_umask(slice_data, fwhm=fwhm, zeroed=False)
        umask_slice_data *= -1
        umask_slice_data[np.where(umask_slice_data < 0)] = 0
        
    else:
        umask_slice_data = gaussian_umask(slice_data, fwhm=fwhm, zeroed=zeroed)
        
    umask_slice_data[np.where(np.isnan(umask_slice_data)==True)] = 0 # zero out nans
    
    if writemap:
        outfn = DR2_Wide_slice_root+"GALFA_HI_W_vels{}_to_{}_USM{}_zeroed_{}.fits".format(vels[0], vels[-1], fwhm, zeroed)
        outhdr= fits.getheader(DR2_Wide_slice_root+"GALFA_HI_W_S{}_V{}kms.fits".format(vels[0], vel0kms))
        fits.writeto(outfn, umask_slice_data, outhdr)
    
    return umask_slice_data 
    
def weighted_mean(NHImap, P857map, weightsmap=None, mask=None):
    
    NHImap_div1E20 = NHImap/1.E20

    # could make this 25th and 75th weighted percentiles or something
    nhi_max_fit = 8.
    nhi_min_fit = 2.

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
    
def plot_wedge_figure():
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
    maskbs = make_mask_2d(bstart=bstart, bstop=90, PS=True, NHIcut=NHIcut, NHImax=NHImax)  
    
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
        onesigmaerrs[0, 0], onesigmaerrs[1, 0] = load_bootstrap_data(vstart="1024", vstop="1024", bstart=bstart, NHIcut=NHIcut, NHImin=NHImin, NHImax=NHImax, Narrow=True, Nblocks=Nblocks, Nsamples=Nsamples, submonopole=submonopole, bstype=bstype)
    except:
        print("Could not load.")
        onesigmaerrs[0] = 1.0
        
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
            
    print(onesigmaerrs)

    plt.figure(figsize=(10, 5))
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
    plt.errorbar(thicknesses, delta857, yerr=onesigmaerrs, fmt='o', color='teal', ecolor='lightgray', elinewidth=3, capsize=0);
    plt.xlabel('$\mathrm{Channel}$ $\mathrm{Width}$ [$\mathrm{km/s}$]', size=15)
    plt.ylabel('$\Delta$ $\mathrm{I}_{857}$ [$\mathrm{MJy}$/Sr]', size=15)
    offhi857 = 54
    maxoffhi857 = 59
    minoffhi857 = 49
    #plt.axhspan(minoffhi857, maxoffhi857, alpha=0.1, color='C1')

    plt.axhline(0., linestyle='dashed', color='black')
    thinlim = 0.184*4*4 # width used in PRL and in the data Lazarian used

    maxx=np.max(thicknesses)
    minx= 0.0

    plt.axvline(thinlim, linestyle='dashed', color="gray")
    plt.fill([thinlim, maxx,maxx], [0, 0, np.max(delta857)], color='C2', alpha=0.1)
    #plt.ylim(-0.1, 2)
    plt.ylim(-0.1, 0.75)
    plt.xlim(minx, maxx)
    
    l2 = np.array((thinlim, 0))

    # Rotate angle
    angle = (np.max(delta857))/(maxx-thinlim)
    trans_angle = plt.gca().transData.transform_angles(np.array((angle,)),
                                                       l2.reshape((1, 2)))[0]

    # Plot text
    #plt.text(l2[0], l2[1], 'text rotated correctly', fontsize=16,
    #               rotation=trans_angle, rotation_mode='anchor')


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
    ##vels=["1020", "1021", "1022", "1023", "1024", "1025", "1026", "1027", "1028"]
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
        