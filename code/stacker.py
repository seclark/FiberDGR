from __future__ import division, print_function
import numpy as np
from astropy.io import fits
import time
import os.path

class HyperCube():
    """
    hyper cube class, possible dimensions (ra x dec x vel x theta)
    """
    def __init__(self, nx=101, ny=101, nvel=21, ntheta=165):
        self.nx = nx
        self.ny = ny
        self.nvel = nvel
        self.ntheta = ntheta
        self.singlecube = singlecube
        
        self.cubehalfx = np.int(np.floor(nx/2.0))
        self.cubehalfy = np.int(np.floor(ny/2.0))
        
        self.hypercube = np.zeros((self.ny, self.nx, self.nvel, self.ntheta), np.float_)
        self.weights_hypercube = np.zeros((self.ny, self.nx, self.nvel, self.ntheta), np.float_)
        

def load_2d_data(datatype="NHI90"):
    """
    Load two-dimensional data to be stacked
    """
    nhi_data_root = "/disks/jansky/a/users/goldston/zheng/151019_NHImaps_SRcorr/data/GNHImaps_SRCORR_final/NHImaps/"
    planck_data_root = "/disks/jansky/a/users/goldston/susan/Planck/"
    
    fn_dict = {"NHI90": nhi_data_root+"GALFA-HI_NHISRCORR_VLSR-90+90kms.fits",
               "NHI400": nhi_data_root+"GNHImaps_SRCORR_final/NHImaps/GALFA-HI_NHISRCORR_VLSR-400+400kms.fits",
               "Rad": planck_data_root+"COM_CompMap_Dust-GNILC-Radiance_2048_R2.00_ONGALFAHI.fits",
               "P857": planck_data_root+"HFI_SkyMap_857_2048_R2.02_full_ONGALFAHI.fits",
               "COM545": planck_data_root+"COM_CompMap_Dust-GNILC-F545_2048_R2.00_ONGALFAHI.fits",
               "Halpha": planck_data_root+"Halpha_finkbeiner03_proj_on_DR2.fits"
              }
    
    fn = fn_dict[datatype]
    data_2d = fits.getdata(fn)
    
    return data_2d

def load_lats():
    """
    load a map of all b values in GALFA-HI footprint 
    """
    bees = np.load("all_galactic_latitudes_galfanhi.npy")
    
    return bees

def prep_stack_on_data(stackon_data, absbcut=False, bcut=[-90, 90], zcut=[0.89, 0.91], biastest=False, verbose=False):
    
    # set bcut, if any -- this means that anything outside of the bcut won't be stacked on. but anything inside window will still be stacked.
    bstart = bcut[0]
    bstop = bcut[1]
    if absbcut:
        print("Tabulating data from |b|>{} to |b|<{}".format(bstart, bstop))
    else:
        print("Tabulating data from b={} to b={}".format(bstart, bstop))

    bees = load_lats()
    if absbcut:
        stackon_data[np.where(np.abs(bees) < bstart)] = 0 
        stackon_data[np.where(np.abs(bees) > bstop)] = 0
    else:
        stackon_data[np.where(bees < bstart)] = 0 
        stackon_data[np.where(bees > bstop)] = 0
    
    # If cutting on RHT intensity
    if biastest is True:
        zstart = zcut[0]
        zstop = zcut[1]
        print("Tabulating data from z={} to z={}".format(zstart, zstop))
        
        # cut based on RHT intensity. Need to subtract threshold! (Because RHT data is intensity over given thresh)
        thresh = 0.7
        print("before z cut, npix = {}".format(len(np.nonzero(velthet)[0])))
        stackon_data[np.where(velthet < (zstart - thresh))] = 0 
        stackon_data[np.where(velthet > (zstop - thresh))] = 0
        print("after z cut, npix = {}".format(len(np.nonzero(velthet)[0])))
    
    # Stack on all nonzero pixels
    nonzeroyx = np.nonzero(stackon_data)
    nonzeroy = nonzeroyx[0]
    nonzerox = nonzeroyx[1]

    if verbose:
        print("There are {} nonzero indices in this map".format(len(nonzerox)))

    return nonzeroy, nonzerox
    

def stack_slicedata(stackthese_data, stackon_data, nonzeroy, nonzerox, centerweight=True, verbose=False, cubenx=101, cubeny=101):
    """
    stack data
    """

    # shape of data to be stacked on
    maxny, maxnx = stackon_data.shape
    
    # square to be stacked into
    stackslice = np.zeros((ny, nx), np.float_)
    cubehalfx = np.int(np.floor(cubenx/2.0))
    cubehalfy = np.int(np.floor(cubeny/2.0))
    
    # Step through nonzero pixels
    for (_y, _x) in zip(nonzeroy, nonzerox):
        
        # the RHT value of the center pixel is the weight
        if biastest is True:
            if centerweight:
                centerval = stackon_data[_y, _x] # weight the biastest as well
            else:
                centerval = 1.0
        else:
            centerval = velthet[_y, _x]
            
            startx = np.int(max(0, _x - cubehalfx))
            stopx = np.int(min(maxnx, _x + cubehalfx + 1))
            starty = np.int(max(0, _y - cubehalfy))
            stopy = np.int(min(maxny, _y + cubehalfy + 1))
            
            RAedgeflag = False
            
            # LH side RA wrap case
            if (_x - cubehalfx) < 0:
                # deal with RA wrap
                RAedgeflag = True
                startx1 = 0
                stopx1 = _x + cubehalfx + 1 # maybe +1
                startx2 = maxnx - (cubehalfx - _x)
                stopx2 = maxnx
                
                # also, partial placement into hypercube
                smallstartx1 = 0
                smallstartx2 = cubehalfx - _x
              
            # RH side RA wrap case  
            if (_x + cubehalfx + 1) > maxnx:
                RAedgeflag = True
                startx1 = 0
                stopx1 = cubehalfx + 1 - maxnx + _x
                startx2 = _x - cubehalfx
                stopx2 = maxnx
                
                # partial placement into hypercube
                smallstartx2 = maxnx - (_x - cubehalfx)
                smallstartx1 = 0
            
            
            # Are you sitting on the edge in DEC?
            if (_y - cubehalfy) < 0:
                smallstarty = cubehalfy - _y 
                if verbose:
                    print("DEC < smallest DEC")
            else:
                smallstarty = 0
            
            if (_y + cubehalfy) > maxny:
                smallstopy = cubehalfy*2 - (_y + cubehalfy - maxny)
                if verbose:
                    print("DEC > largest DEC")
            else:
                smallstopy = cubehalfy*2 + 1
            
            # set into cubes, dealing w/ RA wrap
            if RAedgeflag:
                if weightsslice:
                    stackslice[smallstarty:smallstopy, smallstartx1:smallstartx2] += centerval 
                else:
                    # stack 2d data into hypercube
                    stackslice[smallstarty:smallstopy, smallstartx2:] += centerval * stackthese_data[starty:stopy, startx1:stopx1]
                    stackslice[smallstarty:smallstopy, smallstartx1:smallstartx2] += centerval * stackthese_data[starty:stopy, startx2:maxnx]
                
            # otherwise, proceed
            else:
                if weightsslice:
                    stackslice[smallstarty:smallstopy, :] += centerval 
                else:
                    stackslice[smallstarty:smallstopy, :] += centerval * stackthese_data[starty:stopy, startx:stopx]             
        
        return stackslice

def get_vel_theta_slice(vel_i, theta_i):
    velrangestr = galfa_vel_helpers.all_rht_velstrs[vel_i]
    
    # load vel, theta slice
    velthet_root = "/disks/jansky/a/users/goldston/susan/Wide_maps/single_theta_maps/"+velrangestr+"/"
    velthet_fn = velthet_root + "GALFA_HI_W_"+velrangestr+"_newhdr_SRcorr_w75_s15_t70_theta_"+str(theta_i)+".fits"
    velthet = fits.getdata(velthet_fn)
    
    return velthet
        
def get_slice_fn(v, thet, cubetype="nhi", biastest=False, centerweight=True, absbcut=True, bstart=30, bstop=90, zstart=0.7, zstop=1.0):
    
    if absbcut:
        absbcut_str = "absb_"
    else:
        absbcut_str = ""
    
    if centerweight:
        centervalstr = "_centerw"
    else:
        centervalstr = ""
    
    if biastest is False:
        slice_fn = "temp_hcube_slices/hypercube_{}_v{}_t{}_{}bstart_{}_bstop_{}{}.npy".format(cubetype, _v, _thet, absbcut_str, bstart, bstop, centervalstr)
        
    if biastest is True:
        slice_fn = "temp_hcube_slices/biastest_zcut/hypercube_{}_v{}_t{}_{}bstart_{}_bstop_{}_zstart_{}_zstop_{}{}.npy".format(cubetype, _v, _thet, absbcut_str, bstart, bstop, zstart, zstop, centervalstr)
    
    return slice_fn




biastest=False
centerweight=True
bstart=30
bstop=90
absbcut=True

if biastest is True:
    zstart=0.91
    zstop=0.94
else:
    zstart = 0.7
    zstop = 1.0
    
# all desired data to be stacked
datatypelist = ["NHI90", "NHI400", "Rad", "P857", "COM545", "Halpha"]
    
    
for _v in [9]: # of 21
    print("running velocity {}".format(_v))
    for _thet in np.arange(165): # of 165
    
        slice_fn = get_slice_fn(_v, _thet, cubetype="nhi", biastest=biastest, centerweight=centerweight, absbcut=absbcut, bstart=bstart, bstop=bstop, zstart=zstart, zstop=zstop)
    
        if os.path.isfile(slice_fn):
            print("v {}, t {}, bstart {}, bstop {}, absbcut {}, zstart {}, zstop {} already exists".format(_v, _thet, bstart, bstop, absbcut, zstart, zstop))
            
        else:
            time0 = time.time()
            
            # find data to stack on
            velthet = get_vel_theta_slice(_v, _thet)
            nonzeroy, nonzerox = prep_stack_on_data(velthet, absbcut=absbcut, bcut=[bstart, bstop], zcut=[zstart, zstop], biastest=biastest, verbose=False)
            
            # stack data
            for _datatype in datatypelist:
                stackthese_data = load_2d_data(datatype=_datatype)
                stackslice = stack_slicedata(stackthese_data, velthet, nonzeroy, nonzerox, centerweight=centerweight, verbose=False, weightsslice=False)
                slice_fn = get_slice_fn(_v, _thet, cubetype=_datatype, biastest=biastest, centerweight=centerweight, absbcut=absbcut, bstart=bstart, bstop=bstop, zstart=zstart, zstop=zstop)
                np.save(slice_fn, stackslice)
            
            weightslice = stack_slicedata(stackthese_data, velthet, nonzeroy, nonzerox, centerweight=centerweight, verbose=False, weightsslice=True)
            weight_slice_fn = get_slice_fn(_v, _thet, cubetype="weights", biastest=biastest, centerweight=centerweight, absbcut=absbcut, bstart=bstart, bstop=bstop, zstart=zstart, zstop=zstop)
            np.save(weight_slice_fn, weightslice)
            
            time1 = time.time()
        
            print("finished with velocity {} of 20, thet {} of 164. Took {} min.".format(_v, _thet, (time1-time0)/60.))
            
            

    
    
