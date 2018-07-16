from __future__ import division, print_function
import numpy as np
from astropy.io import fits
import time
import os.path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys 
sys.path.insert(0, '../GalfaCuber/code')
import galfa_vel_helpers

sys.path.insert(0, '../FITSHandling/code')
import cutouts

#print("python version is {}".format(sys.version))

class HyperCube():
    """
    hyper cube class, dimensions (ra x dec x vel x theta)
    """
    def __init__(self, nx=101, ny=101, nvel=21, ntheta=165, singlecube=False):
        self.nx = nx
        self.ny = ny
        self.nvel = nvel
        self.ntheta = ntheta
        self.singlecube = singlecube
        
        self.cubehalfx = np.int(np.floor(nx/2.0))
        self.cubehalfy = np.int(np.floor(ny/2.0))
        
        if self.singlecube:
            self.hypercube = np.zeros((self.ny, self.nx, self.nvel, self.ntheta), np.float_)
        self.weights_hypercube = np.zeros((self.ny, self.nx, self.nvel, self.ntheta), np.float_)
        
    def load_2d_data(self, datatype="nhi", local=False):
        
        if datatype == "nhi":
            if local:
                self.twoddata_fn = "/Users/susanclark/Dropbox/NHImaps/GALFA-HI_NHImap_SRcorr_VLSR-090+0090kms.fits"
            else:
                self.twoddata_fn = "/disks/jansky/a/users/goldston/zheng/151019_NHImaps_SRcorr/data/GNHImaps_SRCORR_final/NHImaps/GALFA-HI_NHISRCORR_VLSR-90+90kms.fits"
        
        self.twoddata = fits.getdata(self.twoddata_fn)
        
    def load_nhi_rad_857(self, local=False):
        if self.singlecube is True:
            print("WARNING: only processing a single 2d data cube")
        else:
            self.hypercube_nhi = np.zeros((self.ny, self.nx, self.nvel, self.ntheta), np.float_)
            self.hypercube_400 = np.zeros((self.ny, self.nx, self.nvel, self.ntheta), np.float_)
            self.hypercube_rad = np.zeros((self.ny, self.nx, self.nvel, self.ntheta), np.float_)
            self.hypercube_857 = np.zeros((self.ny, self.nx, self.nvel, self.ntheta), np.float_)
            
        self.nhi_fn = "/disks/jansky/a/users/goldston/zheng/151019_NHImaps_SRcorr/data/GNHImaps_SRCORR_final/NHImaps/GALFA-HI_NHISRCORR_VLSR-90+90kms.fits"
        self.nhi_400_fn = "/disks/jansky/a/users/goldston/zheng/151019_NHImaps_SRcorr/data/GNHImaps_SRCORR_final/NHImaps/GALFA-HI_NHISRCORR_VLSR-400+400kms.fits"
        self.rad_fn = "/disks/jansky/a/users/goldston/susan/Planck/COM_CompMap_Dust-GNILC-Radiance_2048_R2.00_ONGALFAHI.fits"
        self.P857_fn = "/disks/jansky/a/users/goldston/susan/Planck/HFI_SkyMap_857_2048_R2.02_full_ONGALFAHI.fits"
        
        self.nhi = fits.getdata(self.nhi_fn)
        self.nhi_400 = fits.getdata(self.nhi_400_fn)
        self.radiance = fits.getdata(self.rad_fn)
        self.Planck857 = fits.getdata(self.P857_fn)
    
    def load_mask(self, mask=None):
        """
        load generic mask or set data == NaN to 0 in mask
        """
        if mask is not None:
            self.mask = np.isfinite(self.twoddata)
            
    def load_lats(self):
        """
        create or load a map of all b values in GALFA-HI footprint 
        """
        if os.path.isfile("all_galactic_latitudes_galfanhi.npy"):
            self.bees = np.load("all_galactic_latitudes_galfanhi.npy")
        
        else:
            nhi_hdr = fits.getheader(self.nhi_fn)
            w_galfa = cutouts.make_wcs(nhi_hdr)
            xs = np.arange(0, nhi_hdr["NAXIS1"])
            ys = np.arange(0, nhi_hdr["NAXIS2"])
            X, Y = np.meshgrid(xs, ys)
            ras, decs = cutouts.xys_to_radec(X.ravel(), Y.ravel(), w_galfa)
            ells, bees = cutouts.radecs_to_lb(ras, decs)

            self.ells = ells.reshape(X.shape)
            self.bees = bees.reshape(X.shape)
        
            np.save("all_galactic_latitudes_galfanhi.npy", self.bees)
        
    def tabulate_per_vel_theta(self, vel_i=0, theta_i=0, verbose=False, absbcut=False, bcut=[-90, 90], zcut=[0.89, 0.91], biastest=False, centerweight=True):
        """
        for a given vel, theta slice, step through and record data
        """
        
        velrangestr = galfa_vel_helpers.all_rht_velstrs[vel_i]
        
        # load vel, theta slice
        velthet_root = "/disks/jansky/a/users/goldston/susan/Wide_maps/single_theta_maps/"+velrangestr+"/"
        velthet_fn = velthet_root + "GALFA_HI_W_"+velrangestr+"_newhdr_SRcorr_w75_s15_t70_theta_"+str(theta_i)+".fits"
        velthet = fits.getdata(velthet_fn)
        
        self.maxny, self.maxnx = velthet.shape
        
        # set bcut, if any -- this means that anything outside of the bcut won't be stacked on. but anything inside window will still be stacked.
        self.bstart = bcut[0]
        self.bstop = bcut[1]
        if absbcut:
            print("Tabulating data from |b|>{} to |b|<{}".format(self.bstart, self.bstop))
        else:
            print("Tabulating data from b={} to b={}".format(self.bstart, self.bstop))
        
        if biastest is True:
            self.zstart = zcut[0]
            self.zstop = zcut[1]
            print("Tabulating data from z={} to z={}".format(self.zstart, self.zstop))
            
            # cut based on RHT intensity. Need to subtract threshold!!! (Because RHT data is intensity over given thresh)
            thresh = 0.7
            print("before z cut, npix = {}".format(len(np.nonzero(velthet)[0])))
            velthet[np.where(velthet < (self.zstart - thresh))] = 0 
            velthet[np.where(velthet > (self.zstop - thresh))] = 0
            print("after z cut, npix = {}".format(len(np.nonzero(velthet)[0])))
        
        self.load_lats()
        #print("before b cut, npix = {}".format(len(np.nonzero(velthet)[0])))
        if absbcut:
            velthet[np.where(np.abs(self.bees) < self.bstart)] = 0 
            velthet[np.where(np.abs(self.bees) > self.bstop)] = 0
        else:
            velthet[np.where(self.bees < self.bstart)] = 0 
            velthet[np.where(self.bees > self.bstop)] = 0
        #print("after b cut, npix = {}".format(len(np.nonzero(velthet)[0])))
        nonzeroyx = np.nonzero(velthet)
        nonzeroy = nonzeroyx[0]
        nonzerox = nonzeroyx[1]
        
        if verbose:
            print("There are {} nonzero indices in this map".format(len(nonzerox)))
        
        # Step through nonzero pixels
        for (_y, _x) in zip(nonzeroy, nonzerox):
            
            # the RHT value of the center pixel is the weight
            if biastest is True:
                if centerweight:
                    centerval = velthet[_y, _x] # weight the biastest as well
                else:
                    centerval = 1.0
            else:
                centerval = velthet[_y, _x]
            
            self.startx = np.int(max(0, _x - self.cubehalfx))
            self.stopx = np.int(min(self.maxnx, _x + self.cubehalfx + 1))
            self.starty = np.int(max(0, _y - self.cubehalfy))
            self.stopy = np.int(min(self.maxny, _y + self.cubehalfy + 1))
            
            self.RAedgeflag = False
            
            # LH side RA wrap case
            if (_x - self.cubehalfx) < 0:
                # deal with RA wrap
                self.RAedgeflag = True
                self.startx1 = 0
                self.stopx1 = _x + self.cubehalfx + 1 # maybe +1
                self.startx2 = self.maxnx - (self.cubehalfx - _x)#self.maxnx - _x - self.cubehalfx
                self.stopx2 = self.maxnx
                
                # also, partial placement into hypercube
                self.smallstartx1 = 0
                self.smallstartx2 = self.cubehalfx - _x
              
            # RH side RA wrap case  
            if (_x + self.cubehalfx + 1) > self.maxnx:
                self.RAedgeflag = True
                self.startx1 = 0
                self.stopx1 = self.cubehalfx + 1 - self.maxnx + _x
                self.startx2 = _x - self.cubehalfx
                self.stopx2 = self.maxnx
                
                # partial placement into hypercube
                self.smallstartx2 = self.maxnx - (_x - self.cubehalfx)
                self.smallstartx1 = 0
            
            
            # Are you sitting on the edge in DEC?
            if (_y - self.cubehalfy) < 0:
                self.smallstarty = self.cubehalfy - _y 
                if verbose:
                    print("DEC < smallest DEC")
            else:
                self.smallstarty = 0
            
            if (_y + self.cubehalfy) > self.maxny:
                self.smallstopy = self.cubehalfy*2 - (_y + self.cubehalfy - self.maxny)
                if verbose:
                    print("DEC > largest DEC")
            else:
                self.smallstopy = self.cubehalfy*2 + 1
            
            # set into cubes, dealing w/ RA wrap
            if self.RAedgeflag:
                
                # single 2d data into hypercube
                if self.singlecube:
                    self.hypercube[self.smallstarty:self.smallstopy, self.smallstartx2:, vel_i, theta_i] += centerval*self.twoddata[self.starty:self.stopy, self.startx1:self.stopx1]
                    self.hypercube[self.smallstarty:self.smallstopy, self.smallstartx1:self.smallstartx2, vel_i, theta_i] += centerval*self.twoddata[self.starty:self.stopy, self.startx2:self.maxnx]
                
                # nhi, radiance, 857 into separate hypercubes
                else:
                    self.hypercube_nhi[self.smallstarty:self.smallstopy, self.smallstartx2:, vel_i, theta_i] += centerval*self.nhi[self.starty:self.stopy, self.startx1:self.stopx1]
                    self.hypercube_nhi[self.smallstarty:self.smallstopy, self.smallstartx1:self.smallstartx2, vel_i, theta_i] += centerval*self.nhi[self.starty:self.stopy, self.startx2:self.maxnx]
                    self.hypercube_rad[self.smallstarty:self.smallstopy, self.smallstartx2:, vel_i, theta_i] += centerval*self.radiance[self.starty:self.stopy, self.startx1:self.stopx1]
                    self.hypercube_rad[self.smallstarty:self.smallstopy, self.smallstartx1:self.smallstartx2, vel_i, theta_i] += centerval*self.radiance[self.starty:self.stopy, self.startx2:self.maxnx]
                    self.hypercube_857[self.smallstarty:self.smallstopy, self.smallstartx2:, vel_i, theta_i] += centerval*self.Planck857[self.starty:self.stopy, self.startx1:self.stopx1]
                    self.hypercube_857[self.smallstarty:self.smallstopy, self.smallstartx1:self.smallstartx2, vel_i, theta_i] += centerval*self.Planck857[self.starty:self.stopy, self.startx2:self.maxnx]
                    self.hypercube_400[self.smallstarty:self.smallstopy, self.smallstartx2:, vel_i, theta_i] += centerval*self.nhi_400[self.starty:self.stopy, self.startx1:self.stopx1]
                    self.hypercube_400[self.smallstarty:self.smallstopy, self.smallstartx1:self.smallstartx2, vel_i, theta_i] += centerval*self.nhi_400[self.starty:self.stopy, self.startx2:self.maxnx]
            
                # save weights
                self.weights_hypercube[self.smallstarty:self.smallstopy, self.smallstartx1:self.smallstartx2, vel_i, theta_i] += centerval
            
            # otherwise, proceed
            else:
                if self.singlecube:
                    self.hypercube[self.smallstarty:self.smallstopy, :, vel_i, theta_i] += centerval*self.twoddata[self.starty:self.stopy, self.startx:self.stopx]
                else:
                    self.hypercube_nhi[self.smallstarty:self.smallstopy, :, vel_i, theta_i] += centerval*self.nhi[self.starty:self.stopy, self.startx:self.stopx]
                    self.hypercube_rad[self.smallstarty:self.smallstopy, :, vel_i, theta_i] += centerval*self.radiance[self.starty:self.stopy, self.startx:self.stopx]
                    self.hypercube_857[self.smallstarty:self.smallstopy, :, vel_i, theta_i] += centerval*self.Planck857[self.starty:self.stopy, self.startx:self.stopx]
                    self.hypercube_400[self.smallstarty:self.smallstopy, :, vel_i, theta_i] += centerval*self.nhi_400[self.starty:self.stopy, self.startx:self.stopx]
                    
                self.weights_hypercube[self.smallstarty:self.smallstopy, :, vel_i, theta_i] += centerval
    
    def assemble_hcubes(self, bcut=[-90, 90], zcut=[0.89, 0.91], absbcut=False, biastest=False, minicube=False, centerweight=True):
        if minicube:
            all_vs = [9, 10, 11]
        else:
            all_vs = np.arange(21)
            
        nvel = len(all_vs)
        
        self.hypercube_nhi = np.zeros((self.ny, self.nx, nvel, self.ntheta), np.float_)
        self.hypercube_rad = np.zeros((self.ny, self.nx, nvel, self.ntheta), np.float_)
        self.hypercube_857 = np.zeros((self.ny, self.nx, nvel, self.ntheta), np.float_)
        self.hypercube_400 = np.zeros((self.ny, self.nx, nvel, self.ntheta), np.float_)
        self.weights_hypercube = np.zeros((self.ny, self.nx, nvel, self.ntheta), np.float_)
        
        if biastest:
            print("assembling hcubes for b = {} to {}, z = {} to {}, absbcut {}".format(bcut[0], bcut[1], zcut[0], zcut[1], absbcut))
        else:
            print("assembling hcubes for b = {} to {}, absbcut {}".format(bcut[0], bcut[1], zcut[0], zcut[1], absbcut))
        
        missing_vt_pair = 0
        missing_ts_per_v = np.zeros(len(all_vs))
        
        if absbcut:
            absbcut_str = "absb_"
        else:
            absbcut_str = ""
            
        if biastest:    
            if centerweight:
                centervalstr = "centerw"
            else:
                centervalstr = ""
        
        for i_v, _v in enumerate(all_vs): 
            for _thet in np.arange(165): 
                
                if biastest is False:
                    fn_nhi = "temp_hcube_slices/hypercube_nhi_v{}_t{}_{}bstart_{}_bstop_{}.npy".format(_v, _thet, absbcut_str, bcut[0], bcut[1])
                    fn_rad = "temp_hcube_slices/hypercube_rad_v{}_t{}_{}bstart_{}_bstop_{}.npy".format(_v, _thet, absbcut_str, bcut[0], bcut[1])
                    fn_857 = "temp_hcube_slices/hypercube_857_v{}_t{}_{}bstart_{}_bstop_{}.npy".format(_v, _thet, absbcut_str, bcut[0], bcut[1])
                    fn_400 = "temp_hcube_slices/hypercube_nhi_400_v{}_t{}_{}bstart_{}_bstop_{}.npy".format(_v, _thet, absbcut_str, bcut[0], bcut[1])
                    fn_weights = "temp_hcube_slices/hypercube_weights_v{}_t{}_{}bstart_{}_bstop_{}.npy".format(_v, _thet, absbcut_str, bcut[0], bcut[1])
                else: 
                    fn_nhi = "temp_hcube_slices/biastest_zcut/hypercube_nhi_v{}_t{}_{}bstart_{}_bstop_{}_zstart_{}_zstop_{}_{}.npy".format(_v, _thet, absbcut_str, bcut[0], bcut[1], zcut[0], zcut[1], centervalstr)
                    fn_rad = "temp_hcube_slices/biastest_zcut/hypercube_rad_v{}_t{}_{}bstart_{}_bstop_{}_zstart_{}_zstop_{}_{}.npy".format(_v, _thet, absbcut_str, bcut[0], bcut[1], zcut[0], zcut[1], centervalstr)
                    fn_857 = "temp_hcube_slices/biastest_zcut/hypercube_857_v{}_t{}_{}bstart_{}_bstop_{}_zstart_{}_zstop_{}_{}.npy".format(_v, _thet, absbcut_str, bcut[0], bcut[1], zcut[0], zcut[1], centervalstr)
                    fn_400 = "temp_hcube_slices/biastest_zcut/hypercube_nhi_400_v{}_t{}_{}bstart_{}_bstop_{}_zstart_{}_zstop_{}_{}.npy".format(_v, _thet, absbcut_str, bcut[0], bcut[1], zcut[0], zcut[1], centervalstr)
                    fn_weights = "temp_hcube_slices/biastest_zcut/hypercube_weights_v{}_t{}_{}bstart_{}_bstop_{}_zstart_{}_zstop_{}_{}.npy".format(_v, _thet, absbcut_str, bcut[0], bcut[1], zcut[0], zcut[1], centervalstr)

                
                if os.path.isfile(fn_nhi):
                    self.hypercube_nhi[:, :, i_v, _thet] = np.load(fn_nhi)
                    self.hypercube_rad[:, :, i_v, _thet] = np.load(fn_rad)
                    self.hypercube_857[:, :, i_v, _thet] = np.load(fn_857)
                    self.hypercube_400[:, :, i_v, _thet] = np.load(fn_400)
                    self.weights_hypercube[:, :, i_v, _thet] = np.load(fn_weights)
                else:
                    missing_vt_pair += 1
                    missing_ts_per_v[i_v] += 1
                    
                    print("missing v = {}, t = {}".format(_v, _thet))
                    
        self.nmissing = missing_vt_pair
        print("Number of missing v, theta pairs is {} out of {}".format(missing_vt_pair, self.nvel*self.ntheta))
        print("Number of missing ts per v : {}".format(missing_ts_per_v))
        
    def make_movies(self, bstart=-90, bstop=90, zstart=0.89, zstop=0.91, movietype="nhi", biastest=False):
        
        dataroot = "data/"
        if biastest is True:
            if (movietype == "nhi") or (movietype == "857_nhi") or (movietype == "rad_nhi"):
                hcube_nhi = np.load(dataroot+"hypercube_nhi_bstart_{}_bstop_{}_zstart_{}_zstop_{}.npy".format(bstart, bstop, zstart, zstop))
            if (movietype == "rad") or (movietype == "rad_nhi"):
                hcube_rad = np.load(dataroot+"hypercube_rad_bstart_{}_bstop_{}_zstart_{}_zstop_{}.npy".format(bstart, bstop, zstart, zstop))
            if (movietype == "857") or (movietype == "857_nhi"):
                hcube_857 = np.load(dataroot+"hypercube_857_bstart_{}_bstop_{}_zstart_{}_zstop_{}.npy".format(bstart, bstop, zstart, zstop))
                
            if movietype == "857_nhi":
                print("loading DGR map")
                hcube_857_nhi = hcube_857/hcube_nhi
                
            if movietype == "rad_nhi":
                print("loading DGR map")
                hcube_rad_nhi = hcube_rad/hcube_nhi
        
        else:
            if (movietype == "nhi") or (movietype == "857_nhi") or (movietype == "rad_nhi"):
                hcube_nhi = np.load(dataroot+"hypercube_nhi_bstart_{}_bstop_{}.npy".format(bstart, bstop))
            if (movietype == "rad") or (movietype == "rad_nhi"):
                hcube_rad = np.load(dataroot+"hypercube_rad_bstart_{}_bstop_{}.npy".format(bstart, bstop))
            if (movietype == "857") or (movietype == "857_nhi"):
                hcube_857 = np.load(dataroot+"hypercube_857_bstart_{}_bstop_{}.npy".format(bstart, bstop))
                
            if movietype == "857_nhi":
                print("loading DGR map")
                hcube_857_nhi = hcube_857/hcube_nhi
            
            if movietype == "rad_nhi":
                print("loading DGR map")
                hcube_rad_nhi = hcube_rad/hcube_nhi
        
        for _thet in np.arange(165): # of 165
        
            fig = plt.figure(facecolor="white", figsize=(12,7))
        
            for _i, _v in enumerate(np.arange(21)):
                ax = fig.add_subplot(3, 7, _i + 1)
                if movietype == "nhi":
                    im = ax.imshow(hcube_nhi[:, :, _v, _thet])
                elif movietype == "rad":
                    im = ax.imshow(hcube_rad[:, :, _v, _thet])
                elif movietype == "857":
                    im = ax.imshow(hcube_857[:, :, _v, _thet])
                elif movietype == "857_nhi":
                    im = ax.imshow(hcube_857_nhi[:, :, _v, _thet])
                elif movietype == "rad_nhi":
                    im = ax.imshow(hcube_rad_nhi[:, :, _v, _thet])
                
                #divider = make_axes_locatable(ax)
                #cax = divider.append_axes("right", size="3%", pad=0.05)
                #cbar = plt.colorbar(im, cax=cax)  
                #cbar.outline.set_visible(False)
                
                ax.set_title("{}".format(_v))
                ax.set_xticks([])
                ax.set_yticks([])
            
            if movietype == "nhi":
                plt.suptitle("NHI, theta = {}".format(np.round(np.degrees((np.pi/165)*_thet)), 2))
                if biastest is True:
                    plt.savefig("figures/allvel_nhi_bstart_{}_bstop_{}_zstart_{}_zstop_{}_theta_{}.png".format(bstart, bstop, zstart, zstop, str(_thet).zfill(3)))
                else:
                    plt.savefig("figures/allvel_nhi_bstart_{}_bstop_{}_theta_{}.png".format(bstart, bstop, str(_thet).zfill(3)))
                plt.close()
            elif movietype == "rad":
                plt.suptitle("Radiance, theta = {}".format(np.round(np.degrees((np.pi/165)*_thet)), 2))
                if biastest is True:
                    plt.savefig("figures/allvel_rad_bstart_{}_bstop_{}_zstart_{}_zstop_{}_theta_{}.png".format(bstart, bstop, zstart, zstop, str(_thet).zfill(3)))
                else: 
                    plt.savefig("figures/allvel_rad_bstart_{}_bstop_{}_theta_{}.png".format(bstart, bstop, str(_thet).zfill(3)))
                plt.close()
            elif movietype == "857":
                plt.suptitle("857 GHz, theta = {}".format(np.round(np.degrees((np.pi/165)*_thet)), 2))
                if biastest is True:
                    plt.savefig("figures/allvel_857_bstart_{}_bstop_{}_zstart_{}_zstop_{}_theta_{}.png".format(bstart, bstop, zstart, zstop, str(_thet).zfill(3)))
                else:
                    plt.savefig("figures/allvel_857_bstart_{}_bstop_{}_theta_{}.png".format(bstart, bstop, str(_thet).zfill(3)))
                plt.close()
            elif movietype == "857_nhi":
                plt.suptitle("857 GHz / NHI, theta = {}".format(np.round(np.degrees((np.pi/165)*_thet)), 2))
                if biastest is True:
                    plt.savefig("figures/allvel_857_nhi_bstart_{}_bstop_{}_zstart_{}_zstop_{}_theta_{}.png".format(bstart, bstop, zstart, zstop, str(_thet).zfill(3)))
                else:
                    plt.savefig("figures/allvel_857_nhi_bstart_{}_bstop_{}_theta_{}.png".format(bstart, bstop, str(_thet).zfill(3)))
                plt.close()
            elif movietype == "rad_nhi":
                plt.suptitle("Radiance / NHI, theta = {}".format(np.round(np.degrees((np.pi/165)*_thet)), 2))
                if biastest is True:
                    plt.savefig("figures/allvel_rad_nhi_bstart_{}_bstop_{}_zstart_{}_zstop_{}_theta_{}.png".format(bstart, bstop, zstart, zstop, str(_thet).zfill(3)))
                else:
                    plt.savefig("figures/allvel_rad_nhi_bstart_{}_bstop_{}_theta_{}.png".format(bstart, bstop, str(_thet).zfill(3)))
                plt.close()
                
"""
# run for nhi, radiance, 857
hcube = HyperCube(singlecube=False)
hcube.load_nhi_rad_857(local=False)

biastest=True
centerweight=True
bstart=30#bstart=60 #bstart=80#0
bstop=90#bstop=70 #bstop=90#10
absbcut=True

if absbcut:
    absbcut_str = "absb_"
else:
    absbcut_str = ""

if biastest:    
    if centerweight:
        centervalstr = "centerw"
    else:
        centervalstr = ""

if biastest is True:
    zstart=0.97
    zstop=1.0#0.76#0.85
    
for _v in [9, 10, 11]: # of 21
    print("running velocity {}".format(_v))
    for _thet in np.arange(165): # of 165
    
        if biastest is True:
            outfn = "temp_hcube_slices/biastest_zcut/hypercube_nhi_v{}_t{}_{}bstart_{}_bstop_{}_zstart_{}_zstop_{}_{}.npy".format(_v, _thet, absbcut_str, bstart, bstop, zstart, zstop, centervalstr)
        else:
            outfn = "temp_hcube_slices/hypercube_nhi_v{}_t{}_{}bstart_{}_bstop_{}.npy".format(_v, _thet, absbcut_str, bstart, bstop)
        
        if os.path.isfile(outfn):
            if biastest is True:
                print("v {}, t {}, bstart {}, bstop {}, absbcut {}, zstart {}, zstop {} already exists".format(_v, _thet, bstart, bstop, absbcut, zstart, zstop))
            else:
                print("v {}, t {}, bstart {}, bstop {}, absbcut {} already exists".format(_v, _thet, bstart, bstop, absbcut))
        else:
            time0 = time.time()
            if biastest is True:
                hcube.tabulate_per_vel_theta(vel_i=_v, theta_i=_thet, verbose=False, bcut=[bstart, bstop], zcut=[zstart, zstop], biastest=biastest, absbcut=absbcut, centerweight=centerweight)
            if biastest is False:
                hcube.tabulate_per_vel_theta(vel_i=_v, theta_i=_thet, verbose=False, bcut=[bstart, bstop], biastest=biastest, absbcut=absbcut)
            time1 = time.time()
        
            print("finished with velocity {} of 20, thet {} of 164. Took {} min.".format(_v, _thet, (time1-time0)/60.))
            
            if biastest is False:
                np.save("temp_hcube_slices/hypercube_nhi_v{}_t{}_{}bstart_{}_bstop_{}.npy".format(_v, _thet, absbcut_str, hcube.bstart, hcube.bstop), hcube.hypercube_nhi[:, :, _v, _thet])
                np.save("temp_hcube_slices/hypercube_nhi_400_v{}_t{}_{}bstart_{}_bstop_{}.npy".format(_v, _thet, absbcut_str, hcube.bstart, hcube.bstop), hcube.hypercube_400[:, :, _v, _thet])
                np.save("temp_hcube_slices/hypercube_rad_v{}_t{}_{}bstart_{}_bstop_{}.npy".format(_v, _thet, absbcut_str, hcube.bstart, hcube.bstop), hcube.hypercube_rad[:, :, _v, _thet])
                np.save("temp_hcube_slices/hypercube_857_v{}_t{}_{}bstart_{}_bstop_{}.npy".format(_v, _thet, absbcut_str, hcube.bstart, hcube.bstop), hcube.hypercube_857[:, :, _v, _thet])
                np.save("temp_hcube_slices/hypercube_weights_v{}_t{}_{}bstart_{}_bstop_{}.npy".format(_v, _thet, absbcut_str, hcube.bstart, hcube.bstop), hcube.weights_hypercube[:, :, _v, _thet])

            if biastest is True:
                np.save("temp_hcube_slices/biastest_zcut/hypercube_nhi_v{}_t{}_{}bstart_{}_bstop_{}_zstart_{}_zstop_{}_{}.npy".format(_v, _thet, absbcut_str, hcube.bstart, hcube.bstop, hcube.zstart, hcube.zstop, centervalstr), hcube.hypercube_nhi[:, :, _v, _thet])
                np.save("temp_hcube_slices/biastest_zcut/hypercube_nhi_400_v{}_t{}_{}bstart_{}_bstop_{}_zstart_{}_zstop_{}_{}.npy".format(_v, _thet, absbcut_str, hcube.bstart, hcube.bstop, hcube.zstart, hcube.zstop, centervalstr), hcube.hypercube_400[:, :, _v, _thet])
                np.save("temp_hcube_slices/biastest_zcut/hypercube_rad_v{}_t{}_{}bstart_{}_bstop_{}_zstart_{}_zstop_{}_{}.npy".format(_v, _thet, absbcut_str, hcube.bstart, hcube.bstop, hcube.zstart, hcube.zstop, centervalstr), hcube.hypercube_rad[:, :, _v, _thet])
                np.save("temp_hcube_slices/biastest_zcut/hypercube_857_v{}_t{}_{}bstart_{}_bstop_{}_zstart_{}_zstop_{}_{}.npy".format(_v, _thet, absbcut_str, hcube.bstart, hcube.bstop, hcube.zstart, hcube.zstop, centervalstr), hcube.hypercube_857[:, :, _v, _thet])
                np.save("temp_hcube_slices/biastest_zcut/hypercube_weights_v{}_t{}_{}bstart_{}_bstop_{}_zstart_{}_zstop_{}_{}.npy".format(_v, _thet, absbcut_str, hcube.bstart, hcube.bstop, hcube.zstart, hcube.zstop, centervalstr), hcube.weights_hypercube[:, :, _v, _thet])

"""

# assemble cubes

bstart=30
bstop=90
zstart=0.76
zstop=0.79
minicube=True
absbcut=True
biastest=True
centerweight=True

if absbcut:
    absbcut_str = "absb_"
else:
    absbcut_str = ""
    
if biastest:
    biastest_str = "_zstart_{}_zstop_{}".format(zstart, zstop)
    if centerweight:
        centervalstr = "centerw"
    else:
        centervalstr = ""
else:
    biastest_str = ""
    

hcube = HyperCube(singlecube=False)
hcube.assemble_hcubes(bcut=[bstart, bstop], zcut=[zstart, zstop], biastest=biastest, minicube=minicube, absbcut=absbcut, centerweight=centerweight)

if hcube.nmissing == 0.0:
    if minicube:
        print("Missing no v, t pairs. Saving cubes.")
        np.save("hcubes/hypercube_mini_nhi_{}bstart_{}_bstop_{}{}_{}.npy".format(absbcut_str, bstart, bstop, biastest_str, centervalstr), hcube.hypercube_nhi)
        np.save("hcubes/hypercube_mini_nhi_400_{}bstart_{}_bstop_{}{}_{}.npy".format(absbcut_str, bstart, bstop, biastest_str, centervalstr), hcube.hypercube_nhi)
        np.save("hcubes/hypercube_mini_rad_{}bstart_{}_bstop_{}{}_{}.npy".format(absbcut_str, bstart, bstop, biastest_str, centervalstr), hcube.hypercube_rad)
        np.save("hcubes/hypercube_mini_857_{}bstart_{}_bstop_{}{}_{}.npy".format(absbcut_str, bstart, bstop, biastest_str, centervalstr), hcube.hypercube_857)
        np.save("hcubes/hypercube_mini_weights_{}bstart_{}_bstop_{}{}_{}.npy".format(absbcut_str, bstart, bstop, biastest_str, centervalstr), hcube.weights_hypercube)
    
    else:
        print("Missing no v, t pairs. Saving cubes.")
        np.save("hcubes/hypercube_nhi_{}bstart_{}_bstop_{}{}.npy".format(absbcut_str, bstart, bstop, biastest_str), hcube.hypercube_nhi)
        np.save("hcubes/hypercube_nhi_400_{}bstart_{}_bstop_{}{}.npy".format(absbcut_str, bstart, bstop, biastest_str), hcube.hypercube_nhi)
        np.save("hcubes/hypercube_rad_{}bstart_{}_bstop_{}{}.npy".format(absbcut_str, bstart, bstop, biastest_str), hcube.hypercube_rad)
        np.save("hcubes/hypercube_857_{}bstart_{}_bstop_{}{}.npy".format(absbcut_str, bstart, bstop, biastest_str), hcube.hypercube_857)
        np.save("hcubes/hypercube_weights_{}bstart_{}_bstop_{}{}.npy".format(absbcut_str, bstart, bstop, biastest_str), hcube.weights_hypercube)


#hcube = HyperCube(singlecube=False)
#hcube.make_movies(bstart=70, bstop=80, movietype="rad_nhi", biastest=False)


