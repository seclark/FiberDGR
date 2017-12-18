from __future__ import division, print_function
import numpy as np
from astropy.io import fits

import sys 
sys.path.insert(0, '../GalfaCuber/code')
import galfa_vel_helpers

class HyperCube():
    """
    hyper cube class, dimensions (ra x dec x vel x theta)
    """
    def __init__(self, nx=101, ny=101, nvel=21, ntheta=165):
        self.nx = nx
        self.ny = ny
        self.nvel = nvel
        self.ntheta = ntheta
        
        self.cubehalfx = np.int(np.floor(nx/2.0))
        self.cubehalfy = np.int(np.floor(ny/2.0))
        
        self.hypercube = np.zeros((ny, nx, nvel, ntheta), np.float_)
        
    def load_2d_data(self, datatype="nhi"):
        
        if datatype == "nhi":
            self.twoddata_fn = "/disks/jansky/a/users/goldston/zheng/151019_NHImaps_SRcorr/data/GNHImaps_SRCORR_final/NHImaps/GALFA-HI_NHISRCORR_VLSR-90+90kms.fits"
        
        self.twoddata = fits.getdata(self.twoddata_fn)
        
    def tabulate_per_vel_theta(self, vel_i=0, theta_i=0, verbose=False):
        """
        for a given vel, theta slice, step through and record data
        """
        
        velrangestr = galfa_vel_helpers.all_rht_velstrs[vel_i]
        
        # load vel, theta slice
        velthet_root = "/disks/jansky/a/users/goldston/susan/Wide_maps/single_theta_maps/"+velrangestr+"/"
        velthet_fn = velthet_root + "GALFA_HI_W_"+velrangestr+"_newhdr_SRcorr_w75_s15_t70_theta_"+str(theta_i)+".fits"
        velthet = fits.getdata(velthet_fn)
        
        self.maxny, self.maxnx = velthet.shape
        
        nonzeroyx = np.nonzero(velthet)
        nonzeroy = nonzeroyx[0]
        nonzerox = nonzeroyx[1]
        
        if verbose:
            print("There are {} nonzero indices in this map".format(len(nonzerox)))
        
        for (_y, _x) in zip(nonzeroy, nonzerox):
            centerval = velthet[_y, _x]
            
            print(_x, self.cubehalfx)
            self.startx = np.int(max(0, _x - self.cubehalfx))
            self.stopx = np.int(min(self.maxnx, _x + self.cubehalfx + 1))
            self.starty = np.int(max(0, _y - self.cubehalfy))
            self.stopy = np.int(min(self.maxny, _y + self.cubehalfy + 1))
            
            self.RAedgeflag = False
            # deal with edges
            if self.startx == 0:
                # deal with RA wrap
                self.RAedgeflag = True
                self.startx1 = 0
                self.stopx1 = _x + self.cubehalfx # maybe +1
                self.startx2 = self.maxnx - (_x - self.cubehalfx)
                self.stopx2 = self.maxnx
                
                # also, partial placement into hypercube
                self.smallstartx1 = 0
                self.smallstartx2 = self.cubehalfx - _x
                
            if self.stopx == self.maxnx:
                self.RAedgeflag = True
                self.startx1 = 0
                self.stopx1 = self.cubehalfx - self.maxnx + _x
                self.startx2 = _x - self.cubehalfx
                self.stopx2 = self.maxnx
                
                # partial placement into hypercube
                self.smallstartx1 = self.maxnx - (_x - self.cubehalfx)
                self.smallstartx2 = 0
            
            
            # Are you sitting on the edge in DEC?
            if (_y - self.cubehalfy) < 0:
                self.smallstarty = self.cubehalfy - _y
            else:
                self.smallstarty = 0
            
            if (_y + self.cubehalfy) > self.maxny:
                self.smallstopy = self.cubehalfy*2 + 1 - (_y + self.cubehalfy - self.maxny)
            else:
                self.smallstopy = self.cubehalfy*2 + 1
            
            print(self.smallstarty, self.smallstopy)
            
            if self.RAedgeflag:
                self.hypercube[self.smallstarty:self.smallstopy, self.smallstartx1:self.smallstartx2, vel_i, theta_i] = centerval*self.twoddata[self.starty:self.stopy, self.startx1:self.stopx]
                self.hypercube[self.smallstarty:self.smallstopy, self.smallstartx2:, vel_i, theta_i] = centerval*self.twoddata[self.starty:self.stopy, self.startx2:self.maxnx]
            
            else:
                self.hypercube[self.smallstarty:self.smallstopy, :, vel_i, theta_i] = centerval*self.twoddata[self.starty:self.stopy, self.startx:self.stopx]
            
#test
hcube = HyperCube()
hcube.load_2d_data(datatype="nhi")
hcube.tabulate_per_vel_theta(vel_i=0, theta_i=0, verbose=True)
