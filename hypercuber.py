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
        
        maxny, maxnx = velthet.shape
        
        nonzeroyx = np.nonzero(velthet)
        nonzeroy = nonzeroyx[0]
        nonzerox = nonzeroyx[1]
        
        if verbose:
            print("There are {} nonzero indices in this map".format(len(nonzerox)))
        
        for (_y, _x) in zip(nonzeroy, nonzerox):
            centerval = velthet[_y, _x]
            
            print(_x, self.cubehalfx)
            startx = np.int(max(0, _x - self.cubehalfx))
            stopx = np.int(min(maxnx, _x + self.cubehalfx))
            starty = np.int(max(0, _y - self.cubehalfy))
            stopy = np.int(min(maxny, _y + self.cubehalfy))
            
            # deal with edges
            if startx == 0:
                # deal with RA wrap
                self.edgeflag = True
                startx1 = 0
                stopx1 = _x + self.cubehalfx # maybe +1
                startx2 = maxnx - (_x - self.cubehalfx)
                stopx2 = maxnx
                
                # also, partial placement into hypercube
                smallstartx1 = 0
                smallstartx2 = self.cubehalfx - _x
                
            if stopx == maxnx:
                self.edgeflag = True
                startx1 = 0
                stopx1 = self.cubehalfx - maxnx + _x
                startx2 = _x - self.cubehalfx
                stopx2 = maxnx
                
                # partial placement into hypercube
                smallstartx1 = maxnx - (_x - self.cubehalfx)
                smallstartx2 = 0
            
            print(startx, stopy, starty, stopy)
            
            print("centerval is ", centerval)
            
            if self.edgeflag:
                self.hypercube[starty:stopy, smallstartx1:stopx, vel_i, theta_i] = centerval*self.twoddata[starty:stopy, startx1:stopx]
                self.hypercube[starty:stopy, startx1:stopx, vel_i, theta_i] = centerval*self.twoddata[starty:stopy, startx2:maxnx]
            
            else:
                self.hypercube[:, :, vel_i, theta_i] = centerval*self.twoddata[starty:stopy, startx:stopx]
            
#test
hcube = HyperCube()
hcube.load_2d_data(datatype="nhi")
hcube.tabulate_per_vel_theta(vel_i=0, theta_i=0, verbose=True)
