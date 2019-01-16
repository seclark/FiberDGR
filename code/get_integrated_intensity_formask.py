import numpy as np

# local repo imports
import sys
sys.path.insert(0, '../../GalfaCuber/code')
import galfa_vel_helpers as gvh

mask = fits.getdata('/data/seclark/Planck/mask_GALFA_footprint_absb_gt_30_lt_90_NHImin_0max_8_HFI_PS_True.fits')

vel_root = '/data/seclark/GALFADR2/Wide_maps/'

int_emiss_dict = {}
for _velkey in galfa_name_dict.keys():
    
    # get integer velocity number
    vel_int = np.int(_velkey)
    
    # get filename for velocity slice
    vel_fn = gvh.get_galfa_W_name(vel)
    
    # load velocity slice
    velslice = fits.getdata(vel_root + vel_fn)
    
    # save sum of intensity in mask 
    int_emiss_dict[_velkey] = np.nansum(velslice[np.where(mask > 0)])
    
np.save('/data/seclark/FiberDGR/data/GALFA_HI_Iv_dict_mask_GALFA_footprint_absb_gt_30_lt_90_NHImin_0max_8_HFI_PS_True.npy', int_emiss_dict)