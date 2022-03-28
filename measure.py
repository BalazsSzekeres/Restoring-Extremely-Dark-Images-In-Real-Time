"""
Video Quality Metrics
Copyright (c) 2014 Alex Izvorski <aizvorski@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy
import re
import sys
import scipy.misc

import vifp
import ssim
#import ssim_theano
import psnr
import niqe
import reco
import imageio

def img_greyscale(img):
    return 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]

# ref_file = sys.argv[1]
# dist_file = sys.argv[2]
ref_file = "/home/hvanriessen/projects/Restoring-Extremely-Dark-Images-In-Real-Time/img_results/img_num_2_m_1.0_100k50p.jpg"
dist_file = "/home/hvanriessen/projects/Restoring-Extremely-Dark-Images-In-Real-Time/img_results/img_num_2_m_1.0_100k100p.jpg"

# Inputs are image files
ref = imageio.imread(ref_file).astype(numpy.float32)
dist = imageio.imread(dist_file).astype(numpy.float32)

width, height = ref.shape[1], ref.shape[0]
print("Comparing %s to %s, resolution %d x %d"% (ref_file, dist_file, width, height)) 

# vifp_value = vifp.vifp_mscale(ref, dist)
# print("VIFP=%f"% (vifp_value)) 

ssim_value = ssim.ssim_exact(ref/255, dist/255)
print("SSIM=%f" % (ssim_value)) 

# FIXME this is buggy, disable for now
# ssim_value2 = ssim.ssim(ref/255, dist/255)
# print "SSIM approx=%f" % (ssim_value2)

psnr_value = psnr.psnr(ref, dist)
print("PSNR=%f" % (psnr_value))

# niqe_value = niqe.niqe(dist/255)
# print "NIQE=%f" % (niqe_value)

# reco_value = reco.reco(ref/255, dist/255)
# print("RECO=%f" % (reco_value))
