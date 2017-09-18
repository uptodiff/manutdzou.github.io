"""
STAIN.NORM: various methods for stain normalization.
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np
from PIL import Image
import os,sys
from scipy.optimize import nnls
from skimage.util import img_as_float
from skimage.exposure import rescale_intensity
#sys.path.insert(0,'/home/zou/stain')
#import histomicstk as htk
def compute_macenko_norm_matrix(im, alpha=1.0, beta=0.15):
    """
    Implements the staining normalization method from
    Macenko M. et al. "A method for normalizing histology slides for
    quantitative analysis". ISBI 2009
    :param im:
    :param alpha:
    :param beta:
    :return:
    """
    if im.ndim != 3:
        raise ValueError('Input image must be RGB')
    if 'Io' not in locals().keys():
        Io = 240.0
    if 'maxCRef' not in locals().keys():
        maxCRef = np.array([1.9705,1.0308]).reshape((2,1))
    if 'HERef' not in locals().keys():
        HERef= np.array([0.5626,0.2159,0.7201,0.8012,0.4062,0.5581]).reshape((3,2))
    h, w, _ = im.shape
    im = (im + 1.0) / Io # img_as_float(im)
    # im = rescale_intensity(im, out_range=(0.001, 1.0)) # we'll take log...
    im = im.reshape((h*w, 3), order='F')
    od = -np.log(im) # optical density
    odhat = od[~np.any(od < beta, axis=1), ]
    _, V = np.linalg.eigh(np.cov(odhat, rowvar=0)) # eigenvectors of a symmetric matrix
    theta = np.dot(odhat,V[:, 1:3])
    phi = np.arctan2(theta[:,1], theta[:,0])
    minPhi, maxPhi = np.percentile(phi, [alpha, 100-alpha])
    vec1 = np.dot(V[:,1:3] , np.array([[np.cos(minPhi)],[np.sin(minPhi)]]))
    vec2 = np.dot(V[:,1:3] , np.array([[np.cos(maxPhi)],[np.sin(maxPhi)]]))

    if vec1[0] > vec2[0]:
        HE = np.hstack((vec1, vec2))
    else:
        HE = np.hstack((vec2, vec1))
    Y=od.T
    C = np.mat(HE).I*Y
    maxC = np.percentile(C, 99,axis=1)
    C /= maxC[:,np.newaxis]
    C = np.multiply(C,maxCRef)
    Inorm = np.array(Io*np.exp(-HERef * C))
    Inorm = np.reshape(Inorm.T, (h, w, 3))
    H=0
    E=0
    if 0:
        H = np.array(Io*np.exp(-HERef[:,0] * C[0,:]))
        H = np.reshape(H.T, (h, w, 3))
        H = H.astype(np.uint8)

    if 0:
        E = np.array(Io*np.exp(-HERef[:,1] * C[1,:]))
        E = np.reshape(E.T, (h, w, 3))
        E = E.astype(np.uint8)
    return Inorm,H,E

crc_images_folder = './Kather_texture_2016_larger_images_10'
filenames = ['CRC-Prim-HE-01_APPLICATION.tif',
'CRC-Prim-HE-02_APPLICATION.tif',
'CRC-Prim-HE-03_APPLICATION.tif',
'CRC-Prim-HE-04_APPLICATION.tif',
'CRC-Prim-HE-05_APPLICATION.tif',
'CRC-Prim-HE-06_APPLICATION.tif',
'CRC-Prim-HE-07_APPLICATION.tif',
'CRC-Prim-HE-08_APPLICATION.tif',
'CRC-Prim-HE-09_APPLICATION.tif',
'CRC-Prim-HE-10_APPLICATION.tif']

for filename in filenames:
    tile = np.asarray(Image.open(os.path.join(crc_images_folder, filename)))
    print('applying stain normalization to file {}...'.format(filename))
    tile_normalized ,_,_ = compute_macenko_norm_matrix(tile, alpha=1.0, beta=0.15)
    im = Image.fromarray(tile_normalized.astype(np.uint8))
    im.save(os.path.splitext(filename)[0]+'_normalized.tif')
    print('image saved to disk')
