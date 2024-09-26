#!/usr/bin/env python
import numpy as np
from scipy import ndimage
from argparse import ArgumentParser
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter
from time import time

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-s', '--shape', type=str,
                         default='50,50,50',
                        help='Shape of the image')
    parser.add_argument('-n', '--nspots', type=int, default=100,
                        help='Number of called spots to generate')
    parser.add_argument('-d', '--delta', type=int, default=3,
                        help='Delta value')    
    return parser.parse_args()

def generate_image(shape, nspots):
    im = np.random.rand(*shape)
    z = np.random.randint(0, shape[0], nspots)
    x = np.random.randint(0, shape[1], nspots)
    y = np.random.randint(0, shape[2], nspots)
    return im, [z, x, y, im[z, x, y]] 

def bogdan_fit(im_dif, coordsh, delta_fit):
    z, x, y, h = coordsh
    Xh = np.array([z,x,y,h]).T
    sigmaZ,sigmaXY =1.5,1.5
    zmax, xmax, ymax = im_dif.shape
    im_centers = [[],[],[],[]]
    Xft = []
    
    # get +- delta fit around the local maxima, include all pixels
    for d1 in range(-delta_fit,delta_fit+1):
        for d2 in range(-delta_fit,delta_fit+1):
            for d3 in range(-delta_fit,delta_fit+1):
                if (d1*d1+d2*d2+d3*d3)<=(delta_fit*delta_fit):
                    im_centers[0].append((z+d1))
                    im_centers[1].append((x+d2))
                    im_centers[2].append((y+d3))
                    im_centers[3].append(im_dif[(z+d1)%zmax,(x+d2)%xmax,(y+d3)%ymax])
                    # get the delta for each pixel
                    Xft.append([d1,d2,d3])
    Xft = np.array(Xft)
    im_centers_ = np.array(im_centers)
    
    bk = np.min(im_centers_[3],axis=0)

    # im_centers_[3] -= bk
    print(im_centers_[3].shape)


    a = np.sum(im_centers_[3],axis=0)

    habs = np.zeros_like(bk)

    sz = delta_fit

    Xft = (np.indices([2*sz+1]*3)-sz).reshape([3,-1]).T
    Xft = Xft[np.linalg.norm(Xft,axis=1)<=sz]
    
    sigma = np.array([sigmaZ,sigmaXY,sigmaXY])[np.newaxis]

    Xft_ = Xft/sigma
    norm_G = np.exp(-np.sum(Xft_*Xft_,axis=-1)/2.)
    norm_G=(norm_G-np.mean(norm_G))/np.std(norm_G)
    im_centers__ = im_centers_[3].T.copy()
    im_centers__ = (im_centers__-np.mean(im_centers__,axis=-1)[:,np.newaxis])/np.std(im_centers__,axis=-1)[:,np.newaxis]
    hn = np.mean(im_centers__*norm_G,axis=-1)
    
    # zc = np.sum(im_centers_[0]*im_centers_[3],axis=0)/np.sum(im_centers_[3],axis=0)
    # xc = np.sum(im_centers_[1]*im_centers_[3],axis=0)/np.sum(im_centers_[3],axis=0)
    # yc = np.sum(im_centers_[2]*im_centers_[3],axis=0)/np.sum(im_centers_[3],axis=0)
    Xh = np.array([z,x,y,bk,a,habs,hn,h]).T

    return Xh

def tensor_fit(im_dif, coordsh, delta_fit):
    z, x, y, h = coordsh
    Xh = np.array([z,x,y,h]).T
    zmax, xmax, ymax = im_dif.shape

    print(zmax, xmax, ymax, im_dif.shape)

    # sigmaZ,sigmaXY =1.5,1.5

    sz=delta_fit
    neighborhood_size = 2 * sz + 1

    # Prepare the tensor to store neighborhoods
    neighborhoods = np.zeros((len(h), neighborhood_size, neighborhood_size, neighborhood_size))
    sigma = 1.5

    Xft = (np.indices([2*sz+1]*3)-sz).reshape([3,-1]).T
    Xft = Xft[np.linalg.norm(Xft,axis=1)<=sz]

    for i, (z, x, y, h) in enumerate(Xh):
        # Define the ranges for slicing

        
        # Extract the neighborhood from im_dif
        sub_image = im_dif[int(max(z - sz, 0)):int(min(z + sz + 1, zmax)),
                            int(max(x - sz, 0)):int(min(x + sz + 1, xmax)),
                              int(max(y - sz, 0)):int(min(y + sz + 1, ymax)),
                              ]
        
        # Place sub-image into the tensor, handling edge sizes
        neighborhoods[i, :sub_image.shape[0], :sub_image.shape[1], :sub_image.shape[2]] = sub_image

        # background_corrected = neighborhoods - neighborhoods.min(axis=(1, 2, 3), keepdims=True)


        # Gaussian smoothing
    
    # z score
    neighborhoods = (neighborhoods - np.mean(neighborhoods))/np.std(neighborhoods)

    smoothed_neighborhoods = np.array([gaussian_filter(sub, sigma=sigma, radius=3) for sub in neighborhoods])
    hn = np.mean(smoothed_neighborhoods, axis=(1, 2, 3))


    return [hn, z, x, y]


def main():
    args = parse_args()
    shape = tuple(map(int, args.shape.split(',')))
    im, coordsh = generate_image(shape, args.nspots)
    t1 = time()
    Xh = bogdan_fit(im, coordsh, args.delta)
    t2 = time()
    print(f'Bogdan fit took {t2 - t1} seconds')
    hn_b = Xh[[0,1,2,6], :]
    t1 = time()
    hn_t = tensor_fit(im, coordsh, args.delta)
    t2 = time()
    print(f'Tensor fit took {t2 - t1} seconds')
    
    for i, x, y, z, in enumerate(zip(hn_t[3], hn_t[2], hn_t[1])):
        print(x==hn_b[0][i], y==hn_b[1][i], z==hn_b[2][i])
    
    print(hn_b - hn_t)
if __name__ == '__main__':
    main()