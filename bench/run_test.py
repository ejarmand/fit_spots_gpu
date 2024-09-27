#!/usr/bin/env python
from fit_spots_gpu import SpotFitter
from argparse import ArgumentParser
from time import time


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-s', '--im_shape', type=str,
                         default='50,50,50',
                        help='Shape of the image')
    parser.add_argument('-p', '--pixel_count', type=int, default=100,
                        help='generates a minimum number of max value pixels')
    parser.add_argument('-n', '--n_images', type=int, default=2,
                        help='Number of images to process')
    return parser.parse_args()

def main():
    args = parse_args()
    imsize = args.im_shape.split(',')
    assert len(imsize) == 3, 'Shape must be 3 values'
    imsize = [int(i) for i in imsize]
    fitter = SpotFitter(imsize=imsize)
    start = time()

    fitter.test(n_spots=args.pixel_count, n_images=args.n_images)
    print("Time taken:",time()-start)
    print(f'to process {args.n_images} images of shape {imsize}')

if __name__ == '__main__':
    main()