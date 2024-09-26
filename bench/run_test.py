from fit_spots_easy import SpotFitter
# fitter = SpotFitter(imsize=(50,50,50))
n_images=2
imsize=(50,3000,3000)
fitter = SpotFitter(imsize=imsize)

from time import time
start = time()
fitter.test(n_images=n_images)

print("Time taken:",time()-start)
print(f'to process {n_images} images of shape {imsize}')