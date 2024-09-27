# fit spots GPU

This repo conatians the code to perform spot fitting of merfish images
usign fast gpu algorithms.

These are a modification of code written by Bogdan Bintu.

## Disclaimer! 

The following repository is a draft and has only been tested on synthetic images

### Installation

First download this repository and enter the directory:

```
git clone https://github.com/ejarmand/spot_fitting_gpu.git
cd spot_fitting_gpu
```


install requirements:

```mamba create -n fit_spots_gpu -f requirements.yml``

or to add requirements to existing env (claiming no responsibility if it breaks):

```mamba env update -f requirements.yml```

then install the package with pip:

```pip install -e src```

### example usage

```python
from fit_spots_gpu import SpotFitter
from fit_spots_gpu.image_load import read_im, load_image_dir
import numpy as np
import os
from tqdm import tqdm # not a package requirement provides progress bar


analysis_folder = './data'
img_zarr_files = load_image_dir(expt_path)
color_channels = np.arange(3)
imsize = (50, 3000, 3000)

fitter= SpotFitter(imsize)
zarray = None
for hybe, fov in tqdm(img_zarr_files):
    in_file = analysis_folder+ os.sep +hybe + os.sep +fov
    loaded = False
    for ch in color_channels:
        save_file = analysis_folder+os.sep+fov.split('.')[0]+'--'+hybe+'_fits_icol'+str(icol)+'.npz'
        if save_file not in os.listdir(analysis_folder+os.sep+hybe):
            if not loaded:
                zarry, sz, sx, sy = read_im(in_file)
                loaded = True
            im = np.array(zarray[ch], dtype=np.float32)            
            Xh = fitter.fit(hybe, fov, ch)
            np.savez_compressed(save_file, Xh=Xh)
```