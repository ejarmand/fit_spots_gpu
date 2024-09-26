# steps
# gaussian conv image
# call max points
# link points
# reconvolve points
import os
import torch
import cupy as cp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv3d
import numpy as np
import cupy as cp
from cupyx.scipy import ndimage
from image_load import read_im
from collections import namedtuple
from time import time


class GaussSmooth(nn.Module):
    def __init__(self, imsize=(3000,3000, 50), sz=5):
        super().__init__()
        self.imsize = imsize
        self.sz = sz
        self.conv = Conv3d(in_channels=1, 
                              out_channels=1,
                                kernel_size=(2*sz+1, 2*sz+1, 2*sz+1), padding='same', bias=False)
        X = np.indices([2*sz+1]*3)-sz
        sigma=1.5
        gaussian = np.exp(-np.sum(X**2,axis=0)/2/sigma**2)
        gaussian = gaussian[np.newaxis, np.newaxis, :, :, :]

        self.conv.weight = torch.nn.Parameter(torch.from_numpy(gaussian).float())
        self.conv.weight.requires_grad = False
        # self.conv.bias = torch.nn.Parameter(torch.zeros(1).float32())
        self.conv = self.conv.cuda()
        self.conv.eval()

    def forward(self, x):
        return self.conv(x)
    
class MaxFilterLayer(nn.Module):
    def __init__(self, delta=1)->None:
        super().__init__()
        self.filter_size = (2*delta+1, 2*delta+1, 2*delta+1)
        padding_check = (self.filter_size[0]-1)/2
        padding = int(padding_check)
        self.pad_adjust = False
        if padding_check.is_integer():
            self.pool = nn.MaxPool3d(kernel_size=self.filter_size, 
                                 stride=1, 
                                 padding=padding
                                 )
        else:
            self.pool = nn.MaxPool3d(kernel_size=self.filter_size, 
                                 stride=1, 
                                 padding=padding + 1
                                 )
            self.pad_adjust = True
    def forward(self, x):
        # x shape: [batch_size, channels, depth, height, width]
        
        # Perform max pooling with stride=1 to slide the window over every voxel
        max_pooled = self.pool(x)

        if self.pad_adjust:
            max_pooled = max_pooled[:, :, :-1, :-1]

        # Create a mask of the local maxima
        mask = (x == max_pooled).float()
        # print('n maximums' , mask.sum())

        # print('max', mask[:, :, -1, -1].sum())
        out = x * mask

        # print('n passed max filter:', (out > 0).sum())
        # Zero out non-maximum pixels
        return out

class BoxFilterBySlice(nn.Module):
    def __init__(self, size=30) -> None:
        super().__init__()
        self.size = size
        self.pad_adjust = False
        padding_check = (size-1)/2
        if padding_check.is_integer():
            padding = int(padding_check)
            self.norm = nn.AvgPool2d(size, stride=1,
                            padding=padding,
                            count_include_pad=False)
            
        else:
            padding = int(padding_check)+1
            self.norm = nn.AvgPool2d(size, stride=1,
                            padding=padding,
                            count_include_pad=False)
            self.pad_adjust=True
       
    def forward(self, x):
        norm = self.norm(x)
        if self.pad_adjust:
            norm = norm[:, :, :-1, :-1]
        return x - norm
    
class FilterAndMax(nn.Module):
    def __init__(self, imsize=(3000,3000, 50),
                  sz=5, delta=1, box_size=30) -> None:
        super().__init__()
        self.gauss1 = GaussSmooth(imsize, sz)
        self.max_filter = MaxFilterLayer(delta)
        self.box_filter = BoxFilterBySlice(box_size)

    def forward(self, x):
        x = self.gauss1(x)
        # print('gaussian filter:', x.shape)
        x = self.box_filter(x)
        # print('box filter:', x.shape)
        x_max = self.max_filter(x)
        # print('max filter:', x_max.shape)
        # print('n passed max filter:', (x_max > 0).sum())
        return x, x_max


def density_label_gpu(input):
    '''
    Takes in a 3D image and averages the values and coordinates
    of connected compenents

    '''
    input = cp.asarray(input) 
    spots =  input > 0
    labeled, n_labels = ndimage.label(spots, structure=cp.ones((3,3,3)))
    mean_values = ndimage.mean(input, labeled, cp.arange(1, n_labels+1))

    z, x, y = cp.array(ndimage.center_of_mass(spots.astype(cp.int32), labeled, cp.arange(1, n_labels+1))).T

    Xh = cp.array([z, x, y, mean_values]).T
    return Xh

def density_label_cpu(im_dif):
    '''
    Takes in a 3D image and averages the values and coordinates
    of connected compenents

    implementation from Bogdan Bintu
    '''
    im_dif = im_dif.cpu().numpy()
    z, x, y = np.where(im_dif > 0)
    h = im_dif[z, x, y]
    from scipy import ndimage
    im_keep = np.zeros(im_dif.shape,dtype=bool)
    im_keep[z,x,y]=True
    lbl, nlbl = ndimage.label(im_keep,structure=np.ones([3]*3))
    l=lbl[z,x,y]#labels after reconnection
    ul = np.arange(1,nlbl+1)
    il = np.argsort(l)
    l=l[il]
    z,x,y,h = z[il],x[il],y[il],h[il]
    inds = np.searchsorted(l,ul)
    Xh = np.array([z,x,y,h]).T
    Xh_ = []
    for i_ in range(len(inds)):
        j_=inds[i_+1] if i_<len(inds)-1 else len(Xh)
        Xh_.append(np.mean(Xh[inds[i_]:j_],0))
    Xh=np.array(Xh_)
    return Xh

def smooth_fit_pytorch(delta_fit,
                      Xh, im_dif,
                      sigmaZ=1, 
                      sigmaXY=1.5,
                      zmax=50, 
                      xmax=3000, ymax=3000,
                      device='cuda'):
    
    # im_dif = torch.from_numpy(im_dif_npy).to(dev)
    def get_ind(x,xmax):
        # modify x_ to be within image
        x_ = torch.clone(x)
        bad = x_>=xmax
        x_[bad]=xmax-x_[bad]-2
        bad = x_<0
        x_[bad]=-x_[bad]
        return x_
    z, x, y, h = Xh.T
    z,x,y = z.to(torch.int32),x.to(torch.int32),y.to(torch.int32)
    
    d1,d2,d3 = np.indices([2*delta_fit+1]*3).reshape([3,-1])-delta_fit
    kp = (d1*d1+d2*d2+d3*d3)<=(delta_fit*delta_fit)
    d1,d2,d3 = d1[kp],d2[kp],d3[kp]
    d1 = torch.from_numpy(d1).to(device)
    d2 = torch.from_numpy(d2).to(device)
    d3 = torch.from_numpy(d3).to(device)
    im_centers0 = (z.reshape(-1, 1)+d1).T
    im_centers1 = (x.reshape(-1, 1)+d2).T
    im_centers2 = (y.reshape(-1, 1)+d3).T
    z_ = get_ind(im_centers0,zmax)
    x_ = get_ind(im_centers1,xmax)
    y_ = get_ind(im_centers2,ymax)
    im_centers3 = im_dif[z_,x_,y_]

    # im centers 4
    im_raw_ = im_dif
    im_centers4 = im_raw_[z_,x_,y_]
    habs = im_raw_[z,x,y]
    a=habs

    Xft = torch.stack([d1,d2,d3]).T

    sigma = torch.tensor([sigmaZ,sigmaXY,sigmaXY],dtype=torch.float32,device=device) 
    Xft_ = Xft/sigma
    norm_G = torch.exp(-torch.sum(Xft_*Xft_,-1)/2.)
    norm_G=(norm_G-torch.mean(norm_G))/torch.std(norm_G)

    hn = torch.mean(((im_centers3-im_centers3.mean(0))/im_centers3.std(0))*norm_G.reshape(-1,1),0)
    a = torch.mean(((im_centers4-im_centers4.mean(0))/im_centers4.std(0))*norm_G.reshape(-1,1),0)

    bk = torch.min(im_centers3,0).values
    im_centers3 = im_centers3-bk
    im_centers3 = im_centers3/torch.sum(im_centers3,0)

    zc = torch.sum(im_centers0*im_centers3,0)
    xc = torch.sum(im_centers1*im_centers3,0)
    yc = torch.sum(im_centers2*im_centers3,0)
    Xh = torch.stack([zc,xc,yc,bk,a,habs,hn,h]).T.cpu().detach().numpy()

    return Xh

def final_smooth_fit(delta_fit,
                      Xh, im_dif,
                      sigmaZ=1, 
                      sigmaXY=1.5,
                      zmax=50, 
                      xmax=3000, ymax=3000):
    '''
    I haven't completely worked out what this function does,
    so I'm just reimplementing bogdan's stuff with cupy

    this represents just the latter half fo get_local_max
    
    '''
    z, x, y, h = Xh.T
    z,x, y = z.astype(np.int32), x.astype(np.int32), y.astype(np.int32)
    im_centers = [[],[],[],[],[]]
    Xft = []
    # im_dif = im_dif.tonumpy()
    for d1 in range(-delta_fit,delta_fit+1):
        for d2 in range(-delta_fit,delta_fit+1):
            for d3 in range(-delta_fit,delta_fit+1):
                if (d1*d1+d2*d2+d3*d3)<=(delta_fit*delta_fit):
                    im_centers[0].append((z+d1))
                    im_centers[1].append((x+d2))
                    im_centers[2].append((y+d3))

                    im_centers[3].append(im_dif[(z+d1)%zmax,(x+d2)%xmax,(y+d3)%ymax])
                    Xft.append([d1,d2,d3])

    Xft = cp.array(Xft)
    im_centers_ = cp.array(im_centers)        

    bk = cp.min(im_centers_[3], axis=0)
    im_centers_[3] -= bk
    a = cp.sum(im_centers_[3], axis=0)
    habs = cp.zeros_like(bk)
    habs = im_dif[z%zmax,x%xmax,y%ymax]
    sz = delta_fit
    Xft = (cp.indices([2*sz+1]*3)-sz).reshape([3,-1]).T
    Xft = Xft[cp.linalg.norm(Xft,axis=1)<=sz]

    sigma = cp.array([sigmaZ,sigmaXY,sigmaXY])
    Xft_ = Xft/sigma

    norm_G = cp.exp(-cp.sum(Xft_*Xft_,axis=-1)/2.)
    norm_G=(norm_G-cp.mean(norm_G))/cp.std(norm_G)
    im_centers__ = im_centers_[3].T.copy()
    im_centers__ = (im_centers__-cp.expand_dims(cp.mean(im_centers__,axis=-1), axis=-1))/cp.expand_dims(np.std(im_centers__,axis=-1), axis=-1)
    hn = cp.mean(im_centers__*norm_G,axis=-1)
  
    zc = cp.sum(im_centers_[0]*im_centers_[3],axis=0)/cp.sum(im_centers_[3],axis=0)
    xc = cp.sum(im_centers_[1]*im_centers_[3],axis=0)/cp.sum(im_centers_[3],axis=0)
    yc = cp.sum(im_centers_[2]*im_centers_[3],axis=0)/cp.sum(im_centers_[3],axis=0)
   
    Xh = cp.asnumpy(cp.array([zc,xc,yc,bk,a,habs,hn,h]).T)
    return Xh

class SpotFitter:
    '''
    An object that uses GPU methods for fast spot fitting.

Attributes:

imsize: tuple 
    The size of the image to be processed.

sz: int
    The size of the initial gaussian filter.

box_size: int
    The size of the initial box (lowpass) filter.

delta: int
    The size of the maximum filter. Spots within a range that aren't maximum values are not considered.

delta_fit: int
    The size of the fit filter. This is used to fit the spots to a gaussian.

analysis_folder: str
    The folder where the analysis is stored.

sigmaZ: float
    The Z width of the second gaussian filter in the fit.

sigmaXY: float
    The XY width of the width gaussian filter in the fit.

device: str
    The device to use for the computation. Default is 'cuda'.
    Currently 'CPU' is not supported.

label_method: str
    The method to use for labeling the spots. Default is 'CPU'.
    For repeated runs, 'GPU' may be faster, but has prohibitively
    slow startup for large images due to GPU program compilaiton.

    for a 50 x 1000 x 1000 image the CPU takes 2 seconds, GPU takes .5 seconds,
    but the first GPU run takes 37 seconds to compile. The smaller and more repative
    the image processing, the more the GPU is favored.

    Kernel compilation takes O(n^2) time, wile runtime for both gpu and cpu is O(n).
    For a 3000 x 3000 x 50 image, the CPU takes 18 seconds, gpu compile takes 50 minutes.
    It should be slightly faster to use a GPU for this size if you process over than 220 images.

    currently looking into a way to compile the kernel once and keep it for common image sizes.
redo: bool
    Wether to redo the analysis if the file already exists.

filter_max: nn.Module
    A fast pytorch pipeline for the intial box, gaussian, and maximum filters.
    
    '''
    def __init__(self, 
                 imsize=(3000,3000, 50),
                sz=5,
                box_size=30, 
                delta=1,
                delta_fit=3,
                sigmaZ=1,
                sigmaXY=1.5,
                device='cuda',
                label_method='CPU',
                redo=False):      
        self.imsize = imsize
        self.sz = sz
        self.box_size = box_size
        self.delta = delta
        self.delta_fit = delta_fit
        self.sigmaZ = sigmaZ
        self.sigmaXY = sigmaXY
        self.redo = redo
        self.device = device
        self.label_method = label_method
        self.label_func = density_label_cpu if label_method == 'CPU' else density_label_gpu 
        self.filter_max = FilterAndMax(imsize, sz, delta, box_size).to(self.device)

    def fit(self, imS):
        imS = torch.from_numpy(imS).to(self.device)
        imS, spot_image = self.filter_max(imS)
        spot_image, imS = spot_image.squeeze(), imS.squeeze()
        Xh = self.label_func(spot_image)
        Xh = final_smooth_fit(self.delta_fit, 
                              Xh, 
                              imS, 
                              self.sigmaZ,
                              self.sigmaXY, *self.imsize)
        return Xh
    
    def fit_time(self, imS):
        from time import time
        imS = torch.from_numpy(imS).to(self.device)
        imS, spot_image = self.filter_max(imS)
        spot_image, imS = spot_image.squeeze(), imS.squeeze()

        start = time()

        Xh = density_label_cpu(spot_image)
        print('bogdan time :', time()-start) 
        
        spot_image = cp.asarray(spot_image)
        start = time()
        Xh = density_label_gpu(spot_image)

        print('nd_image_time:', time() - start)

        Xh = torch.as_tensor(Xh, device=self.device)

        Xh = smooth_fit_pytorch(self.delta_fit, 
                              Xh, 
                              imS, 
                              self.sigmaZ,
                              self.sigmaXY, *self.imsize, device=self.device)
        return Xh

    def test_image(self, nspots=100):
        shape = self.imsize
        im = np.random.rand(*shape)
        z = np.random.randint(0, shape[0], nspots)
        x = np.random.randint(0, shape[1], nspots)
        y = np.random.randint(0, shape[2], nspots)
        im[z, x, y] = 1.2
        im = im[np.newaxis, :,  :, :]
        return im
    
    def load_icol(self, hybe, fov, icol):
        fl = hybe+os.sep+fov
        tag = os.path.basename(hybe)
        save_file = self.analysis_folder+os.sep+fov.split('.')[0]+'--'+tag+'_fits_icol'+str(icol)+'.npz'
        
        if os.path.exists(save_file) and not self.redo:
            print('File aleady processed')
            return
        
        im = read_im(fl)
        ncols,sz,sx,sy = im.shape
        imS = np.array(im[icol],dtype=np.float32)
        imS = np.expand_dims(imS, 0)
        return imS
    
    def test(self, n_spots=100, n_images=1):
        for i in range(n_images):
            im = self.test_image(n_spots)
            # im = np.expand_dims(im, 0).astype(np.float32)
            start = time()

            im = im.astype(np.float32)
            Xh = self.fit_time(im)
            print(f"Time taken for image {i}:",time()-start)
        return Xh, im

