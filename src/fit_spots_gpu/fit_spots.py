from scipy.signal import fftconvolve
import os
from itertools import product
import torch
from fft_conv_pytorch import FFTConv3d
import cupy as cp
from cupyx.scipy.ndimage import maximum_filter, label
import torch
import torch.nn as nn
import torch.nn.functional as F






dir_structure = '/projects/ps-renlab2/zgibbs/STARR-FISH/10_12_2023__CRE-20_RNASEQ', 'hybes', 'fovs' # each fov is a zarr, whith a color channel (0-2) 

def fit_image_round(hybe,fov,analysis_folder='/projects/ps-renlab2/zgibbs/STARR-FISH/10_12_2023__CRE-20_RNASEQ',icol=0,redo=False,hth = 50,plt_val=True):
    fl = hybe+os.sep+fov
    tag = os.path.basename(hybe)
    save_file=analysis_folder+os.sep+fov.split('.')[0]+'--'+tag+'_fits_icol'+str(icol)+'.npz'
    if not os.path.exists(save_file) or redo:
        im = read_im(fl)
        ncols,sz,sx,sy = im.shape
        #for icol in np.arange(ncols-1):
        imS = np.array(im[icol],dtype=np.float32) #GFP cy5 half
        #Convolve with gaussian 
        sz=5
        X = np.indices([2*sz+1]*3)-sz
        sigma=1.5
        gaussian = np.exp(-np.sum(X**2,axis=0)/2/sigma**2)
        im_conv = fftconvolve(imS,gaussian,mode='same')/np.sum(gaussian)
        #Subtract local background

        # need norm slice function
        im_conv_ = norm_slice(im_conv,s=30)
        #fit the spots
        Xh = get_local_max(im_conv_,hth,im_raw=im_conv,delta=1,delta_fit=3)
        if plt_val:
            import napari
            v = napari.view_image(im_conv_)
            v.add_image(imS)
            h = Xh[:,-1]
            size = 5+np.clip(h/np.percentile(h,99.9),0,1)*10
            v.add_points(Xh[:,:3],size=size,face_color=[0,0,0,0],edge_color='y')
        np.savez_compressed(save_file,Xh=Xh)


class Conv:
    def __init__(self, imsize=(3000,3000, 50), sz=5):
        self.imsize = imsize
        self.sz = sz
        self.batch_size = batch_size
        self.conv = FFTConv3d(imsize, sz)
        X = np.indices([2*sz+1]*3)-sz
        sigma=1.5
        gaussian = np.exp(-np.sum(X**2,axis=0)/2/sigma**2)
        self.conv.weight = torch.nn.Parameter(torch.from_numpy(gaussian).float32())
        self.conv.weight.requires_grad = False
        # self.conv.bias = torch.nn.Parameter(torch.zeros(1).float32())
        self.conv = self.conv.cuda()
        self.conv.eval()

    def convolve(self, batch):
        return self.conv(batch).detach().cpu().numpy()
    
class MaxFilterLayer(nn.Module):
    def __init__(self, delta=1):
        super(MaxFilterLayer, self).__init__()
        self.filter_size = (2*delta+1, 2*delta+1, 2*delta+1)

    def forward(self, x):
        # x shape: [batch_size, channels, depth, height, width]
        
        # Calculate padding to keep the output size same as input
        padding = self.filter_size // 2

        # Perform max pooling with stride=1 to slide the window over every voxel
        max_pooled = F.max_pool3d(x, kernel_size=self.filter_size, stride=1, padding=padding)

        # Create a mask of the local maxima
        mask = (x == max_pooled).float()

        # Zero out non-maximum pixels
        return x * mask
    
class SpotFitter:
    def __init__(self, imsize=(3000,3000, 50), sz=5):
        self.im = im
        self.sz = sz
        self.save_file = save_file
        self.Xh = get_local_max(im,hth,im_raw=im,delta=1,delta_fit=3)
        np.savez_compressed(save_file,Xh=Xh)

    def fit_spots(self):
        pass
    def local_max(self, im, threshold=50, delta=1,delta_fit=3,dbscan=True,return_centers=False,mins=None,sigmaZ=1,sigmaXY=1.5):
        
def simple_max_filter(data, delta=1):
    data_gpu = cp.asarray(data)
    data_max = maximum_filter(data, size=(2*delta+1, 2*delta+1, 2*delta+1))
    keep = data_gpu == data_max
    result = data_gpu * keep
    return cp.asnumpy(result)


def density_label(input):
    # input is a 3D array
    spots = cp.asarray(input) > 0
    labeled, n_labels = label(spots)
    # labeled_vals = labeled > 0 # labeld values
    vals, ids = cp.unique(labeled, return_inverse=True)
    ids = ids.reshape(input.shape)
    Xh = []
    for label in enumerate(vals[1:]):
        # only link if there are more than one
        if cp.sum(ids == label) > 1:
            vals = cp.argwhere(ids == label)
            x, y, z = cp.mean(vals, axis=0)
            Xh.append(z, x, y, cp.mean(input[vals]))

        else:
            x, y, z = cp.argwhere(ids == label)
            Xh.append(z, x, y, input[z, x, y])

    return cp.array(Xh).T

def spot_mask(shape, xh, delta_fit=3):
    # shape is the shape of the image
    # xh is the output of density label
    # delta_fit is the size of the mask
    mask = cp.zeros(shape)
    
    for d1 in range(-delta_fit,delta_fit+1):
        for d2 in range(-delta_fit,delta_fit+1):
            for d3 in range(-delta_fit,delta_fit+1):
                

def center_kernel(center_array, in_array, delta_fit=3):
    








    # labele_vals = cu.arange(1, n_labels+1)


    # this function averages the location and intensity of labeled spots



    # delta fit stuff
####
    z,x,y,h = z[il],x[il],y[il],h[il]
    inds = np.searchsorted(l,ul)
    Xh = np.array([z,x,y,h]).T
    Xh_ = []
    for i_ in range(len(inds)):
        j_=inds[i_+1] if i_<len(inds)-1 else len(Xh)
        Xh_.append(np.mean(Xh[inds[i_]:j_],0))
    Xh=np.array(Xh_)
    z,x,y,h = Xh.T

####

def fit_centers(xh, delta_fit=3):
    
    pass

####        
# fovs are 
fov_format = f'Conv_zscan__{fov}.zarr'

def load_image_dir(analysis_folder='/projects/ps-renlab2/zgibbs/STARR-FISH/10_12_2023__CRE-20_RNASEQ',
                   redo=False):
    hybes = os.listdir(analysis_folder)
    fovs = os.listdir(analysis_folder+os.sep+hybes[0])
    combinations = list(product(hybes,fovs))
    return combinations

def fit_images(analysis_folder='/projects/ps-renlab2/zgibbs/STARR-FISH/10_12_2023__CRE-20_RNASEQ',
               redo=False, icols=[0,1,2], im_shape=(3000,3000,50), sz=5, n_icols=3):
    combinations = load_image_dir(analysis_folder,redo,icols)
    conv = Conv(im_shape, sz, batch_size=n_icols*2)
    # get two images
    for c1, c2 in zip(combinations[0::2], combinations[1::2]):
        hybe1, fov1 = c1
        hybe2, fov2 = c2
        zarr_path1 = analysis_folder+os.sep + hybe1+os.sep+fov1
        zarr_path2 = analysis_folder+os.sep + hybe2+os.sep+fov2
        im1 = read_im(zarr_path1)
        im2 = read_im(zarr_path2)
        batch = None # stack batch of things
        outs = conv.convolve(batch)
        for out in outs:
            # fit spots
            pass

        
    
def dbscan():
    # label based on 

    pass