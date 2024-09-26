def smooth_fit_pytorch(delta_fit,
                      Xh, im_dif,
                      sigmaZ=1, 
                      sigmaXY=1.5,
                      zmax=50, 
                      xmax=3000, ymax=3000):
    
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
    d1 = torch.from_numpy(d1).to(dev)
    d2 = torch.from_numpy(d2).to(dev)
    d3 = torch.from_numpy(d3).to(dev)
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

    Xft = torch.stack([d1,d2,d3]).T

    bk = torch.min(im_centers3,0).values
    im_centers3 = im_centers3-bk
    im_centers3 = im_centers3/torch.sum(im_centers3,0)

    zc = torch.sum(im_centers0*im_centers3,0)
    xc = torch.sum(im_centers1*im_centers3,0)
    yc = torch.sum(im_centers2*im_centers3,0)
    Xh = torch.stack([zc,xc,yc,bk,a,habs,hn,h]).T.cpu().detach().numpy()

    return Xh