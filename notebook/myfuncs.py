from rdp import rdp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from csv import reader
import math
from scipy import signal
from tsdownsample import M4Downsampler,EveryNthDownsampler,LTTBDownsampler,LTOBDownsampler,\
    MinMaxDownsampler,LTDDownsampler,LTDOBDownsampler,LTOBETDownsampler,LTTBETDownsampler,\
    LTTBETGapDownsampler,LTOBETGapDownsampler,MinMaxGapDownsampler,LTTBETNewDownsampler,\
    LTTBETFurtherDownsampler,MinMaxFPLPDownsampler,LTSDownsampler,ILTSParallelDownsampler
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float
from skimage.metrics import mean_squared_error
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float
from skimage.metrics import mean_squared_error
from skimage.util import invert 
from matplotlib.ticker import AutoMinorLocator, FixedLocator
from matplotlib import image as mpimg
from PIL import Image
from matplotlib.transforms import IdentityTransform
import random
import scipy.ndimage as ndi
from textwrap import wrap
from matplotlib.ticker import ScalarFormatter
import os

def full_frame(width=None, height=None, dpi=None):
    import matplotlib as mpl
    # First we remove any padding from the edges of the figure when saved by savefig. 
    # This is important for both savefig() and show(). Without this argument there is 0.1 inches of padding on the edges by default.
    mpl.rcParams['savefig.pad_inches'] = 0
    figsize = None if width is None else (width/dpi, height/dpi) # so as to control pixel size exactly
    fig = plt.figure(figsize=figsize,dpi=dpi)
    # Then we set up our axes (the plot region, or the area in which we plot things).
    # Usually there is a thin border drawn around the axes, but we turn it off with `frameon=False`.
    ax = plt.axes([0,0,1,1], frameon=False)
    # Then we disable our xaxis and yaxis completely. If we just say plt.axis('off'),
    # they are still used in the computation of the image padding.
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Even though our axes (plot region) are set to cover the whole image with [0,0,1,1],
    # by default they leave padding between the plotted data and the frame. We use tigher=True
    # to make sure the data gets scaled to the full extents of the axes.
    plt.autoscale(tight=True)
    
def exp(t, A, lbda):
    r"""y(t) = A \cdot \exp(-\lambda t)"""
    return A * np.exp(-lbda * t)

def sine(t, omega, phi):
    r"""y(t) = \sin(\omega \cdot t + phi)"""
    return np.sin(omega * t + phi)

def damped_sine(t, A, lbda, omega, phi):
    r"""y(t) = A \cdot \exp(-\lambda t) \cdot \left( \sin \left( \omega t + \phi ) \right)"""
    return exp(t, A, lbda) * sine(t, omega, phi)

def _get_or_conv_mask(f1, f2, win_size: int = 11) -> np.ndarray:
    # get inverted img: 0 for original white background, 255 for original dark foreground
    img1 = cv2.imread(f1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # [0,255], np.uint8. Right now a larger value means less data points (thus brighter/whiter)
    img1 = invert(img1) # [0,255], np.uint8. This step to make a larger value means more data points.
    # img1 = img_as_float(img1) # [0,1]

    img2 = cv2.imread(f2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) # [0,255], np.uint8. Right now a larger value means less data points (thus brighter/whiter)
    img2 = invert(img2) # [0,255], np.uint8. This step to make a larger value means more data points.
    # img2 = img_as_float(img2) # [0,1]

    joined=(img1+img2)>0
    # plt.imshow(joined)
    # plt.show()
    
    or_conv = signal.convolve2d(joined, np.ones((win_size, win_size)), mode="same")
    or_conv_mask = or_conv>0
    # plt.imshow(or_conv_mask)
    # plt.show()

    return or_conv_mask


def match(imfil1,imfil2):    
    img1=cv2.imread(imfil1)    
    (h,w)=img1.shape[:2]    
    img2=cv2.imread(imfil2)    
    resized=cv2.resize(img2,(w,h))    
    (h1,w1)=resized.shape[:2]    
    img1=img_as_float(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)) # img_as_float: the dtype is uint8, means convert [0, 255] to [0, 1]
    img2=img_as_float(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY))
    return ssim(img1,img2,data_range=img1.max()-img1.min())

def match_masked(imfil1,imfil2):
    img1=cv2.imread(imfil1)
    (h,w)=img1.shape[:2]    
    img2=cv2.imread(imfil2)    
    resized=cv2.resize(img2,(w,h))    
    (h1,w1)=resized.shape[:2]    
    img1=img_as_float(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)) # img_as_float: the dtype is uint8, means convert [0, 255] to [0, 1]
    img2=img_as_float(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY))

    or_conv_mask = _get_or_conv_mask(imfil1,imfil2)

    ssim_kwgs = dict(win_size=11, full=True, gradient=False)
    SSIM = ssim(img1, img2, data_range=img1.max()-img1.min(),**ssim_kwgs)[1]

    SSIM_masked = SSIM.ravel()[or_conv_mask.ravel()]
    return np.mean(SSIM_masked)


def mse_in_255_masked(imfil1, imfil2):
    # mse_in_255 use divider as the total number of global pixels
    # while here only considers masked pixels

    img1 = cv2.imread(imfil1)
    img2 = cv2.imread(imfil2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    squared_diff = (img1.astype("float") -img2.astype("float")) ** 2

    or_conv_mask = _get_or_conv_mask(imfil1,imfil2)

    mse_masked = squared_diff.ravel()[or_conv_mask.ravel()].mean() 
    # actually sum is the same as mse_in_255, the only difference is divider
    # mse_in_255 use divider as the total number of global pixels
    # while here only uses the number of masked pixels as divider

    return mse_masked


def mse_in_255(imfil1, imfil2): # mse=mse_in_255/(255*255)
    img1 = cv2.imread(imfil1)
    img2 = cv2.imread(imfil2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    squared_diff = (img1.astype("float") -img2.astype("float")) ** 2
    summed = np.sum(squared_diff)
    num_pix = img1.shape[0] * img1.shape[1] #img1 and 2 should have same shape
    err = summed / num_pix
    return err

def mse(imfil1,imfil2): # mse=mse_in_255/(255*255)
    img1 = cv2.imread(imfil1)
    img2 = cv2.imread(imfil2)
    img1 = img_as_float(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
    img2 = img_as_float(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
    squared_diff = (img1-img2) ** 2
    summed = np.sum(squared_diff)
    num_pix = img1.shape[0] * img1.shape[1] #img1 and 2 should have same shape
    err = summed / num_pix
    return err

def pem20(f1,f2):
    # Pixel Error Margin 20
    # Reference: Data Point Selection for Line Chart Visualization:Methodological Assessment and Evidence-Based Guidelines
    img1 = cv2.imread(f1)
    img2 = cv2.imread(f2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img_diff = img1.astype("float") - img2.astype("float")
    res=0
    for iy, ix in np.ndindex(img_diff.shape):
        x=img_diff[iy, ix]
        if abs(x)>20:
            res=res+1
    num_pix = img1.shape[0] * img1.shape[1] #img1 and 2 should have same shape
    res=res/num_pix
    return res

def compare_vis(f1,f2):
    img1 = cv2.imread(f1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # [0,255], np.uint8. Right now a larger value means less data points (thus brighter/whiter)
    img1 = invert(img1) # [0,255], np.uint8. This step to make a larger value means more data points.
    img1 = img_as_float(img1) # [0,1]
    
    img2 = cv2.imread(f2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) # [0,255], np.uint8. Right now a larger value means less data points (thus brighter/whiter)
    img2 = invert(img2) # [0,255], np.uint8. This step to make a larger value means more data points.
    img2 = img_as_float(img2) # [0,1]
    
    plt.imshow(img1 - img2, cmap="gray")
    plt.title("{} \n vs \n {}\n Black: latter data, but no former data.\n White: former data, but no latter data".format(f1,f2))
    plt.show()

    
def compare_vis_color(f1,f2):
    # color -> data
    img1 = cv2.imread(f1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # [0,255], np.uint8. Right now a larger value means less data points (thus brighter/whiter)
    img1 = invert(img1) # [0,255], np.uint8. This step to make a larger value means more data points.
    img1 = img_as_float(img1) # [0,1]
    
    img2 = cv2.imread(f2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) # [0,255], np.uint8. Right now a larger value means less data points (thus brighter/whiter)
    img2 = invert(img2) # [0,255], np.uint8. This step to make a larger value means more data points.
    img2 = img_as_float(img2) # [0,1]
    
    # data -> color
    plt.imshow(
        np.concatenate( # [w,h,3] shape
            (
                img1[:,:, None],  # RED
                img2[:,:, None],  # GREEN
                np.zeros((*img1.shape, 1)),
            ),
            axis=-1,
        ),
    )
    # GREEN 0-1-0
    # RED 1-0-0
    # YELLOW 1-1-0
    # BLACK 0-0-0
    plt.title("{} \n vs \n {}\n GREEN: latter data, but no former data \n RED: former data, but no latter data \n YELLOW: both latter and former data \n BLACK: neither latter nor former data".format(f1,f2))
    plt.tight_layout()
    plt.savefig('compare_vis_color.png')
    plt.show()

def add_grid_column(ax, fig_width, fig_height, step, horizontal=False):
    ax.set_frame_on(False)

    # Minor ticks
    ax.set_xticks(np.arange(-0.5, fig_width, step), minor=True)
    if horizontal == True:
        ax.set_yticks(np.arange(-0.5, fig_height, step), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which="minor", color="red", linestyle="-", linewidth=0.3)
    ax.tick_params(which="minor",bottom=False, left=False)

    # hid the ticks but keep the grid
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(which="major",bottom=False, left=False)
    ax.set_ylim((fig_height - 0.5, -0.8)) # NOTE the special order because of image
    ax.set_xlim((-0.5, fig_width-0.3))


def subplt_myplot_random(width,height,dpi,name,anti,downsample,nout,lw,gridStep):
    full_frame(width,height,dpi)
    t = np.linspace(0, 5, num=3000)
    A = 1
    lbda = 1
    omega = 20 * np.pi
    phi = 0
    v = damped_sine(t, A, lbda, omega, phi)
    v_min=min(v)
    v_max=max(v)
    t_min=min(t)
    t_max=max(t)
    # MinMaxDownsampler, MinMaxLTTBDownsampler, M4Downsampler, EveryNthDownsampler, LTTBDownsampler
    if downsample == "MinMaxDownsampler":
        s_ds = MinMaxDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif downsample == "MinMaxLTTBDownsampler":
        s_ds = MinMaxLTTBDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif downsample == "M4Downsampler":
        s_ds = M4Downsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif downsample == "EveryNthDownsampler":
        s_ds = EveryNthDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif downsample == "LTTBDownsampler":
        s_ds = LTTBDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    
    plt.plot(t,v,'-',color='k',linewidth=lw,antialiased=anti,markersize=1)
    plt.xlim(t_min, t_max)
    plt.ylim(v_min, v_max)
    plt.savefig(name+'.png',backend='agg')
    plt.close()
    img=cv2.imread(name+'.png')
    plt.imshow(img)
    ax = plt.gca()
    add_grid_column(ax, height, width, gridStep)

def analyze(name,width,anti,downsample,lw,gridOn):
    # (width,height,dpi,name,anti,downsample,nout,lw,gridOn,gridStep)
    print(name+' rendered on a big 300*300 canvas:')
    myplot_random(300,300,96,name+'_rendered_big',anti,downsample,4*width,lw,gridOn,300/width)
    print(name+' rendered on the target small canvas:')
    myplot_random(width,width,96,name+'_rendered_small_target',anti,downsample,4*width,lw,gridOn,1)
    print('overlay '+ name + ' on the small rendered image:')
    from PIL import Image
    img = Image.open(name+'_rendered_big.png')
    back = Image.open(name+'_rendered_small_target.png')
    back = back.resize(img.size)
    blended_image = Image.blend(img, back, 0.7)
    plt.imshow(blended_image)
    plt.show()
    
def subplt_compare_vis(f1,f2):
    img1 = cv2.imread(f1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # [0,255], np.uint8. Right now a larger value means less data points (thus brighter/whiter)
    img1 = invert(img1) # [0,255], np.uint8. This step to make a larger value means more data points.
    img1 = img_as_float(img1) # [0,1]
    
    img2 = cv2.imread(f2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) # [0,255], np.uint8. Right now a larger value means less data points (thus brighter/whiter)
    img2 = invert(img2) # [0,255], np.uint8. This step to make a larger value means more data points.
    img2 = img_as_float(img2) # [0,1]
    
    plt.imshow(img1 - img2, cmap="gray")
    plt.title("{} \n vs \n {}\n Black: latter data, but no former data.\n White: former data, but no latter data".format(f1,f2),
             fontsize = 10)

def subplt_compare_vis_color(f1,f2,gridStep):
    # color -> data
    img1 = cv2.imread(f1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # [0,255], np.uint8. Right now a larger value means less data points (thus brighter/whiter)
    img1 = invert(img1) # [0,255], np.uint8. This step to make a larger value means more data points.
    img1 = img_as_float(img1) # [0,1]
    
    img2 = cv2.imread(f2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) # [0,255], np.uint8. Right now a larger value means less data points (thus brighter/whiter)
    img2 = invert(img2) # [0,255], np.uint8. This step to make a larger value means more data points.
    img2 = img_as_float(img2) # [0,1]
    
    # data -> color
    plt.imshow(
        np.concatenate( # [w,h,3] shape
            (
                img1[:,:, None],  # RED
                img2[:,:, None],  # GREEN
                np.zeros((*img1.shape, 1)),
            ),
            axis=-1,
        ),
    )
    # GREEN 0-1-0
    # RED 1-0-0
    # YELLOW 1-1-0
    # BLACK 0-0-0
    plt.title("{} \n vs \n {}\n GREEN: latter data, but no former data \n RED: former data, but no latter data \n YELLOW: both latter and former data \n BLACK: neither latter nor former data".format(f1,f2),
              fontsize = 10)
    ax = plt.gca()
    add_grid_column(ax, img1.shape[0], img1.shape[1], gridStep, True)

def subplt_myplot_external(width,height,dpi,name,anti,downsample,nout,lw,gridStep,t,v,gridVertical=True,gridHorizontal=False,isPlot=True):
    full_frame(width,height,dpi)
    v_min=min(v)
    v_max=max(v)
    t_min=min(t)
    t_max=max(t)
    
    returnPreselection=False

    print("===================",downsample,"===================")

    if str(downsample) == "MinMaxDownsampler":
        s_ds = MinMaxDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "MinMaxFPLPDownsampler":
        s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "ILTSParallelDownsampler":
        s_ds = ILTSParallelDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    ############## MinMaxLTTB family ##################
    elif str(downsample) == "MinMaxLTTB1Downsampler":
        returnPreselection=True
        # s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=(nout-2)*1+2)
        s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=nout*1)
        # Select downsampled data
        t_pre = t[s_ds] # note that this is deep copy, so safe when later t is modified
        v_pre = v[s_ds]
        s_ds = LTTBETDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "MinMaxLTTB2Downsampler":
        returnPreselection=True
        # s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=(nout-2)*2+2)
        s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=nout*2)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "MinMaxLTTB4Downsampler":
        returnPreselection=True
        # s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=(nout-2)*4+2)
        s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=nout*4)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "MinMaxLTTB6Downsampler":
        returnPreselection=True
        # s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=(nout-2)*6+2)
        s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=nout*6)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "MinMaxLTTB8Downsampler":
        returnPreselection=True
        # s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=(nout-2)*8+2)
        s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=nout*8)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    ############## MinMaxILTS family ##################
    elif str(downsample) == "MinMaxILTS1Downsampler":
        returnPreselection=True
        # s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=(nout-2)*1+2)
        s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=nout*1)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETFurtherDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "MinMaxILTS2Downsampler":
        returnPreselection=True
        # s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=(nout-2)*2+2)
        s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=nout*2)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETFurtherDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "MinMaxILTS4Downsampler":
        returnPreselection=True
        # s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=(nout-2)*4+2)
        s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=nout*4)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETFurtherDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "MinMaxILTS6Downsampler":
        returnPreselection=True
        # s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=(nout-2)*6+2)
        s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=nout*6)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETFurtherDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "MinMaxILTS8Downsampler":
        returnPreselection=True
        # s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=(nout-2)*8+2)
        s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=nout*8)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETFurtherDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    ############## M4LTTB family ##################
    elif str(downsample) == "M4LTTB1Downsampler":
        returnPreselection=True
        s_ds = M4Downsampler().downsample(t, v, n_out=nout*1)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "M4LTTB2Downsampler":
        returnPreselection=True
        s_ds = M4Downsampler().downsample(t, v, n_out=nout*2)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "M4LTTB4Downsampler":
        returnPreselection=True
        s_ds = M4Downsampler().downsample(t, v, n_out=nout*4)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "M4LTTB6Downsampler":
        returnPreselection=True
        s_ds = M4Downsampler().downsample(t, v, n_out=nout*6)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "M4LTTB8Downsampler":
        returnPreselection=True
        s_ds = M4Downsampler().downsample(t, v, n_out=nout*8)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    ############## M4ILTS family ##################
    elif str(downsample) == "M4ILTS1Downsampler":
        returnPreselection=True
        s_ds = M4Downsampler().downsample(t, v, n_out=nout*1)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETFurtherDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "M4ILTS2Downsampler":
        returnPreselection=True
        s_ds = M4Downsampler().downsample(t, v, n_out=nout*2)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETFurtherDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "M4ILTS4Downsampler":
        returnPreselection=True
        s_ds = M4Downsampler().downsample(t, v, n_out=nout*4)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETFurtherDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "M4ILTS6Downsampler":
        returnPreselection=True
        s_ds = M4Downsampler().downsample(t, v, n_out=nout*6)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETFurtherDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    elif str(downsample) == "M4ILTS8Downsampler":
        returnPreselection=True
        s_ds = M4Downsampler().downsample(t, v, n_out=nout*8)
        # Select downsampled data
        t_pre = t[s_ds]
        v_pre = v[s_ds]
        s_ds = LTTBETFurtherDownsampler().downsample(t_pre, v_pre, n_out=nout)
        # Select downsampled data
        t = t_pre[s_ds]
        v = v_pre[s_ds]
    ############## family end ##################
    elif str(downsample) == "MinMaxGapDownsampler":
        s_ds = MinMaxGapDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
#     elif downsample == "MinMaxLTTBDownsampler":
#         s_ds = MinMaxLTTBDownsampler().downsample(t, v, n_out=nout)
#         # Select downsampled data
#         t = t[s_ds]
#         v = v[s_ds]
    elif str(downsample) == "M4Downsampler":
        s_ds = M4Downsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
#     elif str(downsample) == "EveryNthDownsampler":
#         s_ds = EveryNthDownsampler().downsample(t, v, n_out=nout)
#         # Select downsampled data
#         t = t[s_ds]
#         v = v[s_ds]
    elif str(downsample) == "LTOBETDownsampler":
        s_ds = LTOBETDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    # elif str(downsample) == "LLBDownsampler":
    #     s_ds = LLBDownsampler().downsample(t, v, n_out=nout)
    #     # Select downsampled data
    #     t = t[s_ds]
    #     v = v[s_ds]
    elif str(downsample) == "LTTBETDownsampler":
        s_ds = LTTBETDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "LTTBETFurtherDownsampler":
        s_ds = LTTBETFurtherDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "LTSDownsampler":
        s_ds = LTSDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "LTTBETNewDownsampler":
        s_ds = LTTBETNewDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "LTTBETGapDownsampler":
        s_ds = LTTBETGapDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "LTOBETGapDownsampler":
        s_ds = LTOBETGapDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "LTTBDownsampler":
        s_ds = LTTBDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "LTOBDownsampler":
        s_ds = LTOBDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "LTDDownsampler":
        s_ds = LTDDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "LTDOBDownsampler":
        s_ds = LTDOBDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "visval":
        points=[]
        for x, y in zip(t,v):
            points.append((x,y))
        simplifier = VWSimplifier(points)
        tmp=simplifier.from_number(nout)
        t=tmp[:,0]
        v=tmp[:,1]
    elif str(downsample) == "rdp":
        points=[]
        for x, y in zip(t,v):
            points.append((x,y))
        tmp=np.array(rdp(points, epsilon=nout))
        t=tmp[:,0]
        v=tmp[:,1]
    else:
        downsample="original"

    print("number of output points=", len(t))
        
    plt.plot(t,v,'-',color='k',linewidth=lw,antialiased=anti,markersize=1)
    plt.xlim(t_min, t_max)
    plt.ylim(v_min, v_max)
    
    plt.savefig(name+'.png',backend='agg')
    plt.close()

    if isPlot:
        img=cv2.imread(name+'.png')
        plt.imshow(img)
        ax = plt.gca()
        if gridVertical==True:
            add_grid_column(ax, width, height, gridStep, gridHorizontal)
    
    if returnPreselection==False:
        return t,v
    else:
        return t,v,t_pre,v_pre

def rust_myplot_external(w,h,downsample,nout,t,v,outputDir,name,tmin,tmax,vmin,vmax):
    outputCsvPath="{}/ts-{}-{}.csv".format(outputDir,name,nout)

    if str(downsample) == "MinMaxDownsampler":
        s_ds = MinMaxDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "MinMaxFPLPDownsampler":
        s_ds = MinMaxFPLPDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "MinMaxGapDownsampler":
        s_ds = MinMaxGapDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "M4Downsampler":
        s_ds = M4Downsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "LTOBETDownsampler":
        s_ds = LTOBETDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "LTTBETDownsampler":
        s_ds = LTTBETDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "LTTBETFurtherDownsampler":
        s_ds = LTTBETFurtherDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "LTTBETNewDownsampler":
        s_ds = LTTBETNewDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "LTTBETGapDownsampler":
        s_ds = LTTBETGapDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "LTOBETGapDownsampler":
        s_ds = LTOBETGapDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "LTTBDownsampler":
        s_ds = LTTBDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "LTOBDownsampler":
        s_ds = LTOBDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "LTDDownsampler":
        s_ds = LTDDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "LTDOBDownsampler":
        s_ds = LTDOBDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t = t[s_ds]
        v = v[s_ds]
    elif str(downsample) == "visval":
        points=[]
        for x, y in zip(t,v):
            points.append((x,y))
        simplifier = VWSimplifier(points)
        tmp=simplifier.from_number(nout)
        t=tmp[:,0]
        v=tmp[:,1]
    elif str(downsample) == "rdp":
        points=[]
        for x, y in zip(t,v):
            points.append((x,y))
        tmp=np.array(rdp(points, epsilon=nout))
        t=tmp[:,0]
        v=tmp[:,1]
    else:
        downsample="original"
        outputCsvPath="{}/ts-{}.csv".format(outputDir,name)

    print(downsample,len(t))
    print(outputCsvPath)

    # scale t -> x
    x=(t-tmin)/(tmax-tmin)*w
    x=np.floor(x)
    x[x == w] = w-1

    # scale v -> y
    y=(v-vmin)/(vmax-vmin)*h
    y=np.floor(y)
    y[y == h] = h-1

    # save to csv
    df = pd.DataFrame({'time':x,'value':y}) # output csv has header
    df.to_csv(outputCsvPath, sep=',',index=False)

    return t,v

def subplt_myplot_external_compare(width,height,dpi,name,anti,downsample,nout,lw,gridStep,t,v,gridVertical=True,gridHorizontal=False):
    full_frame(width,height,dpi)
    v_min=min(v)
    v_max=max(v)
    t_min=min(t)
    t_max=max(t)
    # MinMaxDownsampler, MinMaxLTTBDownsampler, M4Downsampler, EveryNthDownsampler, LTTBDownsampler
    if str(downsample) == "MinMaxDownsampler":
        s_ds = MinMaxDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t2 = t[s_ds]
        v2 = v[s_ds]
#     elif downsample == "MinMaxLTTBDownsampler":
#         s_ds = MinMaxLTTBDownsampler().downsample(t, v, n_out=nout)
#         # Select downsampled data
#         t = t[s_ds]
#         v = v[s_ds]
    elif str(downsample) == "M4Downsampler":
        s_ds = M4Downsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t2 = t[s_ds]
        v2 = v[s_ds]
    elif str(downsample) == "EveryNthDownsampler":
        s_ds = EveryNthDownsampler().downsample(t, v, n_out=nout)
        # Select downsampled data
        t2 = t[s_ds]
        v2 = v[s_ds]
    elif str(downsample) == "LTTBDownsampler":
        s_ds = LTTBDownsampler().downsample(t, v, n_out=nout+2) # +2 because LTTB keeps the first and last points
        # Select downsampled data
        t2 = t[s_ds]
        v2 = v[s_ds]
        # remove the first and last points kept by LTTB so that the number of output downsampled points is still nout
        t2=t2[1:-1]
        v2=v2[1:-1]
    elif str(downsample) == "visval":
        points=[]
        for x, y in zip(t,v):
            points.append((x,y))
        simplifier = vw.Simplifier(points)
        tmp=simplifier.simplify(number=nout)
        t2=tmp[:,0]
        v2=tmp[:,1]
    elif str(downsample) == "rdp":
        points=[]
        for x, y in zip(t,v):
            points.append((x,y))
        tmp=np.array(rdp(points, epsilon=nout))
        t2=tmp[:,0]
        v2=tmp[:,1]
        
    print(len(t))

    plt.plot(t2,v2,color='k',linewidth=lw,antialiased=anti,markersize=1,label=str(downsample))
    plt.plot(t,v,color='r',linewidth=lw,antialiased=anti,markersize=1,label='original')
    plt.xlim(t_min, t_max)
    plt.ylim(v_min, v_max)
    plt.savefig(name+'.png',backend='agg')
    plt.close()
    img=cv2.imread(name+'.png')
    plt.imshow(img)
    ax = plt.gca()
    if gridVertical==True:
        add_grid_column(ax, width, height, gridStep, gridHorizontal)

def read_ucr(filename,col=0): # col starting from 0
    df=pd.read_csv(filename, header=None, delimiter=r"\s+")
    df2=df.T
    v=df2.iloc[1:,col]
    v=v.to_numpy()
    t = np.linspace(0, 10, num=len(v))
    return t,v

def my_get_bin_idxs(x: np.ndarray, nb_bins: int) -> np.ndarray:
    # Thanks to the `linspace` the data is evenly distributed over the index-range
    # The searchsorted function returns the index positions
    bins = np.searchsorted(x, np.linspace(x[0], x[-1], nb_bins + 1), side="right")
    bins[0] = 0
    bins[-1] = len(x)
    return np.unique(bins)