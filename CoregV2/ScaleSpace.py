"""
Build Scale Space
"""

# Imports
import numpy as np
import cv2
import time
import math
import scipy.ndimage
import matplotlib.pyplot as plt

from .utils import *

# Kernel Functions
def GaussianKernel(size, sigma):
    '''
    Gaussian Kernel
    '''
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.float64(np.exp(-((x**2 + y**2)/(2.0*sigma**2))))
    return g/g.sum()

def ScaledGaussianKernel(scale, sigma=None):
    '''
    Scaled Gaussian Kernel
    '''
    gaussian_sigma = math.sqrt(2)*scale
    width = round(3*gaussian_sigma)
    width_windows = int(2*width+1)
    if sigma is None:
        sigma = gaussian_sigma
    G = GaussianKernel(width_windows, sigma)
    return G

# Diffusion Functions
def NonLinearDiffusion_g1(J_sigma, k):
    '''
    G1 Non-Linear Diffusion Function
    '''
    term1 = (J_sigma / k) ** 2
    g1_J_sigma = np.exp(-term1)
    return g1_J_sigma

def NonLinearDiffusion_g2(J_sigma, k):
    '''
    G2 Non-Linear Diffusion Function
    '''
    g2_J_sigma = 1 / (1 + ((J_sigma**2) / (k**2)))
    return g2_J_sigma

def NonLinearDiffusion_g3(J_sigma, k):
    '''
    G3 Non-Linear Diffusion Function
    '''
    term1 = 3.315 / ((J_sigma / k)**8)
    term2 = np.exp(-term1)
    g3_J_sigma = 1 - term2
    return g3_J_sigma

def ThomasAlgorithm(a, b, Ld):
    '''
    Thomas Algorithm
    '''
    n = a.shape[0]
    m = a
    l = np.zeros(b.shape)
    y = Ld

    # Forward Substituition
    # L*y = d for y
    for k in range(1, n):
        l[k-1] = b[k-1] / m[k-1]
        m[k] = a[k] - l[k-1] * b[k-1]
        y[k] = Ld[k] - l[k-1] * y[k-1]

    # Backward Substituition
    # U*x = y for x
    x = np.zeros(a.shape)
    x[n-1] = y[n-1] / m[n-1]
    for k in range(n-2, -1, -1):
        x[k] = (y[k] - b[k] * x[k+1]) / m[k]

    return x

def AOS_Rows(Ldprev, c, stepsize):
    '''
    AOS_Rows
    '''
    qr_rows = Ldprev.shape[0] - 1
    c_1shift = np.zeros(c.shape)
    c_1shift[:-1] = c[1:]
    qr = c + c_1shift
    py = np.zeros(qr.shape)
    py[0] = qr[0]
    py[-1] = qr[-1]
    for k in range(1, py.shape[0]-1):
        py[k] = qr[k-1] + qr[k]
    
    ay = 1.0 + stepsize * py
    by = -stepsize * qr

    Lty = ThomasAlgorithm(ay, by, Ldprev)

    return Lty

def AOS_Columns(Ldprev, c, stepsize):
    '''
    AOS_Cols
    '''
    qc_cols = Ldprev.shape[1]
    c_1shift = np.zeros(c.shape)
    c_1shift[:, :-1] = c[:, 1:]
    qc = c + c_1shift
    px = np.zeros(qc.shape)
    px[:, 0] = qc[:, 0]
    px[:, -1] = qc[:, -1]
    for k in range(1, px.shape[1]-1):
        px[:, k] = qc[:, k-1] + qc[:, k]
    
    ax = 1.0 + stepsize * px.T
    bx = -stepsize * qc.T

    Ltx = ThomasAlgorithm(ax, bx, Ldprev.T)

    return Ltx

def AOS_Step_Scalar(Ldprev, c, stepsize):
    Lty = AOS_Rows(Ldprev, c, stepsize)
    Ltx = AOS_Columns(Ldprev, c, stepsize)
    Ld = 0.5 * (Lty + Ltx.T)
    return Ld

def CalculateDiffusion(J_sigma, scale, percentile=70, diffusion="g2"):  
    '''
    Calculate Diffusion
    '''
    # Get Gradients of J_sigma => |Grad J_sigma|
    # if imgType == "SAR":
    #     Gx_sigma, Gy_sigma = ROEWA(J_sigma, scale, ratio)
    # else:
    #     Gx_sigma, Gy_sigma = Sobel(J_sigma)
    Gx_sigma = cv2.Sobel(J_sigma, 6, 1, 0, ksize=3)
    Gy_sigma = cv2.Sobel(J_sigma, 6, 0, 1, ksize=3)
    J_sigma_grad = np.sqrt(np.square(Gx_sigma)+np.square(Gy_sigma))

    # Contrast Ratio k = 70th percentile of the gradient histogram of smoothed J
    nbins = 100
    hmax = np.max(J_sigma_grad[1:-1, 1:-1]) # Ignore borders
    binMap = np.floor((J_sigma_grad[1:-1, 1:-1] / hmax) * nbins)
    binMap[binMap == nbins] = nbins-1
    NonZeroMask = J_sigma_grad[1:-1, 1:-1] != 0.0
    n_points = np.count_nonzero(NonZeroMask)
    bins = binMap[NonZeroMask].ravel()

    hist = np.zeros(nbins, dtype=int)
    for b in bins:
        hist[int(b)] += 1
    n_threshold = int(n_points * percentile/100)

    n_elements = 0
    bi = 0
    while (bi < nbins):
        n_elements += hist[bi]
        if n_elements >= n_threshold:
            break
        bi += 1
    bi += 1 # 1 extra increment

    k = 0.03
    if n_elements >= n_threshold:
        k = hmax * (bi / nbins)

    # Calculate g2
    diffFunc = NonLinearDiffusion_g2
    if diffusion == "g1":
        diffFunc = NonLinearDiffusion_g1
    elif diffusion == "g3":
        diffFunc = NonLinearDiffusion_g3

    g_J_sigma = diffFunc(J_sigma_grad, k)

    return g_J_sigma, k

# Gradient Functions
def ROEWA(J, scale, ratio):
    '''
    ROEWA Gradient
    '''
    radius = int(round(2*scale))
    j = list(range(-radius,radius+1,1))
    k = list(range(-radius,radius+1,1))
    xarry,yarry = np.meshgrid(j,k)
    W = np.exp(-(np.abs(xarry)+np.abs(yarry))/scale)
    W34 = np.zeros((2*radius+1,2*radius+1),dtype=float)
    W12 = np.zeros((2*radius+1,2*radius+1),dtype=float)
    W14 = np.zeros((2*radius+1,2*radius+1),dtype=float)
    W23 = np.zeros((2*radius+1,2*radius+1),dtype=float)
    
    W34[radius+1:2*radius+1,:] = W[radius+1:2*radius+1,:]
    W12[0:radius,:] = W[0:radius,:]
    W14[:,radius+1:2*radius+1] = W[:,radius+1:2*radius+1]
    W23[:,0:radius] = W[:,0:radius]

    M34 = scipy.ndimage.correlate(J, W34, mode='nearest')
    M12 = scipy.ndimage.correlate(J, W12, mode='nearest')
    M14 = scipy.ndimage.correlate(J, W14, mode='nearest')
    M23 = scipy.ndimage.correlate(J, W23, mode='nearest')
    
    Gx = np.log(M14/M23)
    Gy = np.log(M34/M12)
    
    Gx[np.where(np.imag(Gx))] = np.abs(Gx[np.where(np.imag(Gx))])
    Gy[np.where(np.imag(Gy))] = np.abs(Gy[np.where(np.imag(Gy))])
    Gx[np.where(np.isfinite(Gx)==0)] = 0
    Gy[np.where(np.isfinite(Gy)==0)] = 0

    return Gx, Gy

def Sobel(J):
    '''
    Sobel Gradient
    '''

    Gx = cv2.Sobel(J, 6, 1, 0, ksize=3)
    Gy = cv2.Sobel(J, 6, 0, 1, ksize=3)

    return Gx, Gy

# Scale Space Functions
def EvolutionTime(sigma):
    '''
    Evolution Time
    '''
    return 0.5 * (sigma**2)

def CalculateHarrisResponse(Gx, Gy, scale, d):
    '''
    Calculate Harris Response
    '''
    Csh_11 = (scale**2) * (Gx ** 2)
    Csh_12 = (scale**2) * Gx*Gy
    Csh_22 = (scale**2) * (Gy ** 2)
    
    gaussian_sigma = (2 ** (0.5)) * scale
    width = round(3*gaussian_sigma)
    width_windows = int(2*width+1)
    W_gaussian = GaussianKernel(width_windows, gaussian_sigma)
    
    l = list(range(0,width_windows,1))
    m = list(range(0,width_windows,1))
    a,b = np.meshgrid(l,m)
    index0,index1 = np.where((((a-width)**2) - 1) + ((b - width - 1)**2) > (width**2))
    W_gaussian[index0, index1] = 0
    
    Csh_11 = scipy.ndimage.correlate(Csh_11, W_gaussian, mode='nearest')
    Csh_12 = scipy.ndimage.correlate(Csh_12, W_gaussian, mode='nearest')
    Csh_21 = Csh_12
    Csh_22 = scipy.ndimage.correlate(Csh_22, W_gaussian, mode='nearest')
    
    HarrisResponse = (Csh_11*Csh_22 - Csh_21*Csh_12) - d*((Csh_11 + Csh_22)**2)

    return HarrisResponse

def CalculateLoG(J, scale, ddepth=2, logKernelSize=3):
    '''
    Calculate LoG
    '''
    LoG = scipy.ndimage.gaussian_laplace(J, scale)
    return LoG

def BuildScaleSpace(I, sigma_0, S, ratio, d, GKernelSize=3, diffusion="g2", imgType="SAR", options=DEFAULT_OPTIONS):
    '''
    Build Scale Space
    '''
    pathFormat = options["path"]

    # Initial Layer
    M,N = I.shape
    sigma = sigma_0

    # Convolve with Gaussian with SD sigma_0
    J = cv2.GaussianBlur(I, GKernelSize, sigma_0)

    ScaleSpaceData = {
        "Js": [],
        "Gxs": [],
        "Gys": [],
        "gradients": [],
        "angles": [],
        "HarrisResponses": [],
        "LoGs": []
    }

    As = []
    # For other layers
    for i in tqdm(range(S), disable=not options["verbose_main"]):
        scale = sigma_0 * (ratio**(i))
        # Compute the gradient using ROEWA Operator for SAR
        if imgType == "SAR":
            Gx, Gy = ROEWA(J, scale, ratio)
        # Compute the gradient using Sobel Operator for Optical
        else:
            Gx, Gy = Sobel(J)
        gradients = np.sqrt(np.square(Gx)+np.square(Gy))
        angles = np.arctan2(Gy, Gx)

        ScaleSpaceData["Js"].append(J)
        ScaleSpaceData["Gxs"].append(Gx)
        ScaleSpaceData["Gys"].append(Gy)
        ScaleSpaceData["gradients"].append(gradients)
        ScaleSpaceData["angles"].append(angles)

        # Calculate Harris Response
        HarrisResponse = CalculateHarrisResponse(Gx, Gy, scale, d)
        ScaleSpaceData["HarrisResponses"].append(HarrisResponse)

        # Calculate LoG
        LoG = CalculateLoG(J, scale, GKernelSize)
        ScaleSpaceData["LoGs"].append(LoG)

        # Compute the next layer
        sigma = sigma_0 * (ratio**(i+1))
        
        # Smooth J to get J_sigma
        J_sigma = cv2.GaussianBlur(J, GKernelSize, scale)
        g_J_sigma, kPercentile = CalculateDiffusion(J_sigma, scale, percentile=70, diffusion=diffusion)
        As.append(g_J_sigma)

        # next layer J
        td = EvolutionTime(sigma) - EvolutionTime(scale)

        J_new = AOS_Step_Scalar(np.copy(J), c=np.copy(g_J_sigma), stepsize=td)

        if options["verbose"]: 
            print("ITERATION", i)
            print("J", J.min(), J.max())
            print("J_sigma", J_sigma.min(), J_sigma.max())
            print("kPercentile", kPercentile)
            print("g_J_sigma", g_J_sigma.min(), g_J_sigma.max())
            print("J_new", J_new.min(), J_new.max())
            print("HarrisResponse", HarrisResponse.min(), HarrisResponse.max())
            print("LoG", LoG.min(), LoG.max())

        if options["plot"] or options["save"]:
            # Plot J Histogram
            options["path"] = pathFormat.format("ScaleSpace_Scalewise_Intermediate_J_new_Histogram_" + str(i+1))
            title = "Scale " + str(i+1) + " J_new Histogram"
            PlotImageHistogram(J_new, 1000, title, options)

            # Show Images
            options["path"] = pathFormat.format("ScaleSpace_Scalewise_Intermediate_" + str(i+1))
            ShowImages_Grid([NORMALISERS["MinMax"](J), NORMALISERS["MinMax"](J_sigma), NORMALISERS["MinMax"](g_J_sigma), 
                                NORMALISERS["MinMax"](J_new), NORMALISERS["MinMax"](HarrisResponse), NORMALISERS["MinMax"](LoG)], 
                                nCols=3, 
                                titles=["J " + str(i+1), "J_sigma", "g_J_sigma", "J_new", "Harris Response", "LoG Response"], 
                                figsize=(10, 10), gap=(0.25, 0.25),
                                options=options)
        
        J = J_new
        # J = NORMALISERS["MinMax"](J_new)

    for k in ScaleSpaceData.keys():
        ScaleSpaceData[k] = np.array(ScaleSpaceData[k])
    
    return ScaleSpaceData

# Display Functions
def DisplayGradients(ScaleSpaceData, options=DEFAULT_OPTIONS):
    '''
    Display Gradients
    '''
    options = dict(options)

    pathFormat = options["path"]
    for i in range(ScaleSpaceData["gradients"].shape[0]):
        # Show Images
        options["path"] = pathFormat.format("ScaleSpace_Scalewise_Responses_" + str(i+1))
        ShowImages_Grid([NORMALISERS["MinMax"](ScaleSpaceData["Js"][i]), NORMALISERS["MinMax"](ScaleSpaceData["gradients"][i]), 
                        NORMALISERS["MinMax"](ScaleSpaceData["HarrisResponses"][i]), NORMALISERS["MinMax"](ScaleSpaceData["LoGs"][i])], 
                        nCols=5, 
                        titles=["J " + str(i+1), "Grads", "Harris", "LoG"], 
                        figsize=(10, 10), gap=(0.25, 0.25), 
                        options=options)
        
def DisplayGradientHistograms(ScaleSpaceData, options=DEFAULT_OPTIONS):
    '''
    Display Gradient Histograms
    '''
    options = dict(options)

    titles = []
    for i in range(ScaleSpaceData["gradients"].shape[0]):
        titles.append("Grads Layer " + str(i+1))

    for i in range(ScaleSpaceData["gradients"].shape[0]):
        # Initial Clear
        plt.clf()

        grads = NORMALISERS["MinMax"](ScaleSpaceData["gradients"][i])
        # Show Gradient Image
        plt.subplot(1, 2, 1)
        plt.imshow(grads)
        plt.title("Grads Layer " + str(i+1))

        # Show Gradient Histogram
        plt.subplot(1, 2, 2)
        plt.hist(grads.ravel(), bins=100)
        plt.title("Grads Histogram")

        # Plot or Save
        if options["save"]:
            plt.savefig(options["path"].format("ScaleSpace_Scalewise_GradHist_" + str(i+1)), bbox_inches='tight')
        if options["plot"]:
            plt.show()

# Runner Functions
def HarrisLaplace_ConstructScaleSpace(I, ScaleSpaceParams, imgType="SAR", options=DEFAULT_OPTIONS):
    '''
    Construct Harris-Laplace Scale Space
    '''
    sigma = ScaleSpaceParams["sigma"]
    S = ScaleSpaceParams["S"]
    ratio = ScaleSpaceParams["ratio"]
    d = ScaleSpaceParams["d"]
    GKernelSize = ScaleSpaceParams["GKernelSize"]
    diffusion = ScaleSpaceParams["diffusion"]

    start_time_sh = time.time()

    options = dict(options)
    ScaleSpaceData = BuildScaleSpace(I, sigma, S, ratio, d, GKernelSize, diffusion, imgType, options)

    end_time_sh = time.time()

    if options["verbose"]:
        print("Time for Compute Gradients:", FormatTime(end_time_sh - start_time_sh, 4))
        print("Response Gradients:", ScaleSpaceData["gradients"].shape)
        print("Response Angles:", ScaleSpaceData["angles"].shape)

    return ScaleSpaceData

# Driver Code
# Params

# Params

# RunCode
print("Reloaded Scale Space Construction!")