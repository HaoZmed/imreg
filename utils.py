import numpy as np
from scipy.interpolate import interp2d
from scipy.ndimage import map_coordinates
from scipy.interpolate import RectBivariateSpline

def my_interp2(I, u):
    """ Calculate the deformation of I by u
    The out-of-range value are set as the boundary value
    Input:
        I: 2D image - h x w ndarray
        u: Deformation field - h x w x 2
    Return:
        iterp_I: Deformed I by u - h x w ndarray
    """
    [h, w] = I.shape

    y = np.linspace(0, h - 1, h)
    x = np.linspace(0, w - 1, w)
    
    xv, yv = np.meshgrid(x, y)
    xi = (xv + u[:, :, 1]).ravel()
    yi = (yv + u[:, :, 0]).ravel()
    
    xi = np.maximum(np.minimum(xi, w), 0)
    yi = np.maximum(np.minimum(yi, h), 0)

    f  = interp2d(x, y, I)
    iterp_I = np.array([f(xq, yq) for xq, yq in zip(xi, yi)]).reshape(I.shape)
    return iterp_I



def corr2(X, Y):
    """ Calculate the 2D correlation coefficient between two given 2D images X and Y.
    Input:
        X: 2D image - h x w ndarray
        Y: 2D image - h x w ndarray
    Return:
        corr: correlation coefficient - scalar
    """
    X = X - np.mean(X)
    Y = Y - np.mean(Y)
    nom = np.sum(X * Y)
    denom = np.sqrt(np.sum(X*X) * np.sum(Y*Y))
    return nom/denom


def gradient(I):
    """ Calculate the gradient of I at each pixel, the boundary condition is Neumann.
    Input:
        I: 2D image - h x w ndarray
    Output:
        dI: gradient of I - h x w x 2 ndarray
    """
    h, w = I.shape
    ext_I = np.zeros((h+2, w+2))
    ext_I[1:h, 1:w] = I
    ext_I[1:h, 0] = I[1:h, 1]
    ext_I[1:h, w] = I[1:h, w-1]
    ext_I[0, 1:w] = I[1, 1:w]
    ext_I[h, 1:w] = I[h-1, 1:w]

