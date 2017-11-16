import numpy as np
from scipy.interpolate import interp2d
from scipy.ndimage import map_coordinates
from scipy.interpolate import RectBivariateSpline

def my_interp2(I, u):
    """ Calculate the deformation of I by u.
    The out-of-range value are set as the boundary value.
    Input:
        I: 2D image - h x w matrix
        u: Deformation field - h x w x 2
    Return:
        iterp_I: Deformed I by u - h x w matrix
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