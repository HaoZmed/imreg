import numpy as np
from scipy.interpolate import interp2d
# from scipy.ndimage import map_coordinates
# from scipy.interpolate import RectBivariateSpline

# Checked
def my_interp2(I, u):
    """ Calculate the deformation of I by u
    The out-of-range value are set as the boundary value
    Input:
        I: 2D image - h x w ndarray
        u: Deformation field - h x w x 2
    Output:
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



# Checked
def corr2(X, Y):
    """ Calculate the 2D correlation coefficient between two given 2D images X and Y.
    Input:
        X: 2D image - h x w ndarray
        Y: 2D image - h x w ndarray
    Output:
        corr: correlation coefficient - scalar
    """
    X = X - np.mean(X)
    Y = Y - np.mean(Y)
    nom = np.sum(X * Y)
    denom = np.sqrt(np.sum(X*X) * np.sum(Y*Y))
    return nom/(denom + 1e-9)



# Checked
def gradient(I):
    """ Calculate the gradient of I at each pixel, the boundary condition is Neumann.
    x-direction : rightward
    y-direction : downward
    Input:
        I: 2D image - h x w ndarray
    Output:
        dx_I, dy_I: gradient of I - h x w x 2 ndarray
    """
    h, w = I.shape
    ext_I = np.pad(I, ((1, 1), (1, 1)), 'edge')
    [dy_I, dx_I] = np.gradient(ext_I)
    return dy_I[1:h+1, 1:w+1], dx_I[1:h+1, 1:w+1]



# Checked
def evol(u, v, dS, interp_S, T, sigma, params):
    """ Flow evolution of deformation field u for 2D deformable registration
    solve one iteration for
        du/dt = Laplace(u) - c(S(x+u) - T(x+v))*DS(x+u) - wC*D(ICC(u, ui))
    where c = wF/(2*sigma^2).
    The semi-implicite discretized form is
        (w - u)/d = Laplace(w) - c(S(x+u) - T(x+v))*DS(x+u) - wC*D(ICC(u, ui))
    where w is the updated u.
    **Note**: the update schemes for u, v, ui, vi are the same.
    Input:
        u        : the deformation field to update
        v        : the other deformation field
        dS       : the gradient of the 2D image
        interp_S : the deformed 2D image by u
        T        : the other 2D image
        sigma    : the standard deviation
        params   : hyper-parameters
    Output:
        w        : the updated version of u
    """
        
    c = params.weight_fitting / (2 * sigma**2)
    w = np.zeros(u.shape)

    # Evolution using AOS
    R = interp_S - T
    interp_dyS = my_interp2(dS[:, :, 0], u)
    interp_dxS = my_interp2(dS[:, :, 1], u)

    interp_v1 = my_interp2(v[:, :, 0], u)
    interp_v2 = my_interp2(v[:, :, 1], u)

    [dyv1, dxv1] = gradient(v[:, :, 0])
    [dyv2, dxv2] = gradient(v[:, :, 1])

    interp_dxv1 = my_interp2(dxv1, u)
    interp_dxv2 = my_interp2(dxv2, u)
    interp_dyv1 = my_interp2(dyv1, u)
    interp_dyv2 = my_interp2(dyv2, u) 

    z1 = u[:, :, 0] + interp_v1
    z2 = u[:, :, 1] + interp_v2

    icc1 = (1 + interp_dyv1) * z1 + interp_dyv2 * z2
    icc2 = interp_dxv1 * z1 + (1 + interp_dxv2) * z2

    w[:, :, 0] = laplace_evol_2D(u[:, :, 0],
        -c*R*interp_dyS - params.weight_corr*icc1, params.step_size)
    w[:, :, 1] = laplace_evol_2D(u[:, :, 1],
        -c*R*interp_dxS - params.weight_corr*icc2, params.step_size)

    return w



# Checked
def laplace_evol_2D(u, f, d):
    """ Evolution function in 2D based on Additive Operator Splitting (AOS)
        Scheme. Thomas tridiagonal solver.
            du/dt = Lap(u) +f 
    Input:
        u : deformation field (one direction) - h x w ndarray
        f :                                   - h x w ndarray
        d : scalar
    Output:
        u : deformation field (one direction) - h x w ndarray
    """
    h, w = u.shape
    N = u.size
    r = (u + d * f)

    xa = -2 * d * np.concatenate((np.ones((h - 1, w)),
                                  np.zeros((1, w))),
                                 axis=0).T.reshape((N, 1))
    xb = -2 * d * np.concatenate((np.zeros((1, w)),
                                  np.ones((h - 1, w))),
                                 axis=0).T.reshape((N, 1))

    ya = -2 * d * np.concatenate((np.ones((w - 1, h)),
                                  np.zeros((1, h))),
                                 axis=0).T.reshape((N, 1))
    yb = -2 * d * np.concatenate((np.zeros((1, h)),
                                  np.ones((w - 1, h))),
                                 axis=0).T.reshape((N, 1))
    
    ux = thomas(xa, 1 - xa - xb, xb, r.T.reshape((N, 1))).reshape((w, h)).T
    uy = thomas(ya, 1 - ya - yb, yb, r.reshape((N, 1))).reshape((h, w)).T
    
    u = (ux + uy.T)/2
    return u





# Checked
def thomas(ud, d, ld, r):
    """ Solve the linear system Ax = b, where A is a tridiagonal matrix
    Inputs:
        ud: upper diagonal of A, ud(n - 1) = 0 - n x 1 array
        d : diagonal of A                      - n x 1 array   
        ld: lower diagonal of A, ld(0) = 0     - n x 1 array
        r : right-hand-side of equation        - n x 1 array
    Outputs:
        x: the solution
    """

    N = d.size
    ud[0] = ud[0] / d[0]
    r[0] = r[0] / d[0]

    for i in range(1, N - 1):
        denom = d[i] - ld[i] * ud[i - 1]
        if denom == 0:
            sys.exit("Division with zero error.")
        ud[i] = ud[i] / denom
        r[i] = (r[i] - ld[i] * r[i - 1]) / denom
    
    r[N - 1] = (r[N - 1] - ld[N - 1] * r[N - 2]) /\
        (d[N - 1] - ld[N - 1] * ud[N - 2])
    x = np.zeros((N, 1))
    x[N - 1] = r[N - 1]
    for i in range(N - 2, -1, -1):
        x[i] = r[i] - ud[i] * x[i + 1]
    
    return x.reshape((N, 1))
