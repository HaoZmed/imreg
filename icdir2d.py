from utils import *
import numpy as np 

class Parameters(object):
    def __init__(self, max_iter,
                 max_corr,
                 lb_sigma,
                 weight_fitting,
                 step_size):
        self.max_iter = max_iter
        self.max_corr = max_corr
        self.lb_sigma = lb_sigma
        self.weight_fitting = weight_fitting
        self.step_size = step_size

class Logs(object):
    def __init__(self, max_iter):
        self.corr = np.zeros((max_iter, ))
        self.obj  = np.zeros((max_iter, ))
        self.err  = np.zeros((max_iter, ))       



def imreg_deformation(S, T, max_iter=200,
                      max_corr=0.995,
                      lb_sigma=0.1,
                      weight_fitting=1,
                      step_size=1e-3,
                      silent=True,
                      print_frequency=5):
    """ Calculate deformation fields that maps the source image
    and the target image to each other. The function will calculate
    the deformation fields such that:
        S(x + u(x)) = T(x + v(x)) 
    by minimizing:
        E(u, v) =  
    Inputs:
        S : source image - 2D matrix (m x n)
        T : taret image - 2D matrix (m x n)
        max_iter: maximum iterations
        max_corr: maximum coefficient correlation between S(x+u) and T(y+v),
                  which is used as the stopping criterion
        weight_fitting: weights for MLE data fitting term
        step_size: step size for evolution
    Outputs:
        u : deformation field - 3D matrix (m x n x 2)
        v : deformation field - 3D matrix (m x n x 2)
    Reference:
        Xiaojing Ye
    """

    # Setting up hyper-parameters
    params = Parameters(max_iter, max_corr, lb_sigma, weight_fitting, step_size)
    log    = Logs(max_iter)
    N = S.size

    # Initialization of the algorithm
    [dx_S, dy_S] = gradient(S);
    [dx_T, dy_T] = gradient(T);
    
    dS = np.concatenate([dx_S, dy_S], axis=2)
    dT = np.concatenate([dx_T, dy_T], axis=2)
    
    null = np.zeros((S.shape))
    dnull = np.concatenate([null, null], axis=2)

    u  = np.zeros((S.shape + (2,)))
    v  = np.zeros((S.shape + (2,)))
    ui = np.zeros((S.shape + (2,)))
    vi = np.zeros((S.shape + (2,)))

    iterp_S, iterp_T = np.copy(S), np.copy(T)
    sigma = 1

    curr_corr = 0
    iter = 0

    # Main loop
    while iter < max_iter and curr_corr < max_corr:
        
        # Update deformation fields
        u  = evol(u, ui, dS, iterp_S, iterp_T, sigma, params)
        ui = evol(ui, u, dnull, null, null, sigma, params)

        v  = evol(v, vi, dS, iterp_T, iterp_S, sigma, params)
        vi = evol(vi, v, dnull, null, null, sigma, params)
        
        # Update deformed images
        iterp_S = my_interp2(S, u)
        iterp_T = my_interp2(T, v)

        # Update sigma
        sigma = np.linalg.norm(iterp_S - iterp_T) / np.sqrt(N)

        curr_corr = corr2(iterp_S, iterp_T)

        if sigma < lb_sigma:
            sigma = lb_sigma
        
        log.corr[iter] = curr_corr
        log.obj[iter] = calculate_objective(u, sigma, params)

        if not silent and print_frequency % iter == 0:
            print("iter = {}, corr = {}".format(iter, curr_corr))        

        iter += 1

    return u, ui, v, vi
