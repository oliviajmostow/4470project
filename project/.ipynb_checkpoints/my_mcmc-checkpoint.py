#imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import my_mcmc as mcmc
from astropy.io import fits
from pathlib import Path
import make_plots as mkp


def generate_data(func, npts, par, sigma=0, upper_bound=10, lower_bound=0):
    """
    Function to generate simulated data from a model. Required inputs:
    func: model that takes in independent variable(s) and returns scalars
    npts: number of points to generate
    par: array of parameters for the model
    sigma: standard deviation for gaussian noise (defaults to zero)
    upper_bound: maximum value of independent variable
    lower_bound: minimum value of independent variable
    Returns:
    t_vals: array of 2-D coordinates with dimensions (npts, npts, 2)
    values: evaluation of the model at each coord. pair; array w/ dimensions of (npts, npts)
    
    *Note: The upper/lower bounds and number of points apply to both Y and X
    """
    points = np.arange(lower_bound, upper_bound, (upper_bound-lower_bound)/npts)
    X,Y = np.meshgrid(points, points)
    values = np.zeros(npts*npts).reshape(npts, npts)
    t_vals = np.zeros(2*npts*npts).reshape(npts,npts,2)
    for i in range(npts):
        for j in range(npts):
            t = np.array([X[i,j], Y[i,j]])
            t_vals[i,j] = t
            values[i,j] = func(t, par) + np.random.normal(0, sigma)
    return t_vals, values, np.array(par)


def loglikelihood(func, t, y, x, sigma): 
    """
    Function to compute the likelihood (probabiltiy of the data for a given
    set of parameters). Returns a scalar value given a function, data points,
    parameters, and estimated error.
    """
    return -(((y-func(t,x))/sigma)**2).sum()


def move(func, t, y, x, sigma, mins, maxes, h = 0.1, TINY=1e-30):
    """
    Step algorithm. Takes a random step in all dimensions of the parameter space,
    computes the likelihood and compares it to the current location. The step is accepted
    always if the (log) likelihood is greater for the new location, and with probability that
    depends on the relative likelihoods if the likelihood is lesser for the new location. 
    Inputs:
    func: model to fit
    t: independent variables (X and Y data points)
    y: values at points given by t
    x: parameters
    sigma: 
    h: max step size
    """
    
    dx = (2*np.random.rand(x.size)-1.)*h
    xnew = x + dx
    
    # compute the priors
    p1 = priors(mins, maxes, x) 
    p2 = priors(mins, maxes, xnew)

    if p2 <= 0: 
        return x, False
  
    p = p2/(p1+TINY)

    # compute the likelihoods
    r1 = loglikelihood(func, t, y, x, sigma)
    r2 = loglikelihood(func, t, y, xnew, sigma)
    

    r = min(np.exp(r2-r1)*p, 1.)
    if(np.random.rand() < r):
        return xnew, True

    return x, False 

def run_mcmc(func, t, y, x, sigma, mins, maxes, h=0.1, burn_in=1000, MAX_CHAIN=500000):
    """
    Uses the step function above to create a chain of samples, storing the values
    of all parameters at each step in an array.
    """
    xchain = [] 
    i = 0

    for _ in range(MAX_CHAIN): 
        x, accepted_move = move(func, t, y, x, sigma, mins, maxes, h)
        if accepted_move:
            i += 1 
            if i > burn_in: 
                xchain.append(x)
                
    return np.array(xchain)


def gaussian_2d_ind_gen(t, x):
    """
    Given a data point t and parameters x, outputs result of corresponding 2D gaussian.
    Used to generate data in test cases. Function is only applicable for X and Y independent.
    """
    sig_x = x[2]
    sig_y = x[3]
    A = x[4]
    mu_x = x[0]
    mu_y = x[1]
    y = A*(np.exp(-.5*(t[0]-mu_x)**2/sig_x))*(np.exp(-.5*(t[1]-mu_y)**2/sig_y))
    return y


def gaussian_2d_ind(t, x):
    """
    Similar to the above function for a single point, but to be used when t is an
    array of coordinate pairs rather than a single point.
    """
    sig_x = x[2]
    sig_y = x[3]
    A = x[4]
    mu_x = x[0]
    mu_y = x[1]
    mu = np.array([mu_x, mu_y])
    g = A*(np.exp(-.5*((t[:,:,0]-mu_x)/sig_x)**2))*(np.exp(-.5*((t[:,:,1]-mu_y)/sig_y)**2))
    return g


def priors(mins, maxes, params):  
    """
    Computes the (flat) prior probability given user-inputted bounds on each parameter. 
    Inputs:
    mins: a list of minimum values for each parameter, in the order
    they will be passed in to the MCMC function as an array
    maxes: list of maximum values, in order
    params: the list of parameter values to check
    Output:
    0 if any of the parameters are out of the specified range
    1 if not
    """
    for i in range(len(mins)):
        if (params[i] < mins[i]) | (params[i] > maxes[i]):
            return 0  
    return 1 

def autocorr(xchain, lag, nparams):
    """
    Computes the correlation between the series and the lagged version of it
    as a means of determining what value to use for the number of burn-in steps.
    """
    correlations = []
    for i in range(nparams):
        correlations.append(np.corrcoef(xchain[lag:,i], xchain[:-lag,i])[0,1])
    return correlations


def process_img(img_star):
    """
    Scales the values in the image to make setting priors simpler.
    """
    img2 = img_star/img_star.max()
    img3 = img2/img2.sum(axis=1).sum()
    return img3

def get_coords(window_size):
    """
    Returns coordinate pairs for a given window size.
    """
    y_vals, x_vals = np.mgrid[:2*window_size, :2*window_size]
    npts = len(x_vals)
    t_vals = np.zeros(2*npts*npts).reshape(npts,npts,2)
    for i in range(npts):
        for j in range(npts):
            t = np.array([x_vals[i,j], y_vals[i,j]])
            t_vals[i,j] = t
    return t_vals, y_vals, x_vals


def trigproduct(t, x):
    """
    Example function to demonstrate use on another 2D function.
    """
    a,b = t
    phi_1, phi_2 = x
    return np.sin(a + phi_1)*np.cos(b + phi_2)

def trigproduct_v2(t, x):
    """
    Same as above, except t has dimension (npts, npts, 2) instead of (2)
    """
    a = t[:,:,0]
    b = t[:,:,1]
    phi_1 = x[0]
    phi_2 = x[1]
    return np.sin(a + phi_1)*np.cos(b + phi_2)


def gaussian_2d_general(t, x):
    """
    More general form of the 2d gaussian where X and Y are not independent, 
    with an additional parameter theta to describe the angle of the elliptical
    shape.
    """
    #define the 6 parameters
    sig_x = x[2]
    sig_y = x[3]
    A = x[4]
    mu_x = x[0]
    mu_y = x[1]
    theta = x[5]
    a = (np.cos(theta)**2)/(2*(sig_x**2)) + (np.sin(theta)**2)/(2*(sig_y**2))
    b = (np.sin(2*theta))/(2*(sig_x**2)) - (np.sin(2*theta))/(2*(sig_y**2))
    c = (np.sin(theta)**2)/(2*(sig_x**2)) + (np.cos(theta)**2)/(2*(sig_y**2))
    x_min_x0 = t[:,:,0]-mu_x 
    y_min_y0 = t[:,:,1]-mu_y
    g = A*(np.exp(-a*(x_min_x0**2) -b*(x_min_x0*y_min_y0) -c*(y_min_y0**2)))
    return g

def gaussian_2d(t, x):
    """
    Function same as above but for generating data (t is 2D).
    """
    #define the 6 parameters
    sig_x = x[2]
    sig_y = x[3]
    A = x[4]
    mu_x = x[0]
    mu_y = x[1]
    theta = x[5]
    a = (np.cos(theta)**2)/(2*(sig_x**2)) + (np.sin(theta)**2)/(2*(sig_y**2))
    b = (np.sin(2*theta))/(2*(sig_x**2)) - (np.sin(2*theta))/(2*(sig_y**2))
    c = (np.sin(theta)**2)/(2*(sig_x**2)) + (np.cos(theta)**2)/(2*(sig_y**2))
    x_min_x0 = t[0]-mu_x 
    y_min_y0 = t[1]-mu_y
    g = A*(np.exp(-a*(x_min_x0**2) -b*(x_min_x0*y_min_y0) -c*(y_min_y0**2)))
    return g

