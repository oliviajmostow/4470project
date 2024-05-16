import numpy as np
from numpy import random
import pandas as pd

#going to try and do this specifically for a 2d gaussian, but this fits more into testing than the general code
def log_likelihood(y_dist='gaussian', params, x, y, model):
    """
    Computes the log likelihood
    """      
    log_l = -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))
    return 


def log_probability(log_likelihood, log_prior):
    """
    Function to compute the log probability which will go into the sampler.
    Inputs: log of likelihood function and prior probability distribution
    """
    pass


#function to sample the probability distribution
#start by imagining that there's only one parameter to estimate
#at some point im going to want to store the values for every step but start without doing that
def ens_sampler(probability, guess, burn_in=500, steps=10000, nwalkers=100, sigma):
    """
    Sample function. Required inputs are a log probability and (). 
    Optional inputs are the number of steps to throw out (burn in), the number of steps to take,
    and the number of walkers.
    Outputs ???
    """
    #set up array to store the positions of the walkers, should have dimension (steps, nwalkers)
    X = np.zeros(steps, nwalkers)
    #initialize all walkers in a region around the mean value of the prior
    X0 = guess + np.random.normal(0, np.std(prior)) #does this make sense?
    #take the specified number of steps, following the metropolis algorithm
    v = np.copy(v_0)
    X = np.zeros(steps)
    step_num = 0
    accepts = 0
    for i in range(nwalkers):
        while step_num < burn_in:
            X[step_num+1,i] = metropolis_hastings(probability, X[step_num,i], sigma)[0]
        step_num = 0
        while step_num < steps:
            X[step_num+1,i] = metropolis_hastings(probability, X[step_num,i], sigma)[0]
            accepts += metropolis_hastings(probability, X[step_num,i], sigma)[1]
    #compute acceptance fraction
    accept_frac = accepts/steps
    return X, accept_frac


def metropolis_hastings(probability, x0, sigma):
    """
    Metropolis-Hastings algorithm to determine how to move. Takes in a probability distribution function
    that is used to compare the relative probability of the proposed position with the old position. 
    Other required inputs: initial position, sigma (scale for step).
    Outputs: position after the step and whether or not the step was accepted
    """
    x_prop = np.random.normal(loc=v0, scale=sigma)
    acceptance_prob = probability(x_prop)/probability(x0)
    u = np.random.rand()
    if u <= acceptance_prob:
        x1 = proposed_x
        accept = 1
    else:
        x1 = x0
        accept = 0
    return np.array(x1, a)



