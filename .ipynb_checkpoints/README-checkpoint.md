# ASTR 4470 Final Project - MCMC Parameter Estimation

This is a code for parameter estimation using Monte Carlo methods, with the primary application of fitting a 2D gaussian. For astronomy, this can be used to fit the profiles of star images (which makes tasks such as calculating the FWHM (full width at half-maximum) easy. The module has the capacity to fit other multidimensional functions written by the user, however, all necessary functions for a 2D gaussian are already provided within the module.

### Getting started
To use the code, simply download the module files **my_mcmc.py** and **make_plots.py**. You will also need to install/import standard packages such as numpy and matplotlib (for more detail see the examples below). The **make_plots** module contains functions to produce histograms of the parameters, trace plots to determine convergence of the chain, scatter plots to look at the autocorrelation, and (for the case of fitting a star profile) images of the model side-by-side with the data. The **my_mcmc** module contains functions to generate synthetic data, create a Markov chain, as well as the specific functions needed for a 2D gaussian.

### Examples and Usage
As mentioned above, the code is well-suited for fitting a 2D gaussian which is useful for analyzing images of stars. In this repository there are two example images as well as the code to process and fit their profiles in the **test.ipynb** notebook. 
In the same notebook, there are some toy examples for generating and fitting simulated data that can be used to ensure the code is running correctly.
