import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from tqdm import tqdm
from posteriors import*
from Common_Tools import*
from Levy_Generators import*
from Levy_State_Space import*
from mcmc_sampler import*
from scipy.integrate import quad






# Define the integrand function
def nvm_integrand(t, alpha, beta, epsilon):
    """
    The integrand for the truncated rate calculation.
    
    Args:
        t: Integration variable.
        alpha: Stability parameter (alpha > 0).
        beta: Tempering parameter (beta > 0).
        epsilon: Truncation threshold (epsilon > 0).
        
    Returns:
        The value of the integrand at t.
    """
    return np.exp(-beta * epsilon * t) / t**(alpha + 1)


# Function to compute the integral numerically for the rate parameter
def compute_lambda(alpha, beta, C, epsilon):
    """
    Computes the integral for the truncated rate of the tempered stable subordinator.
    
    Args:
        alpha: Stability parameter (alpha > 0).
        beta: Tempering parameter (beta > 0).
        C: Scale factor (C > 0).
        epsilon: Truncation threshold (epsilon > 0).
        
    Returns:
        The value of lambda and the numerical error estimate.
    """
    # Perform numerical integration
    result, error = quad(nvm_integrand, 1, np.inf, args=(alpha, beta, epsilon))
    
    # Scale the result
    lambda_value = (C / epsilon**alpha) * result
    return lambda_value, error



def NVM_ts_gt(x_axis,jump_sizes,muw,sigmaw,alpha,beta,C,epsilon):
    #x_axis (T,)
    #jump_sizes (N,)
    #muw float
    #sigmaw float
    #alpha float
    #beta float
    #C float
    jump_sizes = jump_sizes[jump_sizes > epsilon]
    overall_rate, error = compute_lambda(alpha, beta, C, epsilon)
    rates = np.ones(len(jump_sizes))
    rates = rates /len(jump_sizes)*overall_rate

    ground_truth_measure = NVM_ground_truth_measure(rates, jump_sizes, x_axis, muw, sigmaw)

    return ground_truth_measure



#The integrand for the integral
def ts_integrand(x, alpha, beta,C):

    return C*np.exp(-beta * x) / x**(alpha + 1)




def upper_tail_ts_measure(alpha, beta, C,x_axis):

    """
    Incrementally computes the upper tail function for a range of epsilon values.
    
    Args:
        alpha: Stability parameter (alpha > 0).
        beta: Tempering parameter (beta > 0).
        C: Scale factor (C > 0).
        x_axis: Array of truncation thresholds for the upper tail integral to be evaluated (ascending order)
        
    Returns:
        A 1D array of upper tail function values corresponding to the x_axis.
    """
    upper_tail_values = np.zeros_like(x_axis)
    x_axis = x_axis[::-1] #Reverse the x_axis for more convenient iteration
    #Initialize the upper tail integral and the integration error
    total_integral, error = quad(ts_integrand, x_axis[0], np.inf, args=(alpha, beta, C))
    upper_tail_values[0] = total_integral
    # Loop through epsilon values
    for i in range(len(x_axis) - 1):
        x_current = x_axis[i]
        x_next = x_axis[i + 1]
        
        # Integrate over the new segment (epsilon_next, epsilon_current)
        segment_integral,segment_error = quad(ts_integrand, x_next, x_current, args=(alpha, beta,C))
        total_integral += segment_integral  # Update the accumulated integral
        error += segment_error
        upper_tail_values[i+1] = total_integral
    upper_tail_values = upper_tail_values[::-1]
    return upper_tail_values



#This function computes the tail functions for NVM ground truth density
#The cdf and sf in this project are always needed in pair
#A main benefit of this scheme is that the truncation parameter can be arbitrarily small, as long as the final shape is comparable to the true one it would be fine. 
#The rate neednot be exact due to the additional stability introduced by using the tail functions
def NVM_ground_truth_tail_functions(x_axis,jump_sizes,muw,sigmaw,alpha,beta,C,epsilon=0.001): 
    """
    Compute the CDF for a mixture of Gaussians based on the NVM ground truth measure using SciPy.

    Args:
    Returns:
        ndarray: CDF values evaluated at the points x_axis.
    """

    if 0 in x_axis:
        raise ValueError("The input x_axis contains 0, which is not allowed. Please modify your input.")
    #An additional explicit truncation for more regulated behaviour by the parameters. The controlled behaviour by the parameters is more understood with explicit truncation
    jump_sizes = jump_sizes[jump_sizes > epsilon]
    overall_rate, error = compute_lambda(alpha, beta, C, epsilon)
    rates = np.ones(len(jump_sizes))
    rates = rates /len(jump_sizes)*overall_rate #These would be the weigths assigned for the mixture model
    #Separate the axis into the positive and negative halves first
    negative_half = x_axis[x_axis < 0]  
    positive_half = x_axis[x_axis > 0]
    # Compute kernel parameters
    kernel_means = muw * jump_sizes[:, None]  # Shape: (N, 1)
    kernel_stds = sigmaw * np.sqrt(jump_sizes[:, None])  # Shape: (N, 1)

    # Compute Gaussian CDFs for all kernels at points in x_axis
    kernels_cdf = norm.cdf(negative_half[None, :], loc=kernel_means, scale=kernel_stds)  # Shape: (N/2, M) Compute the cdf for the negative half
    kernels_sf = norm.sf(positive_half[None, :], loc=kernel_means, scale=kernel_stds)
    
    # Weighted sum of kernels (mixture CDF)
    smoothed_cdf = np.sum(rates[:, None] * kernels_cdf, axis=0)
    smoothed_sf = np.sum(rates[:, None] * kernels_sf, axis=0)
    return smoothed_cdf,smoothed_sf

