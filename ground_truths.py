import numpy as np
from scipy.linalg import expm #This is the automatic matrix expnent solver
import math
import seaborn as sns
from scipy.special import logsumexp
from numba import jit
import matplotlib.pyplot as plt
from tqdm import tqdm
from posteriors import*
from Common_Tools import*
from Levy_Generators import*
from Levy_State_Space import*
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation, PillowWriter
from mcmc_sampler import*
from scipy.special import gammaincc
from scipy.special import gamma
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



#Sampling from a mixture of Gamma distribution
def _sample_gamma_mixture(N, shapes, scales, weights, rng):
    """
    Sample N positive jump sizes from a Gamma mixture:
      z ~ sum_k weights[k] * Gamma(shape=shapes[k], scale=scales[k])
    """
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()  # safety
    shapes  = np.asarray(shapes, dtype=float)
    scales  = np.asarray(scales, dtype=float)
    K = len(weights)
    if not (len(shapes) == len(scales) == K):
        raise ValueError("shapes, scales, and weights must have the same length.")

    # Choose component for each sample
    comp = rng.choice(K, size=N, p=weights)
    # Draw per component in batches (vectorized by component)
    z = np.empty(N, dtype=float)
    for k in range(K):
        idx = (comp == k)
        cnt = idx.sum()
        if cnt:
            # numpy gamma uses shape (k) and scale (theta)
            z[idx] = rng.gamma(shape=shapes[k], scale=scales[k], size=cnt)
    return z
#NVM Ground truth tails generator for mixture of Gamma distribution 
def nvm_tail_functions_mc(
    x_axis,
    muw,
    sigmaw,
    rate_lambda,
    shapes,
    scales,
    weights,
    N=100_000,
    epsilon=0.0,
    chunk_size=20_000,
    random_state=None,
):
    """
    Monte-Carlo NVM ground-truth tail functions for a finite- or infinite-activity subordinator
    with *known* total rate `rate_lambda` and jump-size density f given by a Gamma mixture.

    Returns two arrays:
      - left_tail for x_axis[x<0]:  ν_X((−∞, x])  (a CDF-like left tail on the negative half)
      - right_tail for x_axis[x>0]: ν_X([x, ∞))   (a survival function on the positive half)

    Parameters
    ----------
    x_axis : (M,) array-like
        Grid of x values. Must not contain 0 (the function splits by sign).
    muw : float
        Drift parameter in the Gaussian kernel: mean = muw * z.
    sigmaw : float
        Scale parameter in the Gaussian kernel: std = sigmaw * sqrt(z).
    rate_lambda : float
        Overall jump rate λ of the NVM Lévy measure (must be known).
    shapes, scales, weights : sequences of floats
        Parameters of the bimodal Gamma mixture for f(z).
        Gamma is parameterized by shape k and scale θ.
    N : int
        Number of Monte-Carlo draws z ~ f to use.
    epsilon : float
        Optional truncation: discard z <= epsilon (useful for stability if desired).
        The effective sample size after truncation is used in the 1/N factor.
    chunk_size : int
        Process samples in chunks of this size to limit memory.
    random_state : int or np.random.Generator or None
        RNG seed or Generator.

    Returns
    -------
    left_tail, right_tail : np.ndarray
        Arrays corresponding to x_axis[x<0] and x_axis[x>0], respectively.
    """
    x_axis = np.asarray(x_axis, dtype=float)

    if np.any(x_axis == 0):
        raise ValueError("x_axis contains 0; please exclude it (the function splits by sign).")

    # Split x-axis
    neg_mask = x_axis < 0
    pos_mask = x_axis > 0
    x_neg = x_axis[neg_mask]
    x_pos = x_axis[pos_mask]

    # Prepare RNG
    rng = np.random.default_rng(random_state) if not isinstance(random_state, np.random.Generator) else random_state

    # Draw jump sizes from the Gamma mixture
    z = _sample_gamma_mixture(N, shapes, scales, weights, rng)
    # Optional truncation (to mirror your previous function's explicit control)
    if epsilon > 0.0:
        z = z[z > epsilon]

    eff_N = z.size
    if eff_N == 0:
        raise ValueError("No jump sizes left after truncation; decrease epsilon or increase N.")

    # Accumulate MC sums in chunks to avoid creating huge (N x |x|) matrices
    left_sum  = np.zeros_like(x_neg, dtype=float) if x_neg.size else None
    right_sum = np.zeros_like(x_pos, dtype=float) if x_pos.size else None

    start = 0
    while start < eff_N:
        end = min(start + chunk_size, eff_N)
        z_chunk = z[start:end]                                # (B,)
        means = muw * z_chunk[:, None]                        # (B,1)
        stds  = sigmaw * np.sqrt(z_chunk)[:, None]            # (B,1)

        if x_neg.size:
            # Φ(x_neg; mean, std)
            left_sum += norm.cdf(x_neg[None, :], loc=means, scale=stds).sum(axis=0)
        if x_pos.size:
            # SF(x_pos; mean, std) = 1 - Φ(x_pos; mean, std)
            right_sum += norm.sf(x_pos[None, :], loc=means, scale=stds).sum(axis=0)

        start = end

    # Monte Carlo average, then scale by λ
    scale = rate_lambda / eff_N
    left_tail  = scale * left_sum  if x_neg.size else np.array([], dtype=float)
    right_tail = scale * right_sum if x_pos.size else np.array([], dtype=float)

    return left_tail, right_tail



#Checking or Experiment Functions
def _nvm_ts_gt_experiment():
    #Define the truncation threshold parameter
    epsilon = 0.05
    #Define the nvm and tempered stable parameters first
    muw = 2.0
    sigmaw = 1.0
    beta=0.2
    alpha = 0.5
    C=0.2
    sim_rate = 50 #The parameter c in the latex notes

    #Define the simulation time and run the simulation
    T = 10000.0
    sub_jump_sizes,jump_times = tempered_stable_process_jumps((beta,alpha,C),T,sim_rate)

    nvm_axis = np.linspace(-10,10,1000)

    ground_truth_measure = NVM_ts_gt(nvm_axis,sub_jump_sizes[0,:],muw,sigmaw,alpha,beta,C,epsilon)
    plt.figure()
    plt.plot(nvm_axis,ground_truth_measure)
    plt.show()

    return



def _ts_gt_experiment():
    x_axis = np.linspace(0.001,5,1000)
    beta=0.2
    alpha = 0.5
    C=0.2

    upper_tail_values = upper_tail_ts_measure(alpha, beta, C,x_axis)
    plt.figure()
    plt.plot(x_axis,upper_tail_values)
    #plt.xscale("log")
    #plt.yscale("log")
    plt.show()
    return

#The following is the actions to be run when the file is called directly
def main():
    #_nvm_ts_gt_experiment()
    _ts_gt_experiment()
    return

if __name__ == "__main__":
    main()



