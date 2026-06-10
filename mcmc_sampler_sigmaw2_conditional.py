"""MCMC samplers conditioning on the NVM variance parameter sigma_w^2.

Variants of the Langevin / NVM inference routines that treat the NVM variance
parameter sigma_w^2 as a conditioning variable, including stationary-domain and
alternative-initialisation versions used to produce the paper's experiments.

Part of the code accompanying:
    Lin, B. Z. & Godsill, S. (2025). Bayesian Non-Parametric Inference for
    Lévy Measures in State-Space Models. arXiv:2505.22587.
"""
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


#No DP Alpha Sampling Here
def langevin_lm_theta_inference_all_fixed_DP_alpha_conditional_sigmaw2(num_iter,block_width,overlapping_width,observations,time_axis,initial_jump_sizes,initial_jump_times,sigmav2,rate_alpha_prior,rate_beta_prior,dir_K,base_gamma_a_mean,base_gamma_a_std,base_gamma_b_mean,base_gamma_b_std,sigmaw2,theta_prior,theta_step_size,base_gamma_a_step_size,base_gamma_b_step_size,return_states=False,base_gamma_a=False,base_gamma_b=False,dir_alpha=False,muw_prior_mean = 0.0,muw_prior_variance=1.0):
    """
    Additional Inputs:
        sigmav2 instead of kv for the conditional sampler. Note that sigmav2 is the variance but not the standard deviation
        sigmaw2 instead of alphaw and betaw for direct conditioning.


         Note that the Gamma densities here all follow the shape-scale convention as in np.random.gamma
        base_gamma_a_mean : Gaussian prior N(base_a_mean,base_a_std^2)for the first a parameter in the Gamma(a',b') base density of the DP
        base_gamma_a_std  : same as above. std is used insteade of var to cater for the numpy convention
        base_gamma_b_mean : Gaussian prior N(base_b_mean,base_b_std^2)for the second b parameter in the Gamma(a',b') base density of the DP
        base_gamma_b_std
        base_gamma_a_step_size,base_gamma_b_step_size: GRW proposal std sizes
    """
    #Initialise the prior system structure
    #alpha_prior and beta_prior are prior on the exponential rate parameter
    A = np.zeros((2, 2))
    A[0, 1] = 1.0
    A[1, 1] = theta_prior
    h = np.zeros((2,1))
    h[1,0] = 1.0
    g = np.zeros((1,3))
    g[0,0] = 1 #Only the integral term x is being observed
    R = np.array([sigmav2])


    #Initial subordiantor jump sizes and times
    jump_times = initial_jump_times
    jump_sizes = initial_jump_sizes


    #Initialise Kalman filter
    mean_prior = np.zeros((3,1))
    mean_prior[2,0] = muw_prior_mean
    covariance_prior = np.zeros((3,3))
    covariance_prior[2,2] = muw_prior_variance
    grouped_jump_sizes,grouped_jump_times = group_jumps(jump_sizes,jump_times,time_axis)
    
    #Create the initial Kalman reference path for the MH step within Gibbs.
    #The previous_log_likelihood is stored for the latter MH step for updating the system structure
    previous_log_likelihood, individual_log_likelihoods,x_means,x_covariances = compute_log_likelihood_langevin_conditional_on_sigmaw2(mean_prior,covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,sigmaw2)
    prior_path = integrate_to_path(jump_sizes,jump_times,time_axis)

    #Initialize the samples
    """
    Initial jump rate sampling
    """
    sample_rate = exp_rate_post(jump_times, rate_alpha_prior, rate_beta_prior, num_samp=1)
    """
    Initial jump density sampling
    dir_alpha is now obtained from the initialized posterior distirbution for it
    """
    #Initially, either sample the hyper-parameters from the priors for initialization or read from the initialized values

    if not base_gamma_a:
        base_gamma_a = np.random.normal(base_gamma_a_mean,base_gamma_a_std) #Note that the numpy normal uses std but not var
    else:
        base_gamma_a = base_gamma_a
    if not base_gamma_b:
        base_gamma_b = np.random.normal(base_gamma_b_mean,base_gamma_b_std)
    else:
        base_gamma_b = base_gamma_b
    sample_measure = js_dist_post(jump_sizes,dir_alpha,dir_K,base_gamma_a,base_gamma_b)

    #Containers of the samples
    dir_alphas = [dir_alpha]
    base_gamma_as = [base_gamma_a]
    base_gamma_bs = [base_gamma_b]
    sample_paths = [prior_path]
    sample_measures = [sample_measure]
    sample_rates = [sample_rate]
    sample_thetas = [theta_prior]
    sample_x_means=[x_means]
    sample_x_covariances = [x_covariances]
    overall_acceptance_probabilities = []
    theta_acceptance_probabilities = np.zeros(num_iter)
    theta_accept_count = 0
    for iter in tqdm(range(num_iter),desc="Langevin System Inference Progress:"):
        """
        Step 1
        theta parameter sampling
        """
        proposed_theta = sample_thetas[-1] + np.random.randn()*theta_step_size
            #Update the Langevin matrix
        A[1, 1] = proposed_theta
        #No need to update alphaw_post and betaw_post here, since we only want the likelihood
        new_log_likelihood, _,_,_ = compute_log_likelihood_langevin_conditional_on_sigmaw2(mean_prior,covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,sigmaw2)
        theta_acceptance_probability = new_log_likelihood-previous_log_likelihood
        theta_acceptance_probabilities[iter] = theta_acceptance_probability
        if np.log(np.random.rand())<= theta_acceptance_probability:# Acceptance case
            sample_thetas.append(proposed_theta)
            #previous_log_likelihood = new_log_likelihood
            theta_accept_count += 1
        else: #Rejection case
            sample_thetas.append(sample_thetas[-1])

        """
        Step 2
        jump sizes sampling
        """
        #Sample jump sizes and times and update the Kalman reference path
        #Conditioning on the previous samples update the hidden states
        grouped_jump_sizes,grouped_jump_times,new_individual_log_likelihoods,x_means,x_covariances,overall_acceptance_probability = overlapping_block_updates_conditional_on_sigmaw2(block_width,overlapping_width,grouped_jump_sizes,grouped_jump_times,time_axis,observations,sample_thetas[-1],sigmav2,sigmaw2,sample_rate,sample_measure,individual_log_likelihoods,x_means,x_covariances)
        #Compute the log likelihood for the new parameters
        previous_log_likelihood = np.sum(new_individual_log_likelihoods)

        jump_sizes_array = [arr for arr in grouped_jump_sizes.copy() if arr.size > 0]
        jump_times_array = [arr for arr in grouped_jump_times.copy() if arr.size > 0]
        jump_sizes_array = np.concatenate(jump_sizes_array)
        jump_times_array = np.concatenate(jump_times_array)
        jump_sizes = np.zeros((1,len(jump_sizes_array)))
        jump_times = np.zeros((1,len(jump_times_array)))
        jump_sizes[0,:] = jump_sizes_array
        jump_times[0,:] = jump_times_array

        sample_x_means.append(x_means)
        sample_x_covariances.append(x_covariances)

        sample_path = integrate_to_path(jump_sizes,jump_times,time_axis)
        sample_paths.append(sample_path)
        overall_acceptance_probabilities.append(overall_acceptance_probability)
        """
        Step 3
        jump rate sampling
        """
        sample_rate = exp_rate_post(jump_times, rate_alpha_prior, rate_beta_prior, num_samp=1)
        """
        Step 4
        jump density sampling
        """
        #Sample first the hyper-parameters
        base_gamma_a,base_gamma_b = dir_gamma_base_params_post(jump_sizes,base_gamma_a,base_gamma_b,base_gamma_a_step_size,base_gamma_b_step_size,base_gamma_a_mean,base_gamma_a_std,base_gamma_b_mean,base_gamma_b_std)
        #Then sample the measure conditional on the hyper-parameters sampled
        sample_measure = js_dist_post(jump_sizes,dir_alpha,dir_K,base_gamma_a,base_gamma_b)
        sample_measures.append(sample_measure)
        sample_rates.append(sample_rate)
        dir_alphas.append(dir_alpha)
        base_gamma_as.append(base_gamma_a)
        base_gamma_bs.append(base_gamma_b)


        

    
    theta_acceptance_probabilities = np.exp(theta_acceptance_probabilities)
    overall_theta_acceptance_probability = theta_accept_count/num_iter
    #Return lists of samples, each of which is a list of num_iter length 
    if return_states:
        return sample_measures,sample_rates,sample_paths,sample_thetas,overall_theta_acceptance_probability,sample_x_means,sample_x_covariances,jump_sizes,jump_times,dir_alphas,base_gamma_as,base_gamma_bs

    else:   
        return sample_measures,sample_rates,sample_paths,sample_thetas,overall_theta_acceptance_probability,sample_x_means,sample_x_covariances,dir_alphas,base_gamma_as,base_gamma_bs


#Shape-scale convention for Gamma here, opposite to the other function.
def langevin_lm_theta_inference_all_DP_alpha_inferred_Gamma_base_Gaussian_update_conditional_sigmaw2(num_iter,block_width,overlapping_width,observations,time_axis,initial_jump_sizes,initial_jump_times,sigmav2,rate_alpha_prior,rate_beta_prior,dir_K,dir_alpha_a,dir_alpha_b,base_gamma_a_mean,base_gamma_a_std,base_gamma_b_mean,base_gamma_b_std,sigmaw2,theta_prior,theta_step_size,base_gamma_a_step_size,base_gamma_b_step_size,return_states=False,base_gamma_a=False,base_gamma_b=False,dir_alpha=False,muw_prior_mean = 0.0,muw_prior_variance=1.0,compute_smoothing = False):
    """
    Additional Inputs:
        sigmav2 instead of kv for the conditional sampler. Note that sigmav2 is the variance but not the standard deviation
        sigmaw2 instead of alphaw and betaw for direct conditioning.


         Note that the Gamma densities here all follow the shape-scale convention as in np.random.gamma
        dir_alpha_a : a parameter in the Gamma(a,b) prior for the Dirichlet process alpha parameter
        dir_alpha_b : b parameter same as above
        base_gamma_a_mean : Gaussian prior N(base_a_mean,base_a_std^2)for the first a parameter in the Gamma(a',b') base density of the DP
        base_gamma_a_std  : same as above. std is used insteade of var to cater for the numpy convention
        base_gamma_b_mean : Gaussian prior N(base_b_mean,base_b_std^2)for the second b parameter in the Gamma(a',b') base density of the DP
        base_gamma_b_std
        base_gamma_a_step_size,base_gamma_b_step_size: GRW proposal std sizes
    """
    #Initialise the prior system structure
    #alpha_prior and beta_prior are prior on the exponential rate parameter
    A = np.zeros((2, 2))
    A[0, 1] = 1.0
    A[1, 1] = theta_prior
    h = np.zeros((2,1))
    h[1,0] = 1.0
    g = np.zeros((1,3))
    g[0,0] = 1 #Only the integral term x is being observed
    R = np.array([sigmav2])


    #Initial subordiantor jump sizes and times
    jump_times = initial_jump_times
    jump_sizes = initial_jump_sizes


    #Initialise Kalman filter
    mean_prior = np.zeros((3,1))
    mean_prior[2,0] = muw_prior_mean
    covariance_prior = np.zeros((3,3))
    covariance_prior[2,2] = muw_prior_variance
    grouped_jump_sizes,grouped_jump_times = group_jumps(jump_sizes,jump_times,time_axis)
    
    #Create the initial Kalman reference path for the MH step within Gibbs.
    #The previous_log_likelihood is stored for the latter MH step for updating the system structure
    if compute_smoothing:
        previous_log_likelihood, individual_log_likelihoods,x_means,x_covariances,smoothed_means,smoothed_covariances = compute_log_likelihood_langevin_conditional_on_sigmaw2(mean_prior,covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,sigmaw2,compute_smoothing=compute_smoothing)
    else:
        previous_log_likelihood, individual_log_likelihoods,x_means,x_covariances = compute_log_likelihood_langevin_conditional_on_sigmaw2(mean_prior,covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,sigmaw2,compute_smoothing=compute_smoothing)
    prior_path = integrate_to_path(jump_sizes,jump_times,time_axis)

    #Initialize the samples
    """
    Initial jump rate sampling
    """
    sample_rate = exp_rate_post(jump_times, rate_alpha_prior, rate_beta_prior, num_samp=1)
    """
    Initial jump density sampling
    dir_alpha is now obtained from the initialized posterior distirbution for it
    """
    #Initially, either sample the hyper-parameters from the priors for initialization or read from the initialized values
    if not dir_alpha:
        dir_alpha = np.random.gamma(dir_alpha_a,dir_alpha_b)#Shape-scale convention
    else:
        dir_alpha = dir_alpha
    if not base_gamma_a:
        base_gamma_a = np.random.normal(base_gamma_a_mean,base_gamma_a_std) #Note that the numpy normal uses std but not var
    else:
        base_gamma_a = base_gamma_a
    if not base_gamma_b:
        base_gamma_b = np.random.normal(base_gamma_b_mean,base_gamma_b_std)
    else:
        base_gamma_b = base_gamma_b
    sample_measure = js_dist_post(jump_sizes,dir_alpha,dir_K,base_gamma_a,base_gamma_b)

    #Containers of the samples
    dir_alphas = [dir_alpha]
    base_gamma_as = [base_gamma_a]
    base_gamma_bs = [base_gamma_b]
    sample_paths = [prior_path]
    sample_measures = [sample_measure]
    sample_rates = [sample_rate]
    sample_thetas = [theta_prior]
    #The sample filtering distributions
    sample_x_means=[x_means]
    sample_x_covariances = [x_covariances]
    #The sample smoothing distributions if computed
    if compute_smoothing:
        sample_smoothed_means = [smoothed_means]
        sample_smoothed_covariances = [smoothed_covariances]
    overall_acceptance_probabilities = []
    theta_acceptance_probabilities = np.zeros(num_iter)
    theta_accept_count = 0
    for iter in tqdm(range(num_iter),desc="Langevin System Inference Progress:"):
        """
        Step 1
        theta parameter sampling
        """
        proposed_theta = sample_thetas[-1] + np.random.randn()*theta_step_size
            #Update the Langevin matrix
        A[1, 1] = proposed_theta
        #No need to update alphaw_post and betaw_post here, since we only want the likelihood
        new_log_likelihood, _,_,_ = compute_log_likelihood_langevin_conditional_on_sigmaw2(mean_prior,covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,sigmaw2)
        theta_acceptance_probability = new_log_likelihood-previous_log_likelihood
        theta_acceptance_probabilities[iter] = theta_acceptance_probability
        if np.log(np.random.rand())<= theta_acceptance_probability:# Acceptance case
            sample_thetas.append(proposed_theta)
            #previous_log_likelihood = new_log_likelihood
            theta_accept_count += 1
        else: #Rejection case
            sample_thetas.append(sample_thetas[-1])
        #print(new_log_likelihood)
        """
        Step 2
        jump sizes sampling
        """
        #Sample jump sizes and times and update the Kalman reference path
        #Conditioning on the previous samples update the hidden states
        if compute_smoothing:
            grouped_jump_sizes,grouped_jump_times,new_individual_log_likelihoods,x_means,x_covariances,overall_acceptance_probability,smoothed_means, smoothed_covariances = overlapping_block_updates_conditional_on_sigmaw2(block_width,overlapping_width,grouped_jump_sizes,grouped_jump_times,time_axis,observations,sample_thetas[-1],sigmav2,sigmaw2,sample_rate,sample_measure,individual_log_likelihoods,x_means,x_covariances,compute_smoothing=compute_smoothing)
        else:
            grouped_jump_sizes,grouped_jump_times,new_individual_log_likelihoods,x_means,x_covariances,overall_acceptance_probability = overlapping_block_updates_conditional_on_sigmaw2(block_width,overlapping_width,grouped_jump_sizes,grouped_jump_times,time_axis,observations,sample_thetas[-1],sigmav2,sigmaw2,sample_rate,sample_measure,individual_log_likelihoods,x_means,x_covariances,compute_smoothing=compute_smoothing)
        #Compute the log likelihood for the new parameters
        previous_log_likelihood = np.sum(new_individual_log_likelihoods)
        #print(previous_log_likelihood)
        jump_sizes_array = [arr for arr in grouped_jump_sizes.copy() if arr.size > 0]
        jump_times_array = [arr for arr in grouped_jump_times.copy() if arr.size > 0]
        jump_sizes_array = np.concatenate(jump_sizes_array)
        jump_times_array = np.concatenate(jump_times_array)
        jump_sizes = np.zeros((1,len(jump_sizes_array)))
        jump_times = np.zeros((1,len(jump_times_array)))
        jump_sizes[0,:] = jump_sizes_array
        jump_times[0,:] = jump_times_array

        sample_x_means.append(x_means)
        sample_x_covariances.append(x_covariances)
        if compute_smoothing:
            sample_smoothed_means.append(smoothed_means)
            sample_smoothed_covariances.append(smoothed_covariances)

        sample_path = integrate_to_path(jump_sizes,jump_times,time_axis)
        sample_paths.append(sample_path)
        overall_acceptance_probabilities.append(overall_acceptance_probability)
        """
        Step 3
        jump rate sampling
        """
        sample_rate = exp_rate_post(jump_times, rate_alpha_prior, rate_beta_prior, num_samp=1)
        """
        Step 4
        jump density sampling
        """
        #Sample first the hyper-parameters
        dir_alpha = dir_alpha_post(jump_sizes,dir_alpha,dir_alpha_a,dir_alpha_b)#shape-scale convention for this function
        base_gamma_a,base_gamma_b = dir_gamma_base_params_post(jump_sizes,base_gamma_a,base_gamma_b,base_gamma_a_step_size,base_gamma_b_step_size,base_gamma_a_mean,base_gamma_a_std,base_gamma_b_mean,base_gamma_b_std)
        #Then sample the measure conditional on the hyper-parameters sampled
        sample_measure = js_dist_post(jump_sizes,dir_alpha,dir_K,base_gamma_a,base_gamma_b)
        sample_measures.append(sample_measure)
        sample_rates.append(sample_rate)
        dir_alphas.append(dir_alpha)
        base_gamma_as.append(base_gamma_a)
        base_gamma_bs.append(base_gamma_b)


        

    
    theta_acceptance_probabilities = np.exp(theta_acceptance_probabilities)
    overall_theta_acceptance_probability = theta_accept_count/num_iter
    #Return lists of samples, each of which is a list of num_iter length 
    if return_states:
        if compute_smoothing:
            return sample_measures,sample_rates,sample_paths,sample_thetas,overall_theta_acceptance_probability,sample_x_means,sample_x_covariances,jump_sizes,jump_times,dir_alphas,base_gamma_as,base_gamma_bs, sample_smoothed_means, sample_smoothed_covariances
        else:
            return sample_measures,sample_rates,sample_paths,sample_thetas,overall_theta_acceptance_probability,sample_x_means,sample_x_covariances,jump_sizes,jump_times,dir_alphas,base_gamma_as,base_gamma_bs

    else:   
        if compute_smoothing:
             return sample_measures,sample_rates,sample_paths,sample_thetas,overall_theta_acceptance_probability,sample_x_means,sample_x_covariances,dir_alphas,base_gamma_as,base_gamma_bs,sample_smoothed_means, sample_smoothed_covariances
        else:
            return sample_measures,sample_rates,sample_paths,sample_thetas,overall_theta_acceptance_probability,sample_x_means,sample_x_covariances,dir_alphas,base_gamma_as,base_gamma_bs



#Alternative Initialization Algorithm
#Shape-scale convention for Gamma here, opposite to the other function.
def alternative_initialization_langevin_lm_theta_inference_all_DP_alpha_inferred_Gamma_base_Gaussian_update_conditional_sigmaw2(num_iter,block_width,overlapping_width,observations,time_axis,sigmav2,rate_alpha_prior,rate_beta_prior,dir_K,dir_alpha_a,dir_alpha_b,base_gamma_a_mean,base_gamma_a_std,base_gamma_b_mean,base_gamma_b_std,sigmaw2,theta_prior,theta_step_size,base_gamma_a_step_size,base_gamma_b_step_size,return_states=False,base_gamma_a=False,base_gamma_b=False,dir_alpha=False,muw_prior_mean = 0.0,muw_prior_variance=1.0):
    """
    Additional Inputs:
        sigmav2 instead of kv for the conditional sampler. Note that sigmav2 is the variance but not the standard deviation
        sigmaw2 instead of alphaw and betaw for direct conditioning.


         Note that the Gamma densities here all follow the shape-scale convention as in np.random.gamma
        dir_alpha_a : a parameter in the Gamma(a,b) prior for the Dirichlet process alpha parameter
        dir_alpha_b : b parameter same as above
        base_gamma_a_mean : Gaussian prior N(base_a_mean,base_a_std^2)for the first a parameter in the Gamma(a',b') base density of the DP
        base_gamma_a_std  : same as above. std is used insteade of var to cater for the numpy convention
        base_gamma_b_mean : Gaussian prior N(base_b_mean,base_b_std^2)for the second b parameter in the Gamma(a',b') base density of the DP
        base_gamma_b_std
        base_gamma_a_step_size,base_gamma_b_step_size: GRW proposal std sizes
    """
    #Initialise the prior system structure
    #alpha_prior and beta_prior are prior on the exponential rate parameter
    A = np.zeros((2, 2))
    A[0, 1] = 1.0
    A[1, 1] = theta_prior
    h = np.zeros((2,1))
    h[1,0] = 1.0
    g = np.zeros((1,3))
    g[0,0] = 1 #Only the integral term x is being observed
    R = np.array([sigmav2])






    #Initialization

    initial_rate = np.random.gamma(rate_alpha_prior,1/rate_beta_prior,1) #We follow the shape-rate convnetion, which is the opposite to the shape-scale convention in numpy.
    
    #Initially, either sample the hyper-parameters from the priors for initialization or read from the initialized values
    if not dir_alpha:
        dir_alpha = np.random.gamma(dir_alpha_a,1/dir_alpha_b)#Shape-rate to shape-scale convention
    else:
        dir_alpha = dir_alpha
    if not base_gamma_a:
        base_gamma_a = np.random.normal(base_gamma_a_mean,base_gamma_a_std) #Note that the numpy normal uses std but not var
    else:
        base_gamma_a = base_gamma_a
    if not base_gamma_b:
        base_gamma_b = np.random.normal(base_gamma_b_mean,base_gamma_b_std)
    else:
        base_gamma_b = base_gamma_b

    sample_measure = js_dist_post(np.zeros((1, 0)),dir_alpha,dir_K,base_gamma_a,base_gamma_b) #Feed an empty jump sizes for prior sampling

    jump_sizes,jump_times = compound_poisson_DP(initial_rate,sample_measure,time_axis[-1])

    #Initialise Kalman filter
    mean_prior = np.zeros((3,1))
    mean_prior[2,0] = muw_prior_mean
    covariance_prior = np.zeros((3,3))
    covariance_prior[2,2] = muw_prior_variance
    grouped_jump_sizes,grouped_jump_times = group_jumps(jump_sizes,jump_times,time_axis)
    
    #Create the initial Kalman reference path for the MH step within Gibbs.
    #The previous_log_likelihood is stored for the latter MH step for updating the system structure
    previous_log_likelihood, individual_log_likelihoods,x_means,x_covariances = compute_log_likelihood_langevin_conditional_on_sigmaw2(mean_prior,covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,sigmaw2)
    prior_path = integrate_to_path(jump_sizes,jump_times,time_axis)



    #Containers of the samples
    dir_alphas = [dir_alpha]
    base_gamma_as = [base_gamma_a]
    base_gamma_bs = [base_gamma_b]
    sample_paths = [prior_path]
    sample_measures = [sample_measure]
    sample_rates = [initial_rate]
    sample_thetas = [theta_prior]
    sample_x_means=[x_means]
    sample_x_covariances = [x_covariances]
    overall_acceptance_probabilities = []
    theta_acceptance_probabilities = np.zeros(num_iter)
    theta_accept_count = 0
    for iter in tqdm(range(num_iter),desc="Langevin System Inference Progress:"):
        """
        Step 1
        theta parameter sampling
        """
        proposed_theta = sample_thetas[-1] + np.random.randn()*theta_step_size
            #Update the Langevin matrix
        A[1, 1] = proposed_theta
        #No need to update alphaw_post and betaw_post here, since we only want the likelihood
        new_log_likelihood, _,_,_ = compute_log_likelihood_langevin_conditional_on_sigmaw2(mean_prior,covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,sigmaw2)
        theta_acceptance_probability = new_log_likelihood-previous_log_likelihood
        theta_acceptance_probabilities[iter] = theta_acceptance_probability
        if np.log(np.random.rand())<= theta_acceptance_probability:# Acceptance case
            sample_thetas.append(proposed_theta)
            #previous_log_likelihood = new_log_likelihood
            theta_accept_count += 1
        else: #Rejection case
            sample_thetas.append(sample_thetas[-1])

        """
        Step 2
        jump sizes sampling
        """
        #Sample jump sizes and times and update the Kalman reference path
        #Conditioning on the previous samples update the hidden states
        grouped_jump_sizes,grouped_jump_times,new_individual_log_likelihoods,x_means,x_covariances,overall_acceptance_probability = overlapping_block_updates_conditional_on_sigmaw2(block_width,overlapping_width,grouped_jump_sizes,grouped_jump_times,time_axis,observations,sample_thetas[-1],sigmav2,sigmaw2,sample_rates[-1],sample_measure,individual_log_likelihoods,x_means,x_covariances)
        #Compute the log likelihood for the new parameters
        previous_log_likelihood = np.sum(new_individual_log_likelihoods)

        jump_sizes_array = [arr for arr in grouped_jump_sizes.copy() if arr.size > 0]
        jump_times_array = [arr for arr in grouped_jump_times.copy() if arr.size > 0]
        jump_sizes_array = np.concatenate(jump_sizes_array)
        jump_times_array = np.concatenate(jump_times_array)
        jump_sizes = np.zeros((1,len(jump_sizes_array)))
        jump_times = np.zeros((1,len(jump_times_array)))
        jump_sizes[0,:] = jump_sizes_array
        jump_times[0,:] = jump_times_array

        sample_x_means.append(x_means)
        sample_x_covariances.append(x_covariances)

        sample_path = integrate_to_path(jump_sizes,jump_times,time_axis)
        sample_paths.append(sample_path)
        overall_acceptance_probabilities.append(overall_acceptance_probability)
        """
        Step 3
        jump rate sampling
        """
        sample_rate = exp_rate_post(jump_times, rate_alpha_prior, rate_beta_prior, num_samp=1)
        """
        Step 4
        jump density sampling
        """
        #Sample first the hyper-parameters
        dir_alpha = dir_alpha_post(jump_sizes,dir_alpha,dir_alpha_a,dir_alpha_b)#shape-scale convention for this function
        base_gamma_a,base_gamma_b = dir_gamma_base_params_post(jump_sizes,base_gamma_a,base_gamma_b,base_gamma_a_step_size,base_gamma_b_step_size,base_gamma_a_mean,base_gamma_a_std,base_gamma_b_mean,base_gamma_b_std)
        #Then sample the measure conditional on the hyper-parameters sampled
        sample_measure = js_dist_post(jump_sizes,dir_alpha,dir_K,base_gamma_a,base_gamma_b)
        sample_measures.append(sample_measure)
        sample_rates.append(sample_rate)
        dir_alphas.append(dir_alpha)
        base_gamma_as.append(base_gamma_a)
        base_gamma_bs.append(base_gamma_b)


        

    
    theta_acceptance_probabilities = np.exp(theta_acceptance_probabilities)
    overall_theta_acceptance_probability = theta_accept_count/num_iter
    #Return lists of samples, each of which is a list of num_iter length 
    if return_states:
        return sample_measures,sample_rates,sample_paths,sample_thetas,overall_theta_acceptance_probability,sample_x_means,sample_x_covariances,jump_sizes,jump_times,dir_alphas,base_gamma_as,base_gamma_bs

    else:   
        return sample_measures,sample_rates,sample_paths,sample_thetas,overall_theta_acceptance_probability,sample_x_means,sample_x_covariances,dir_alphas,base_gamma_as,base_gamma_bs







#Shape-scale convention for Gamma here, opposite to the other function.
def stationary_domain_langevin_lm_theta_inference_all_DP_alpha_inferred_Gamma_base_Gaussian_update_conditional_sigmaw2(num_iter,block_width,overlapping_width,observations,time_axis,initial_jump_sizes,initial_jump_times,sigmav2,rate_alpha_prior,rate_beta_prior,dir_K,dir_alpha_a,dir_alpha_b,base_gamma_a_mean,base_gamma_a_std,base_gamma_b_mean,base_gamma_b_std,sigmaw2,theta_prior,theta_step_size,base_gamma_a_step_size,base_gamma_b_step_size,return_states=False,base_gamma_a=False,base_gamma_b=False,dir_alpha=False,muw_prior_mean = 0.0,muw_prior_variance=1.0):
    """
    Additional Inputs:
        sigmav2 instead of kv for the conditional sampler. Note that sigmav2 is the variance but not the standard deviation
        sigmaw2 instead of alphaw and betaw for direct conditioning.


         Note that the Gamma densities here all follow the shape-scale convention as in np.random.gamma
        dir_alpha_a : a parameter in the Gamma(a,b) prior for the Dirichlet process alpha parameter
        dir_alpha_b : b parameter same as above
        base_gamma_a_mean : Gaussian prior N(base_a_mean,base_a_std^2)for the first a parameter in the Gamma(a',b') base density of the DP
        base_gamma_a_std  : same as above. std is used insteade of var to cater for the numpy convention
        base_gamma_b_mean : Gaussian prior N(base_b_mean,base_b_std^2)for the second b parameter in the Gamma(a',b') base density of the DP
        base_gamma_b_std
        base_gamma_a_step_size,base_gamma_b_step_size: GRW proposal std sizes
    """
    #Initialise the prior system structure
    #alpha_prior and beta_prior are prior on the exponential rate parameter
    A = np.zeros((2, 2))
    A[0, 1] = 1.0
    A[1, 1] = theta_prior
    h = np.zeros((2,1))
    h[1,0] = 1.0
    g = np.zeros((1,3))
    #####################g[0,0] = 1 #Only the integral term x is being observed
    g[0,1] = 1 #Change the observed state from x to dx/dt for the case of observing the stationary domain directly.
    R = np.array([sigmav2])


    #Initial subordiantor jump sizes and times
    jump_times = initial_jump_times
    jump_sizes = initial_jump_sizes


    #Initialise Kalman filter
    mean_prior = np.zeros((3,1))
    mean_prior[2,0] = muw_prior_mean
    covariance_prior = np.zeros((3,3))
    covariance_prior[2,2] = muw_prior_variance
    grouped_jump_sizes,grouped_jump_times = group_jumps(jump_sizes,jump_times,time_axis)
    
    #Create the initial Kalman reference path for the MH step within Gibbs.
    #The previous_log_likelihood is stored for the latter MH step for updating the system structure
    previous_log_likelihood, individual_log_likelihoods,x_means,x_covariances = compute_log_likelihood_langevin_conditional_on_sigmaw2(mean_prior,covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,sigmaw2)
    prior_path = integrate_to_path(jump_sizes,jump_times,time_axis)

    #Initialize the samples
    """
    Initial jump rate sampling
    """
    sample_rate = exp_rate_post(jump_times, rate_alpha_prior, rate_beta_prior, num_samp=1)
    """
    Initial jump density sampling
    dir_alpha is now obtained from the initialized posterior distirbution for it
    """
    #Initially, either sample the hyper-parameters from the priors for initialization or read from the initialized values
    if not dir_alpha:
        dir_alpha = np.random.gamma(dir_alpha_a,dir_alpha_b)#Shape-scale convention
    else:
        dir_alpha = dir_alpha
    if not base_gamma_a:
        base_gamma_a = np.random.normal(base_gamma_a_mean,base_gamma_a_std) #Note that the numpy normal uses std but not var
    else:
        base_gamma_a = base_gamma_a
    if not base_gamma_b:
        base_gamma_b = np.random.normal(base_gamma_b_mean,base_gamma_b_std)
    else:
        base_gamma_b = base_gamma_b
    sample_measure = js_dist_post(jump_sizes,dir_alpha,dir_K,base_gamma_a,base_gamma_b)

    #Containers of the samples
    dir_alphas = [dir_alpha]
    base_gamma_as = [base_gamma_a]
    base_gamma_bs = [base_gamma_b]
    sample_paths = [prior_path]
    sample_measures = [sample_measure]
    sample_rates = [sample_rate]
    sample_thetas = [theta_prior]
    sample_x_means=[x_means]
    sample_x_covariances = [x_covariances]
    overall_acceptance_probabilities = []
    theta_acceptance_probabilities = np.zeros(num_iter)
    theta_accept_count = 0
    for iter in tqdm(range(num_iter),desc="Langevin System Inference Progress:"):
        """
        Step 1
        theta parameter sampling
        """
        proposed_theta = sample_thetas[-1] + np.random.randn()*theta_step_size
            #Update the Langevin matrix
        A[1, 1] = proposed_theta
        #No need to update alphaw_post and betaw_post here, since we only want the likelihood
        new_log_likelihood, _,_,_ = compute_log_likelihood_langevin_conditional_on_sigmaw2(mean_prior,covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,sigmaw2)
        theta_acceptance_probability = new_log_likelihood-previous_log_likelihood
        theta_acceptance_probabilities[iter] = theta_acceptance_probability
        if np.log(np.random.rand())<= theta_acceptance_probability:# Acceptance case
            sample_thetas.append(proposed_theta)
            #previous_log_likelihood = new_log_likelihood
            theta_accept_count += 1
        else: #Rejection case
            sample_thetas.append(sample_thetas[-1])

        """
        Step 2
        jump sizes sampling
        """
        #Sample jump sizes and times and update the Kalman reference path
        #Conditioning on the previous samples update the hidden states
        grouped_jump_sizes,grouped_jump_times,new_individual_log_likelihoods,x_means,x_covariances,overall_acceptance_probability = overlapping_block_updates_conditional_on_sigmaw2_variable_system_structure(block_width,overlapping_width,grouped_jump_sizes,grouped_jump_times,time_axis,observations,sigmaw2,sample_rate,sample_measure,individual_log_likelihoods,x_means,x_covariances,A,h,g,R)
        #Compute the log likelihood for the new parameters
        previous_log_likelihood = np.sum(new_individual_log_likelihoods)

        jump_sizes_array = [arr for arr in grouped_jump_sizes.copy() if arr.size > 0]
        jump_times_array = [arr for arr in grouped_jump_times.copy() if arr.size > 0]
        jump_sizes_array = np.concatenate(jump_sizes_array)
        jump_times_array = np.concatenate(jump_times_array)
        jump_sizes = np.zeros((1,len(jump_sizes_array)))
        jump_times = np.zeros((1,len(jump_times_array)))
        jump_sizes[0,:] = jump_sizes_array
        jump_times[0,:] = jump_times_array

        sample_x_means.append(x_means)
        sample_x_covariances.append(x_covariances)

        sample_path = integrate_to_path(jump_sizes,jump_times,time_axis)
        sample_paths.append(sample_path)
        overall_acceptance_probabilities.append(overall_acceptance_probability)
        """
        Step 3
        jump rate sampling
        """
        sample_rate = exp_rate_post(jump_times, rate_alpha_prior, rate_beta_prior, num_samp=1)
        """
        Step 4
        jump density sampling
        """
        #Sample first the hyper-parameters
        dir_alpha = dir_alpha_post(jump_sizes,dir_alpha,dir_alpha_a,dir_alpha_b)#shape-scale convention for this function
        base_gamma_a,base_gamma_b = dir_gamma_base_params_post(jump_sizes,base_gamma_a,base_gamma_b,base_gamma_a_step_size,base_gamma_b_step_size,base_gamma_a_mean,base_gamma_a_std,base_gamma_b_mean,base_gamma_b_std)
        #Then sample the measure conditional on the hyper-parameters sampled
        sample_measure = js_dist_post(jump_sizes,dir_alpha,dir_K,base_gamma_a,base_gamma_b)
        sample_measures.append(sample_measure)
        sample_rates.append(sample_rate)
        dir_alphas.append(dir_alpha)
        base_gamma_as.append(base_gamma_a)
        base_gamma_bs.append(base_gamma_b)


        

    
    theta_acceptance_probabilities = np.exp(theta_acceptance_probabilities)
    overall_theta_acceptance_probability = theta_accept_count/num_iter
    #Return lists of samples, each of which is a list of num_iter length 
    if return_states:
        return sample_measures,sample_rates,sample_paths,sample_thetas,overall_theta_acceptance_probability,sample_x_means,sample_x_covariances,jump_sizes,jump_times,dir_alphas,base_gamma_as,base_gamma_bs

    else:   
        return sample_measures,sample_rates,sample_paths,sample_thetas,overall_theta_acceptance_probability,sample_x_means,sample_x_covariances,dir_alphas,base_gamma_as,base_gamma_bs







#Alternative Initialization Algorithm in the Stationary Domain
#Shape-scale convention for Gamma here, opposite to the other function.
def stationary_domain_alternative_initialization_langevin_lm_theta_inference_all_DP_alpha_inferred_Gamma_base_Gaussian_update_conditional_sigmaw2(num_iter,block_width,overlapping_width,observations,time_axis,sigmav2,rate_alpha_prior,rate_beta_prior,dir_K,dir_alpha_a,dir_alpha_b,base_gamma_a_mean,base_gamma_a_std,base_gamma_b_mean,base_gamma_b_std,sigmaw2,theta_prior,theta_step_size,base_gamma_a_step_size,base_gamma_b_step_size,return_states=False,base_gamma_a=False,base_gamma_b=False,dir_alpha=False,muw_prior_mean = 0.0,muw_prior_variance=1.0):
    """
    Additional Inputs:
        sigmav2 instead of kv for the conditional sampler. Note that sigmav2 is the variance but not the standard deviation
        sigmaw2 instead of alphaw and betaw for direct conditioning.


         Note that the Gamma densities here all follow the shape-scale convention as in np.random.gamma
        dir_alpha_a : a parameter in the Gamma(a,b) prior for the Dirichlet process alpha parameter
        dir_alpha_b : b parameter same as above
        base_gamma_a_mean : Gaussian prior N(base_a_mean,base_a_std^2)for the first a parameter in the Gamma(a',b') base density of the DP
        base_gamma_a_std  : same as above. std is used insteade of var to cater for the numpy convention
        base_gamma_b_mean : Gaussian prior N(base_b_mean,base_b_std^2)for the second b parameter in the Gamma(a',b') base density of the DP
        base_gamma_b_std
        base_gamma_a_step_size,base_gamma_b_step_size: GRW proposal std sizes
    """
    #Initialise the prior system structure
    #alpha_prior and beta_prior are prior on the exponential rate parameter
    A = np.zeros((2, 2))
    A[0, 1] = 1.0
    A[1, 1] = theta_prior
    h = np.zeros((2,1))
    h[1,0] = 1.0
    g = np.zeros((1,3))
    #####################g[0,0] = 1 #Only the integral term x is being observed
    g[0,1] = 1 #Change the observed state from x to dx/dt for the case of observing the stationary domain directly.
    R = np.array([sigmav2])





    #Initialization

    initial_rate = np.random.gamma(rate_alpha_prior,1/rate_beta_prior,1) #We follow the shape-rate convnetion, which is the opposite to the shape-scale convention in numpy.
    
    #Initially, either sample the hyper-parameters from the priors for initialization or read from the initialized values
    if not dir_alpha:
        dir_alpha = np.random.gamma(dir_alpha_a,1/dir_alpha_b)#Shape-rate to shape-scale convention
    else:
        dir_alpha = dir_alpha
    if not base_gamma_a:
        base_gamma_a = np.random.normal(base_gamma_a_mean,base_gamma_a_std) #Note that the numpy normal uses std but not var
    else:
        base_gamma_a = base_gamma_a
    if not base_gamma_b:
        base_gamma_b = np.random.normal(base_gamma_b_mean,base_gamma_b_std)
    else:
        base_gamma_b = base_gamma_b

    sample_measure = js_dist_post(np.zeros((1, 0)),dir_alpha,dir_K,base_gamma_a,base_gamma_b) #Feed an empty jump sizes for prior sampling

    jump_sizes,jump_times = compound_poisson_DP(initial_rate,sample_measure,time_axis[-1])

    #Initialise Kalman filter
    mean_prior = np.zeros((3,1))
    mean_prior[2,0] = muw_prior_mean
    covariance_prior = np.zeros((3,3))
    covariance_prior[2,2] = muw_prior_variance
    grouped_jump_sizes,grouped_jump_times = group_jumps(jump_sizes,jump_times,time_axis)
    
    #Create the initial Kalman reference path for the MH step within Gibbs.
    #The previous_log_likelihood is stored for the latter MH step for updating the system structure
    previous_log_likelihood, individual_log_likelihoods,x_means,x_covariances = compute_log_likelihood_langevin_conditional_on_sigmaw2(mean_prior,covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,sigmaw2)
    prior_path = integrate_to_path(jump_sizes,jump_times,time_axis)



    #Containers of the samples
    dir_alphas = [dir_alpha]
    base_gamma_as = [base_gamma_a]
    base_gamma_bs = [base_gamma_b]
    sample_paths = [prior_path]
    sample_measures = [sample_measure]
    sample_rates = [initial_rate]
    sample_thetas = [theta_prior]
    sample_x_means=[x_means]
    sample_x_covariances = [x_covariances]
    overall_acceptance_probabilities = []
    theta_acceptance_probabilities = np.zeros(num_iter)
    theta_accept_count = 0
    for iter in tqdm(range(num_iter),desc="Langevin System Inference Progress:"):
        """
        Step 1
        theta parameter sampling
        """
        proposed_theta = sample_thetas[-1] + np.random.randn()*theta_step_size
            #Update the Langevin matrix
        A[1, 1] = proposed_theta
        #No need to update alphaw_post and betaw_post here, since we only want the likelihood
        new_log_likelihood, _,_,_ = compute_log_likelihood_langevin_conditional_on_sigmaw2(mean_prior,covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,sigmaw2)
        theta_acceptance_probability = new_log_likelihood-previous_log_likelihood
        theta_acceptance_probabilities[iter] = theta_acceptance_probability
        if np.log(np.random.rand())<= theta_acceptance_probability:# Acceptance case
            sample_thetas.append(proposed_theta)
            #previous_log_likelihood = new_log_likelihood
            theta_accept_count += 1
        else: #Rejection case
            sample_thetas.append(sample_thetas[-1])

        """
        Step 2
        jump sizes sampling
        """
        #Sample jump sizes and times and update the Kalman reference path
        #Conditioning on the previous samples update the hidden states
        grouped_jump_sizes,grouped_jump_times,new_individual_log_likelihoods,x_means,x_covariances,overall_acceptance_probability = overlapping_block_updates_conditional_on_sigmaw2_variable_system_structure(block_width,overlapping_width,grouped_jump_sizes,grouped_jump_times,time_axis,observations,sigmaw2,sample_rates[-1],sample_measure,individual_log_likelihoods,x_means,x_covariances,A,h,g,R)
        #Compute the log likelihood for the new parameters
        previous_log_likelihood = np.sum(new_individual_log_likelihoods)

        jump_sizes_array = [arr for arr in grouped_jump_sizes.copy() if arr.size > 0]
        jump_times_array = [arr for arr in grouped_jump_times.copy() if arr.size > 0]
        jump_sizes_array = np.concatenate(jump_sizes_array)
        jump_times_array = np.concatenate(jump_times_array)
        jump_sizes = np.zeros((1,len(jump_sizes_array)))
        jump_times = np.zeros((1,len(jump_times_array)))
        jump_sizes[0,:] = jump_sizes_array
        jump_times[0,:] = jump_times_array

        sample_x_means.append(x_means)
        sample_x_covariances.append(x_covariances)

        sample_path = integrate_to_path(jump_sizes,jump_times,time_axis)
        sample_paths.append(sample_path)
        overall_acceptance_probabilities.append(overall_acceptance_probability)
        """
        Step 3
        jump rate sampling
        """
        sample_rate = exp_rate_post(jump_times, rate_alpha_prior, rate_beta_prior, num_samp=1)
        """
        Step 4
        jump density sampling
        """
        #Sample first the hyper-parameters
        dir_alpha = dir_alpha_post(jump_sizes,dir_alpha,dir_alpha_a,dir_alpha_b)#shape-scale convention for this function
        base_gamma_a,base_gamma_b = dir_gamma_base_params_post(jump_sizes,base_gamma_a,base_gamma_b,base_gamma_a_step_size,base_gamma_b_step_size,base_gamma_a_mean,base_gamma_a_std,base_gamma_b_mean,base_gamma_b_std)
        #Then sample the measure conditional on the hyper-parameters sampled
        sample_measure = js_dist_post(jump_sizes,dir_alpha,dir_K,base_gamma_a,base_gamma_b)
        sample_measures.append(sample_measure)
        sample_rates.append(sample_rate)
        dir_alphas.append(dir_alpha)
        base_gamma_as.append(base_gamma_a)
        base_gamma_bs.append(base_gamma_b)


        

    
    theta_acceptance_probabilities = np.exp(theta_acceptance_probabilities)
    overall_theta_acceptance_probability = theta_accept_count/num_iter
    #Return lists of samples, each of which is a list of num_iter length 
    if return_states:
        return sample_measures,sample_rates,sample_paths,sample_thetas,overall_theta_acceptance_probability,sample_x_means,sample_x_covariances,jump_sizes,jump_times,dir_alphas,base_gamma_as,base_gamma_bs

    else:   
        return sample_measures,sample_rates,sample_paths,sample_thetas,overall_theta_acceptance_probability,sample_x_means,sample_x_covariances,dir_alphas,base_gamma_as,base_gamma_bs










#Gamma Process Prior case. Note that this is shape-rate convention, opposite to the above.
def langevin_lm_theta_inference_Gamma_Process_Prior_conditional_sigmaw2(num_iter,block_width,overlapping_width,observations,time_axis,initial_jump_sizes,initial_jump_times,sigmav2,GaP_beta,GaP_K,GaP_alpha_a,GaP_alpha_b,base_gamma_a_mean,base_gamma_a_std,base_gamma_b_mean,base_gamma_b_std,sigmaw2,theta_prior,theta_step_size,base_gamma_a_step_size,base_gamma_b_step_size,return_states=False,base_gamma_a=False,base_gamma_b=False,GaP_alpha=False,muw_prior_mean = 0.0,muw_prior_variance=1.0):
    """
    Also change kv to sigmav2, variance convention.
    Change alphaw and betaw to sigmaw2, variance convention again.

    There is not rate_gamma_alpha now, since it is now the same as the DP_alpha
    Additional Inputs: Note that the Gamma densities here all follow the shape-rate convention opposite to np.random.gamma
        GaP_beta: The Gamma process beta parameter, the rate parameter of the Gamma prior to the normalization constant/ rate parameter. The alpha parameter in the prior would be the same parameter to the DP prior.
        GaP_K: The finite GaP truncation parameter
        GaP_alpha_a : a parameter in the Gamma(a,b) prior for the Dirichlet process and normalization constant prior alpha parameter
        GaP_alpha_b : b parameter same as above

        base_gamma_a_mean : Gaussian prior N(base_a_mean,base_a_std^2)for the first a parameter in the Gamma(a',b') base density of the DP
        base_gamma_a_std  : same as above. std is used insteade of var to cater for the numpy convention
        base_gamma_b_mean : Gaussian prior N(base_b_mean,base_b_std^2)for the second b parameter in the Gamma(a',b') base density of the DP
        base_gamma_b_std
        base_gamma_a_step_size,base_gamma_b_step_size: GRW proposal std sizes
    """
    #Initialise the prior system structure
    #alpha_prior and beta_prior are prior on the exponential rate parameter
    A = np.zeros((2, 2))
    A[0, 1] = 1.0
    A[1, 1] = theta_prior
    h = np.zeros((2,1))
    h[1,0] = 1.0
    g = np.zeros((1,3))
    g[0,0] = 1 #Only the integral term x is being observed
    R = np.array([sigmav2])


    #Initial subordiantor jump sizes and times
    jump_times = initial_jump_times
    jump_sizes = initial_jump_sizes


    #Initialise Kalman filter
    mean_prior = np.zeros((3,1))
    mean_prior[2,0] = muw_prior_mean
    covariance_prior = np.zeros((3,3))
    covariance_prior[2,2] = muw_prior_variance
    grouped_jump_sizes,grouped_jump_times = group_jumps(jump_sizes,jump_times,time_axis)
    
    #Create the initial Kalman reference path for the MH step within Gibbs.
    #The previous_log_likelihood is stored for the latter MH step for updating the system structure
    previous_log_likelihood, individual_log_likelihoods,x_means,x_covariances = compute_log_likelihood_langevin_conditional_on_sigmaw2(mean_prior,covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,sigmaw2)
    prior_path = integrate_to_path(jump_sizes,jump_times,time_axis)

    #Initialize the samples
    """
    Initial jump density sampling
    """
    #Initially, either sample the hyper-parameters from the priors for initialization or keep the initialized values
    if not GaP_alpha:
        GaP_alpha = np.random.gamma(GaP_alpha_a,1/GaP_alpha_b)  #Shape-rate to shape-scale convention

    if not base_gamma_a:
        base_gamma_a = np.random.normal(base_gamma_a_mean,base_gamma_a_std) #Note that the numpy normal uses std but not var
    if not base_gamma_b:
        base_gamma_b = np.random.normal(base_gamma_b_mean,base_gamma_b_std)


    """
    Initial jump rate sampling
    """
    sample_rate = exp_rate_post(jump_times, GaP_alpha, GaP_beta, num_samp=1)

    sample_measure = js_dist_post(jump_sizes,GaP_alpha,GaP_K,base_gamma_a,base_gamma_b)

    #Containers of the samples
    GaP_alphas = []
    base_gamma_as = [base_gamma_a]
    base_gamma_bs = [base_gamma_b]
    sample_paths = [prior_path]
    sample_measures = [sample_measure]
    sample_rates = [sample_rate]
    sample_thetas = [theta_prior]
    sample_x_means=[x_means]
    sample_x_covariances = [x_covariances]
    overall_acceptance_probabilities = []
    theta_acceptance_probabilities = np.zeros(num_iter)
    theta_accept_count = 0
    for iter in tqdm(range(num_iter),desc="Langevin System Inference Progress:"):
        """
        Step 1
        theta parameter sampling
        """
        proposed_theta = sample_thetas[-1] + np.random.randn()*theta_step_size
            #Update the Langevin matrix
        A[1, 1] = proposed_theta
        #No need to update alphaw_post and betaw_post here, since we only want the likelihood
        new_log_likelihood, _,_,_ = compute_log_likelihood_langevin_conditional_on_sigmaw2(mean_prior,covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,sigmaw2)
        theta_acceptance_probability = new_log_likelihood-previous_log_likelihood
        theta_acceptance_probabilities[iter] = theta_acceptance_probability
        if np.log(np.random.rand())<= theta_acceptance_probability:# Acceptance case
            sample_thetas.append(proposed_theta)
            #previous_log_likelihood = new_log_likelihood
            theta_accept_count += 1
        else: #Rejection case
            sample_thetas.append(sample_thetas[-1])

        """
        Step 2
        jump sizes sampling
        """
        #Sample jump sizes and times and update the Kalman reference path
        #Conditioning on the previous samples update the hidden states
        grouped_jump_sizes,grouped_jump_times,new_individual_log_likelihoods,x_means,x_covariances,overall_acceptance_probability = overlapping_block_updates_conditional_on_sigmaw2(block_width,overlapping_width,grouped_jump_sizes,grouped_jump_times,time_axis,observations,sample_thetas[-1],sigmav2,sigmaw2,sample_rate,sample_measure,individual_log_likelihoods,x_means,x_covariances)
        #Compute the log likelihood for the new parameters
        previous_log_likelihood = np.sum(new_individual_log_likelihoods)

        jump_sizes_array = [arr for arr in grouped_jump_sizes.copy() if arr.size > 0]
        jump_times_array = [arr for arr in grouped_jump_times.copy() if arr.size > 0]
        jump_sizes_array = np.concatenate(jump_sizes_array)
        jump_times_array = np.concatenate(jump_times_array)
        jump_sizes = np.zeros((1,len(jump_sizes_array)))
        jump_times = np.zeros((1,len(jump_times_array)))
        jump_sizes[0,:] = jump_sizes_array
        jump_times[0,:] = jump_times_array

        sample_x_means.append(x_means)
        sample_x_covariances.append(x_covariances)

        sample_path = integrate_to_path(jump_sizes,jump_times,time_axis)
        sample_paths.append(sample_path)
        overall_acceptance_probabilities.append(overall_acceptance_probability)

        """
        Step 3
        jump rate sampling
        """
        sample_rate = exp_rate_post(jump_times, GaP_alpha, GaP_beta, num_samp=1)
        """
        Step 4
        jump density sampling
        """
        #Sample first the hyper-parameters
        GaP_alpha = GaP_alpha_post(jump_sizes,jump_times,GaP_beta,GaP_alpha_a,GaP_alpha_b)
        base_gamma_a,base_gamma_b = dir_gamma_base_params_post(jump_sizes,base_gamma_a,base_gamma_b,base_gamma_a_step_size,base_gamma_b_step_size,base_gamma_a_mean,base_gamma_a_std,base_gamma_b_mean,base_gamma_b_std)
        #Then sample the measure conditional on the hyper-parameters sampled
        sample_measure = js_dist_post(jump_sizes,GaP_alpha,GaP_K,base_gamma_a,base_gamma_b)
        sample_measures.append(sample_measure)
        sample_rates.append(sample_rate)
        GaP_alphas.append(GaP_alpha)
        base_gamma_as.append(base_gamma_a)
        base_gamma_bs.append(base_gamma_b)



        

    
    theta_acceptance_probabilities = np.exp(theta_acceptance_probabilities)
    overall_theta_acceptance_probability = theta_accept_count/num_iter
    #Return lists of samples, each of which is a list of num_iter length 
    if return_states:
        return sample_measures,sample_rates,sample_paths,sample_thetas,overall_theta_acceptance_probability,sample_x_means,sample_x_covariances,jump_sizes,jump_times,GaP_alphas,base_gamma_as,base_gamma_bs

    else:   
        return sample_measures,sample_rates,sample_paths,sample_thetas,overall_theta_acceptance_probability,sample_x_means,sample_x_covariances,GaP_alphas,base_gamma_as,base_gamma_bs




#The specific NVM Process Case
#Shape-scale convention for Gamma here, opposite to the other function.
def nvm_lm_theta_inference_all_DP_alpha_inferred_Gamma_base_Gaussian_update_conditional_sigmaw2(num_iter,block_width,overlapping_width,observations,time_axis,initial_jump_sizes,initial_jump_times,sigmav2,rate_alpha_prior,rate_beta_prior,dir_K,dir_alpha_a,dir_alpha_b,base_gamma_a_mean,base_gamma_a_std,base_gamma_b_mean,base_gamma_b_std,sigmaw2,base_gamma_a_step_size,base_gamma_b_step_size,return_states=False,base_gamma_a=False,base_gamma_b=False,dir_alpha=False,muw_prior_mean = 0.0,muw_prior_variance=1.0):
    """
    Additional Inputs:
        sigmav2 instead of kv for the conditional sampler. Note that sigmav2 is the variance but not the standard deviation
        sigmaw2 instead of alphaw and betaw for direct conditioning.


         Note that the Gamma densities here all follow the shape-scale convention as in np.random.gamma
        dir_alpha_a : a parameter in the Gamma(a,b) prior for the Dirichlet process alpha parameter
        dir_alpha_b : b parameter same as above
        base_gamma_a_mean : Gaussian prior N(base_a_mean,base_a_std^2)for the first a parameter in the Gamma(a',b') base density of the DP
        base_gamma_a_std  : same as above. std is used insteade of var to cater for the numpy convention
        base_gamma_b_mean : Gaussian prior N(base_b_mean,base_b_std^2)for the second b parameter in the Gamma(a',b') base density of the DP
        base_gamma_b_std
        base_gamma_a_step_size,base_gamma_b_step_size: GRW proposal std sizes
    """
    #Initialise the prior system structure
    #alpha_prior and beta_prior are prior on the exponential rate parameter
    A = np.zeros((1, 1))
    h = np.ones((1,1))
    g = np.zeros((1,2))
    g[0,0] = 1 #Only the integral term x is being observed
    R = np.array([sigmav2])


    #Initial subordiantor jump sizes and times
    jump_times = initial_jump_times
    jump_sizes = initial_jump_sizes


    #Initialise Kalman filter
    mean_prior = np.zeros((2,1))
    mean_prior[1,0] = muw_prior_mean
    covariance_prior = np.zeros((2,2))
    covariance_prior[1,1] = muw_prior_variance
    grouped_jump_sizes,grouped_jump_times = group_jumps(jump_sizes,jump_times,time_axis)
    
    #Create the initial Kalman reference path for the MH step within Gibbs.
    #The previous_log_likelihood is stored for the latter MH step for updating the system structure
    previous_log_likelihood, individual_log_likelihoods,x_means,x_covariances = compute_log_likelihood_nvm_conditional_on_sigmaw2(mean_prior,covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,g,R,sigmaw2)
    prior_path = integrate_to_path(jump_sizes,jump_times,time_axis)

    #Initialize the samples
    """
    Initial jump rate sampling
    """
    sample_rate = exp_rate_post(jump_times, rate_alpha_prior, rate_beta_prior, num_samp=1)
    """
    Initial jump density sampling
    dir_alpha is now obtained from the initialized posterior distirbution for it
    """
    #Initially, either sample the hyper-parameters from the priors for initialization or read from the initialized values
    if not dir_alpha:
        dir_alpha = np.random.gamma(dir_alpha_a,dir_alpha_b)#Shape-scale convention
    else:
        dir_alpha = dir_alpha
    if not base_gamma_a:
        base_gamma_a = np.random.normal(base_gamma_a_mean,base_gamma_a_std) #Note that the numpy normal uses std but not var
    else:
        base_gamma_a = base_gamma_a
    if not base_gamma_b:
        base_gamma_b = np.random.normal(base_gamma_b_mean,base_gamma_b_std)
    else:
        base_gamma_b = base_gamma_b
    sample_measure = js_dist_post(jump_sizes,dir_alpha,dir_K,base_gamma_a,base_gamma_b)

    #Containers of the samples
    dir_alphas = [dir_alpha]
    base_gamma_as = [base_gamma_a]
    base_gamma_bs = [base_gamma_b]
    sample_paths = [prior_path]
    sample_measures = [sample_measure]
    sample_rates = [sample_rate]
    sample_x_means=[x_means]
    sample_x_covariances = [x_covariances]
    overall_acceptance_probabilities = []
    theta_acceptance_probabilities = np.zeros(num_iter)
    theta_accept_count = 0
    for iter in tqdm(range(num_iter),desc="Langevin System Inference Progress:"):

        """
        Step 1
        jump sizes sampling
        """
        #Sample jump sizes and times and update the Kalman reference path
        #Conditioning on the previous samples update the hidden states
        grouped_jump_sizes,grouped_jump_times,new_individual_log_likelihoods,x_means,x_covariances,overall_acceptance_probability = nvm_overlapping_block_updates_conditional_on_sigmaw2(block_width,overlapping_width,grouped_jump_sizes,grouped_jump_times,time_axis,observations,sigmav2,sigmaw2,sample_rate,sample_measure,individual_log_likelihoods,x_means,x_covariances)
        #Compute the log likelihood for the new parameters
        previous_log_likelihood = np.sum(new_individual_log_likelihoods)

        jump_sizes_array = [arr for arr in grouped_jump_sizes.copy() if arr.size > 0]
        jump_times_array = [arr for arr in grouped_jump_times.copy() if arr.size > 0]
        jump_sizes_array = np.concatenate(jump_sizes_array)
        jump_times_array = np.concatenate(jump_times_array)
        jump_sizes = np.zeros((1,len(jump_sizes_array)))
        jump_times = np.zeros((1,len(jump_times_array)))
        jump_sizes[0,:] = jump_sizes_array
        jump_times[0,:] = jump_times_array

        sample_x_means.append(x_means)
        sample_x_covariances.append(x_covariances)

        sample_path = integrate_to_path(jump_sizes,jump_times,time_axis)
        sample_paths.append(sample_path)
        overall_acceptance_probabilities.append(overall_acceptance_probability)
        """
        Step 2
        jump rate sampling
        """
        sample_rate = exp_rate_post(jump_times, rate_alpha_prior, rate_beta_prior, num_samp=1)
        """
        Step 3
        jump density sampling
        """
        #Sample first the hyper-parameters
        dir_alpha = dir_alpha_post(jump_sizes,dir_alpha,dir_alpha_a,dir_alpha_b)#shape-scale convention for this function
        base_gamma_a,base_gamma_b = dir_gamma_base_params_post(jump_sizes,base_gamma_a,base_gamma_b,base_gamma_a_step_size,base_gamma_b_step_size,base_gamma_a_mean,base_gamma_a_std,base_gamma_b_mean,base_gamma_b_std)
        #Then sample the measure conditional on the hyper-parameters sampled
        sample_measure = js_dist_post(jump_sizes,dir_alpha,dir_K,base_gamma_a,base_gamma_b)
        sample_measures.append(sample_measure)
        sample_rates.append(sample_rate)
        dir_alphas.append(dir_alpha)
        base_gamma_as.append(base_gamma_a)
        base_gamma_bs.append(base_gamma_b)


        

    if return_states:
        return sample_measures,sample_rates,sample_paths,sample_x_means,sample_x_covariances,jump_sizes,jump_times,dir_alphas,base_gamma_as,base_gamma_bs

    else:   
        return sample_measures,sample_rates,sample_paths,sample_x_means,sample_x_covariances,dir_alphas,base_gamma_as,base_gamma_bs

