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


#This is the Bayesian MCMC algorithm for inferring the Levy measure from a Langevin Levy state space model
def langevin_lm_inference(num_iter,block_width,overlapping_width,observations,time_axis,initial_jump_sizes,initial_jump_times,theta,kv,rate_alpha_prior,rate_beta_prior,dir_alpha,dir_K,alphaw,betaw):
    #alpha_prior and beta_prior are prior on the exponential rate parameter
    A = np.zeros((2, 2))
    A[0, 1] = 1.0
    A[1, 1] = theta
    h = np.zeros((2,1))
    h[1,0] = 1.0
    g = np.zeros((1,3))
    g[0,0] = 1 #Only the integral term x is being observed
    R = np.array([kv])


    #Initial subordiantor jump sizes and times
    jump_times = initial_jump_times
    jump_sizes = initial_jump_sizes


    #Initialise Kalman filter
    covariance_prior = np.zeros((3,3))
    covariance_prior[2,2] = 1.0
    grouped_jump_sizes,grouped_jump_times = group_jumps(jump_sizes,jump_times,time_axis)
    
    #Create the initial Kalman reference path for the MH step within Gibbs
    log_likelihood, individual_log_likelihoods,x_means,x_covariances = compute_log_likelihood_langevin(np.zeros((3,1)),covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,alphaw,betaw)
    prior_path = integrate_to_path(jump_sizes,jump_times,time_axis)
    
    sample_rate = exp_rate_post(jump_times, rate_alpha_prior, rate_beta_prior, num_samp=1)
    sample_measure = js_dist_post(jump_sizes,dir_alpha,dir_K)

    #Containers of the samples
    sample_paths = []
    sample_measures = []
    sample_rates = []
    overall_acceptance_probabilities = []
    for iter in tqdm(range(num_iter),desc="Langevin System Inference Progress:"):
        #Sample jump sizes and times and update the Kalman reference path
        grouped_jump_sizes,grouped_jump_times,new_individual_log_likelihoods,x_means,x_covariances,overall_acceptance_probability = overlapping_block_updates(block_width,overlapping_width,grouped_jump_sizes,grouped_jump_times,time_axis,observations,theta,kv,alphaw,betaw,sample_rate,sample_measure,individual_log_likelihoods,x_means,x_covariances)
        jump_sizes_array = [arr for arr in grouped_jump_sizes.copy() if arr.size > 0]
        jump_times_array = [arr for arr in grouped_jump_times.copy() if arr.size > 0]
        jump_sizes_array = np.concatenate(jump_sizes_array)
        jump_times_array = np.concatenate(jump_times_array)
        jump_sizes = np.zeros((1,len(jump_sizes_array)))
        jump_times = np.zeros((1,len(jump_times_array)))
        jump_sizes[0,:] = jump_sizes_array
        jump_times[0,:] = jump_times_array


        sample_path = integrate_to_path(jump_sizes,jump_times,time_axis)
        sample_paths.append(sample_path)
        sample_measures.append(sample_measure)
        sample_rates.append(sample_rate)
        overall_acceptance_probabilities.append(overall_acceptance_probability)
        sample_rate = exp_rate_post(jump_times, rate_alpha_prior, rate_beta_prior, num_samp=1)
        sample_measure = js_dist_post(jump_sizes,dir_alpha,dir_K)

    #Return lists of samples    
    return sample_measures,sample_rates,sample_paths,overall_acceptance_probabilities



#The previous version but with additional hidden states returned
def langevin_lm_inference_all(num_iter,block_width,overlapping_width,observations,time_axis,initial_jump_sizes,initial_jump_times,theta,kv,rate_alpha_prior,rate_beta_prior,dir_alpha,dir_K,alphaw,betaw):
    #alpha_prior and beta_prior are prior on the exponential rate parameter
    A = np.zeros((2, 2))
    A[0, 1] = 1.0
    A[1, 1] = theta
    h = np.zeros((2,1))
    h[1,0] = 1.0
    g = np.zeros((1,3))
    g[0,0] = 1 #Only the integral term x is being observed
    R = np.array([kv])


    #Initial subordiantor jump sizes and times
    jump_times = initial_jump_times
    jump_sizes = initial_jump_sizes


    #Initialise Kalman filter
    covariance_prior = np.zeros((3,3))
    covariance_prior[2,2] = 1.0
    grouped_jump_sizes,grouped_jump_times = group_jumps(jump_sizes,jump_times,time_axis)
    
    #Create the initial Kalman reference path for the MH step within Gibbs
    previous_log_likelihood, individual_log_likelihoods,x_means,x_covariances,alphaw_post,betaw_post = compute_log_likelihood_langevin_all(np.zeros((3,1)),covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,alphaw,betaw)
    prior_path = integrate_to_path(jump_sizes,jump_times,time_axis)
    
    sample_rate = exp_rate_post(jump_times, rate_alpha_prior, rate_beta_prior, num_samp=1)
    sample_measure = js_dist_post(jump_sizes,dir_alpha,dir_K)

    #Containers of the samples
    sample_paths = [prior_path]
    sample_measures = [sample_measure]
    sample_rates = [sample_rate]
    sample_alphaw_posts = [alphaw_post]
    sample_betaw_posts = [betaw_post]
    sample_x_means=[x_means]
    sample_x_covariances = [x_covariances]
    overall_acceptance_probabilities = []

    for iter in tqdm(range(num_iter),desc="Langevin System Inference Progress:"):
        #Sample jump sizes and times and update the Kalman reference path
        grouped_jump_sizes,grouped_jump_times,new_individual_log_likelihoods,x_means,x_covariances,overall_acceptance_probability = overlapping_block_updates(block_width,overlapping_width,grouped_jump_sizes,grouped_jump_times,time_axis,observations,theta,kv,alphaw,betaw,sample_rate,sample_measure,individual_log_likelihoods,x_means,x_covariances)
        jump_sizes_array = [arr for arr in grouped_jump_sizes.copy() if arr.size > 0]
        jump_times_array = [arr for arr in grouped_jump_times.copy() if arr.size > 0]
        jump_sizes_array = np.concatenate(jump_sizes_array)
        jump_times_array = np.concatenate(jump_times_array)
        jump_sizes = np.zeros((1,len(jump_sizes_array)))
        jump_times = np.zeros((1,len(jump_times_array)))
        jump_sizes[0,:] = jump_sizes_array
        jump_times[0,:] = jump_times_array


        sample_path = integrate_to_path(jump_sizes,jump_times,time_axis)
        sample_paths.append(sample_path)
        overall_acceptance_probabilities.append(overall_acceptance_probability)
        sample_rate = exp_rate_post(jump_times, rate_alpha_prior, rate_beta_prior, num_samp=1)
        sample_measure = js_dist_post(jump_sizes,dir_alpha,dir_K)

        #Store the samples
        sample_measures.append(sample_measure)
        sample_rates.append(sample_rate)
        sample_x_means.append(x_means)
        sample_x_covariances.append(x_covariances)
        sample_alphaw_posts.append(alphaw_post)
        sample_betaw_posts.append(betaw_post)

    #Return lists of samples    
    return sample_measures,sample_rates,sample_paths,overall_acceptance_probabilities,sample_x_means,sample_x_covariances,sample_alphaw_posts,sample_betaw_posts





#This is the most basic version with sequential updates
#This tyee of algorithms forms the basis of the future work for sampling also the system structure.
#This is the Bayesian MCMC algorithm for inferring both the Levy measure and the theta parameter from a Langevin Levy state space model.
#note that the theta parameter input is now the theta parameter prior compared to the previous function without inferring the theta parameter
#A Gaussian random walk proposal will be used here for inferring theta parameter
def langevin_lm_theta_inference(num_iter,block_width,overlapping_width,observations,time_axis,initial_jump_sizes,initial_jump_times,kv,rate_alpha_prior,rate_beta_prior,dir_alpha,dir_K,alphaw,betaw,theta_prior,theta_step_size):
    #Initialise the prior system structure
    #alpha_prior and beta_prior are prior on the exponential rate parameter
    A = np.zeros((2, 2))
    A[0, 1] = 1.0
    A[1, 1] = theta_prior
    h = np.zeros((2,1))
    h[1,0] = 1.0
    g = np.zeros((1,3))
    g[0,0] = 1 #Only the integral term x is being observed
    R = np.array([kv])


    #Initial subordiantor jump sizes and times
    jump_times = initial_jump_times
    jump_sizes = initial_jump_sizes


    #Initialise Kalman filter
    covariance_prior = np.zeros((3,3))
    covariance_prior[2,2] = 1.0
    grouped_jump_sizes,grouped_jump_times = group_jumps(jump_sizes,jump_times,time_axis)
    
    #Create the initial Kalman reference path for the MH step within Gibbs.
    #The previous_log_likelihood is stored for the latter MH step for updating the system structure
    previous_log_likelihood, individual_log_likelihoods,x_means,x_covariances = compute_log_likelihood_langevin(np.zeros((3,1)),covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,alphaw,betaw)
    prior_path = integrate_to_path(jump_sizes,jump_times,time_axis)

    #Initialize the samples
    sample_rate = exp_rate_post(jump_times, rate_alpha_prior, rate_beta_prior, num_samp=1)
    sample_measure = js_dist_post(jump_sizes,dir_alpha,dir_K)

    #Containers of the samples
    sample_paths = []
    sample_measures = []
    sample_rates = []
    sample_thetas = [theta_prior]
    overall_acceptance_probabilities = []
    theta_acceptance_probabilities = np.zeros(num_iter)
    theta_accept_count = 0
    for iter in tqdm(range(num_iter),desc="Langevin System Inference Progress:"):
        #MH step for theta parameter sampling
        proposed_theta = sample_thetas[-1] + np.random.randn()*theta_step_size
            #Update the Langevin matrix
        A[1, 1] = proposed_theta
        new_log_likelihood, _,_,_ = compute_log_likelihood_langevin(np.zeros((3,1)),covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,alphaw,betaw)
        theta_acceptance_probability = new_log_likelihood-previous_log_likelihood
        theta_acceptance_probabilities[iter] = theta_acceptance_probability
        if np.log(np.random.rand())<= theta_acceptance_probability:# Acceptance case
            sample_thetas.append(proposed_theta)
            #previous_log_likelihood = new_log_likelihood
            theta_accept_count += 1
        else: #Rejection case
            sample_thetas.append(sample_thetas[-1])


        #Sample jump sizes and times and update the Kalman reference path
        #Conditioning on the previous samples
        grouped_jump_sizes,grouped_jump_times,new_individual_log_likelihoods,x_means,x_covariances,overall_acceptance_probability = overlapping_block_updates(block_width,overlapping_width,grouped_jump_sizes,grouped_jump_times,time_axis,observations,sample_thetas[-1],kv,alphaw,betaw,sample_rate,sample_measure,individual_log_likelihoods,x_means,x_covariances)
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


        sample_path = integrate_to_path(jump_sizes,jump_times,time_axis)
        sample_paths.append(sample_path)
        sample_measures.append(sample_measure)
        sample_rates.append(sample_rate)
        overall_acceptance_probabilities.append(overall_acceptance_probability)
        #Sample rate parameter and the jump size distribution
        sample_rate = exp_rate_post(jump_times, rate_alpha_prior, rate_beta_prior, num_samp=1)
        sample_measure = js_dist_post(jump_sizes,dir_alpha,dir_K)


        

    
    theta_acceptance_probabilities = np.exp(theta_acceptance_probabilities)
    overall_theta_acceptance_probability = theta_accept_count/num_iter
    #Return lists of samples    
    return sample_measures,sample_rates,sample_paths,sample_thetas,overall_theta_acceptance_probability





#A modified version of the algorithm above with all inference results returned
def langevin_lm_theta_inference_all(num_iter,block_width,overlapping_width,observations,time_axis,initial_jump_sizes,initial_jump_times,kv,rate_alpha_prior,rate_beta_prior,dir_alpha,dir_K,alphaw,betaw,theta_prior,theta_step_size,return_states=False):
    #Initialise the prior system structure
    #alpha_prior and beta_prior are prior on the exponential rate parameter
    A = np.zeros((2, 2))
    A[0, 1] = 1.0
    A[1, 1] = theta_prior
    h = np.zeros((2,1))
    h[1,0] = 1.0
    g = np.zeros((1,3))
    g[0,0] = 1 #Only the integral term x is being observed
    R = np.array([kv])


    #Initial subordiantor jump sizes and times
    jump_times = initial_jump_times
    jump_sizes = initial_jump_sizes


    #Initialise Kalman filter
    covariance_prior = np.zeros((3,3))
    covariance_prior[2,2] = 1.0
    grouped_jump_sizes,grouped_jump_times = group_jumps(jump_sizes,jump_times,time_axis)
    
    #Create the initial Kalman reference path for the MH step within Gibbs.
    #The previous_log_likelihood is stored for the latter MH step for updating the system structure
    previous_log_likelihood, individual_log_likelihoods,x_means,x_covariances,alphaw_post,betaw_post = compute_log_likelihood_langevin_all(np.zeros((3,1)),covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,alphaw,betaw)
    prior_path = integrate_to_path(jump_sizes,jump_times,time_axis)

    #Initialize the samples
    sample_rate = exp_rate_post(jump_times, rate_alpha_prior, rate_beta_prior, num_samp=1)
    sample_measure = js_dist_post(jump_sizes,dir_alpha,dir_K)

    #Containers of the samples
    sample_paths = [prior_path]
    sample_measures = [sample_measure]
    sample_rates = [sample_rate]
    sample_thetas = [theta_prior]
    sample_alphaw_posts = [alphaw_post]
    sample_betaw_posts = [betaw_post]
    sample_x_means=[x_means]
    sample_x_covariances = [x_covariances]
    overall_acceptance_probabilities = []
    theta_acceptance_probabilities = np.zeros(num_iter)
    theta_accept_count = 0
    for iter in tqdm(range(num_iter),desc="Langevin System Inference Progress:"):
        #MH step for theta parameter sampling
        proposed_theta = sample_thetas[-1] + np.random.randn()*theta_step_size
            #Update the Langevin matrix
        A[1, 1] = proposed_theta
        #No need to update alphaw_post and betaw_post here, since we only want the likelihood
        new_log_likelihood, _,_,_ = compute_log_likelihood_langevin(np.zeros((3,1)),covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,alphaw,betaw)
        theta_acceptance_probability = new_log_likelihood-previous_log_likelihood
        theta_acceptance_probabilities[iter] = theta_acceptance_probability
        if np.log(np.random.rand())<= theta_acceptance_probability:# Acceptance case
            sample_thetas.append(proposed_theta)
            #previous_log_likelihood = new_log_likelihood
            theta_accept_count += 1
        else: #Rejection case
            sample_thetas.append(sample_thetas[-1])


        #Sample jump sizes and times and update the Kalman reference path
        #Conditioning on the previous samples update the hidden states
        grouped_jump_sizes,grouped_jump_times,new_individual_log_likelihoods,x_means,x_covariances,overall_acceptance_probability,alphaw_post,betaw_post = overlapping_block_updates_all(block_width,overlapping_width,grouped_jump_sizes,grouped_jump_times,time_axis,observations,sample_thetas[-1],kv,alphaw,betaw,sample_rate,sample_measure,individual_log_likelihoods,x_means,x_covariances,alphaw_post,betaw_post)
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
        #Sample rate parameter and the jump size distribution
        sample_rate = exp_rate_post(jump_times, rate_alpha_prior, rate_beta_prior, num_samp=1)
        sample_measure = js_dist_post(jump_sizes,dir_alpha,dir_K)
        sample_measures.append(sample_measure)
        sample_rates.append(sample_rate)


        sample_alphaw_posts.append(alphaw_post)
        sample_betaw_posts.append(betaw_post)


        

    
    theta_acceptance_probabilities = np.exp(theta_acceptance_probabilities)
    overall_theta_acceptance_probability = theta_accept_count/num_iter
    #Return lists of samples, each of which is a list of num_iter length 
    if return_states:
        return sample_measures,sample_rates,sample_paths,sample_thetas,overall_theta_acceptance_probability,sample_x_means,sample_x_covariances,sample_alphaw_posts,sample_betaw_posts,jump_sizes,jump_times

    else:   
        return sample_measures,sample_rates,sample_paths,sample_thetas,overall_theta_acceptance_probability,sample_x_means,sample_x_covariances,sample_alphaw_posts,sample_betaw_posts





#An extended version of the algorithm above with DP hyper-parameters inferred
def langevin_lm_theta_inference_all_DP_alpha_inferred_Gamma_base_Gaussian_update(num_iter,block_width,overlapping_width,observations,time_axis,initial_jump_sizes,initial_jump_times,kv,rate_alpha_prior,rate_beta_prior,dir_K,dir_alpha_a,dir_alpha_b,base_gamma_a_mean,base_gamma_a_std,base_gamma_b_mean,base_gamma_b_std,alphaw,betaw,theta_prior,theta_step_size,base_gamma_a_step_size,base_gamma_b_step_size,return_states=False,base_gamma_a=False,base_gamma_b=False,dir_alpha=False):
    """
    Additional Inputs: Note that the Gamma densities here all follow the shape-scale convention as in np.random.gamma
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
    R = np.array([kv])


    #Initial subordiantor jump sizes and times
    jump_times = initial_jump_times
    jump_sizes = initial_jump_sizes


    #Initialise Kalman filter
    covariance_prior = np.zeros((3,3))
    covariance_prior[2,2] = 1.0
    grouped_jump_sizes,grouped_jump_times = group_jumps(jump_sizes,jump_times,time_axis)
    
    #Create the initial Kalman reference path for the MH step within Gibbs.
    #The previous_log_likelihood is stored for the latter MH step for updating the system structure
    previous_log_likelihood, individual_log_likelihoods,x_means,x_covariances,alphaw_post,betaw_post = compute_log_likelihood_langevin_all(np.zeros((3,1)),covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,alphaw,betaw)
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
    sample_measure = js_dist_post(jump_sizes,dir_alpha,dir_K,base_gamma_a,base_gamma_b)

    #Containers of the samples
    dir_alphas = [dir_alpha]
    base_gamma_as = [base_gamma_a]
    base_gamma_bs = [base_gamma_b]
    sample_paths = [prior_path]
    sample_measures = [sample_measure]
    sample_rates = [sample_rate]
    sample_thetas = [theta_prior]
    sample_alphaw_posts = [alphaw_post]
    sample_betaw_posts = [betaw_post]
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
        new_log_likelihood, _,_,_ = compute_log_likelihood_langevin(np.zeros((3,1)),covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,alphaw,betaw)
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
        grouped_jump_sizes,grouped_jump_times,new_individual_log_likelihoods,x_means,x_covariances,overall_acceptance_probability,alphaw_post,betaw_post = overlapping_block_updates_all(block_width,overlapping_width,grouped_jump_sizes,grouped_jump_times,time_axis,observations,sample_thetas[-1],kv,alphaw,betaw,sample_rate,sample_measure,individual_log_likelihoods,x_means,x_covariances,alphaw_post,betaw_post)
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
        dir_alpha = dir_alpha_post(jump_sizes,dir_alpha,dir_alpha_a,dir_alpha_b)
        base_gamma_a,base_gamma_b = dir_gamma_base_params_post(jump_sizes,base_gamma_a,base_gamma_b,base_gamma_a_step_size,base_gamma_b_step_size,base_gamma_a_mean,base_gamma_a_std,base_gamma_b_mean,base_gamma_b_std)
        #Then sample the measure conditional on the hyper-parameters sampled
        sample_measure = js_dist_post(jump_sizes,dir_alpha,dir_K,base_gamma_a,base_gamma_b)
        sample_measures.append(sample_measure)
        sample_rates.append(sample_rate)
        dir_alphas.append(dir_alpha)
        base_gamma_as.append(base_gamma_a)
        base_gamma_bs.append(base_gamma_b)
        sample_alphaw_posts.append(alphaw_post)
        sample_betaw_posts.append(betaw_post)


        

    
    theta_acceptance_probabilities = np.exp(theta_acceptance_probabilities)
    overall_theta_acceptance_probability = theta_accept_count/num_iter
    #Return lists of samples, each of which is a list of num_iter length 
    if return_states:
        return sample_measures,sample_rates,sample_paths,sample_thetas,overall_theta_acceptance_probability,sample_x_means,sample_x_covariances,sample_alphaw_posts,sample_betaw_posts,jump_sizes,jump_times,dir_alphas,base_gamma_as,base_gamma_bs

    else:   
        return sample_measures,sample_rates,sample_paths,sample_thetas,overall_theta_acceptance_probability,sample_x_means,sample_x_covariances,sample_alphaw_posts,sample_betaw_posts,dir_alphas,base_gamma_as,base_gamma_bs



#The same version as the one above but with alternative initialization
def alternative_initialization_langevin_lm_theta_inference_all_DP_alpha_inferred_Gamma_base_Gaussian_update(num_iter,block_width,overlapping_width,observations,time_axis,kv,rate_alpha_prior,rate_beta_prior,dir_K,dir_alpha_a,dir_alpha_b,base_gamma_a_mean,base_gamma_a_std,base_gamma_b_mean,base_gamma_b_std,alphaw,betaw,theta_prior,theta_step_size,base_gamma_a_step_size,base_gamma_b_step_size,return_states=False,base_gamma_a=False,base_gamma_b=False,dir_alpha=False):
    """
    Changed Inputs:
    Initial jump sizes and times removed, and the alternative initialization is used to draw a prior IGSDP, parameters of which have been provided already.

    Additional Inputs: Note that the Gamma densities here all follow the shape-scale convention as in np.random.gamma
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
    R = np.array([kv])




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
    covariance_prior = np.zeros((3,3))
    covariance_prior[2,2] = 1.0
    grouped_jump_sizes,grouped_jump_times = group_jumps(jump_sizes,jump_times,time_axis)
    
    #Create the initial Kalman reference path for the MH step within Gibbs.
    #The previous_log_likelihood is stored for the latter MH step for updating the system structure
    previous_log_likelihood, individual_log_likelihoods,x_means,x_covariances,alphaw_post,betaw_post = compute_log_likelihood_langevin_all(np.zeros((3,1)),covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,alphaw,betaw)
    prior_path = integrate_to_path(jump_sizes,jump_times,time_axis)


    

    

    #Containers of the samples
    dir_alphas = [dir_alpha]
    base_gamma_as = [base_gamma_a]
    base_gamma_bs = [base_gamma_b]
    sample_paths = [prior_path]
    sample_measures = [sample_measure]
    sample_rates = [initial_rate]
    sample_thetas = [theta_prior]
    sample_alphaw_posts = [alphaw_post]
    sample_betaw_posts = [betaw_post]
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
        new_log_likelihood, _,_,_ = compute_log_likelihood_langevin(np.zeros((3,1)),covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,alphaw,betaw)
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
        grouped_jump_sizes,grouped_jump_times,new_individual_log_likelihoods,x_means,x_covariances,overall_acceptance_probability,alphaw_post,betaw_post = overlapping_block_updates_all(block_width,overlapping_width,grouped_jump_sizes,grouped_jump_times,time_axis,observations,sample_thetas[-1],kv,alphaw,betaw,sample_rates[-1],sample_measure,individual_log_likelihoods,x_means,x_covariances,alphaw_post,betaw_post)
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
        dir_alpha = dir_alpha_post(jump_sizes,dir_alpha,dir_alpha_a,dir_alpha_b)
        base_gamma_a,base_gamma_b = dir_gamma_base_params_post(jump_sizes,base_gamma_a,base_gamma_b,base_gamma_a_step_size,base_gamma_b_step_size,base_gamma_a_mean,base_gamma_a_std,base_gamma_b_mean,base_gamma_b_std)
        #Then sample the measure conditional on the hyper-parameters sampled
        sample_measure = js_dist_post(jump_sizes,dir_alpha,dir_K,base_gamma_a,base_gamma_b)
        sample_measures.append(sample_measure)
        sample_rates.append(sample_rate)
        dir_alphas.append(dir_alpha)
        base_gamma_as.append(base_gamma_a)
        base_gamma_bs.append(base_gamma_b)
        sample_alphaw_posts.append(alphaw_post)
        sample_betaw_posts.append(betaw_post)


        

    
    theta_acceptance_probabilities = np.exp(theta_acceptance_probabilities)
    overall_theta_acceptance_probability = theta_accept_count/num_iter
    #Return lists of samples, each of which is a list of num_iter length 
    if return_states:
        return sample_measures,sample_rates,sample_paths,sample_thetas,overall_theta_acceptance_probability,sample_x_means,sample_x_covariances,sample_alphaw_posts,sample_betaw_posts,jump_sizes,jump_times,dir_alphas,base_gamma_as,base_gamma_bs

    else:   
        return sample_measures,sample_rates,sample_paths,sample_thetas,overall_theta_acceptance_probability,sample_x_means,sample_x_covariances,sample_alphaw_posts,sample_betaw_posts,dir_alphas,base_gamma_as,base_gamma_bs




def langevin_lm_theta_inference_Gamma_Process_Prior(num_iter,block_width,overlapping_width,observations,time_axis,initial_jump_sizes,initial_jump_times,kv,GaP_beta,GaP_K,GaP_alpha_a,GaP_alpha_b,base_gamma_a_mean,base_gamma_a_std,base_gamma_b_mean,base_gamma_b_std,alphaw,betaw,theta_prior,theta_step_size,base_gamma_a_step_size,base_gamma_b_step_size,return_states=False,base_gamma_a=False,base_gamma_b=False,GaP_alpha=False):
    """
    There is not rate_gamma_alpha now, since it is now the same as the DP_alpha
    Additional Inputs: Note that the Gamma densities here all follow the shape-scale convention as in np.random.gamma
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
    R = np.array([kv])


    #Initial subordiantor jump sizes and times
    jump_times = initial_jump_times
    jump_sizes = initial_jump_sizes


    #Initialise Kalman filter
    covariance_prior = np.zeros((3,3))
    covariance_prior[2,2] = 1.0
    grouped_jump_sizes,grouped_jump_times = group_jumps(jump_sizes,jump_times,time_axis)
    
    #Create the initial Kalman reference path for the MH step within Gibbs.
    #The previous_log_likelihood is stored for the latter MH step for updating the system structure
    previous_log_likelihood, individual_log_likelihoods,x_means,x_covariances,alphaw_post,betaw_post = compute_log_likelihood_langevin_all(np.zeros((3,1)),covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,alphaw,betaw)
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
    sample_alphaw_posts = [alphaw_post]
    sample_betaw_posts = [betaw_post]
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
        new_log_likelihood, _,_,_ = compute_log_likelihood_langevin(np.zeros((3,1)),covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,alphaw,betaw)
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
        grouped_jump_sizes,grouped_jump_times,new_individual_log_likelihoods,x_means,x_covariances,overall_acceptance_probability,alphaw_post,betaw_post = overlapping_block_updates_all(block_width,overlapping_width,grouped_jump_sizes,grouped_jump_times,time_axis,observations,sample_thetas[-1],kv,alphaw,betaw,sample_rate,sample_measure,individual_log_likelihoods,x_means,x_covariances,alphaw_post,betaw_post)
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
        sample_alphaw_posts.append(alphaw_post)
        sample_betaw_posts.append(betaw_post)


        

    
    theta_acceptance_probabilities = np.exp(theta_acceptance_probabilities)
    overall_theta_acceptance_probability = theta_accept_count/num_iter
    #Return lists of samples, each of which is a list of num_iter length 
    if return_states:
        return sample_measures,sample_rates,sample_paths,sample_thetas,overall_theta_acceptance_probability,sample_x_means,sample_x_covariances,sample_alphaw_posts,sample_betaw_posts,jump_sizes,jump_times,GaP_alphas,base_gamma_as,base_gamma_bs

    else:   
        return sample_measures,sample_rates,sample_paths,sample_thetas,overall_theta_acceptance_probability,sample_x_means,sample_x_covariances,sample_alphaw_posts,sample_betaw_posts,GaP_alphas,base_gamma_as,base_gamma_bs






"""This one is wrong!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
#The joint sampler here is not quite appropriate, so do not use it.
#This is the joint update algorithm for theta inference
#There is only a change in the update sequence in this algorithm, and the interface is exactly the same as the previous one with sequential updates
def langevin_lm_theta_joint_inference(num_iter,block_width,overlapping_width,observations,time_axis,initial_jump_sizes,initial_jump_times,kv,rate_alpha_prior,rate_beta_prior,dir_alpha,dir_K,alphaw,betaw,theta_prior,theta_step_size):
    #Initialise the prior system structure
    #alpha_prior and beta_prior are prior on the exponential rate parameter
    A = np.zeros((2, 2))
    A[0, 1] = 1.0
    A[1, 1] = theta_prior
    h = np.zeros((2,1))
    h[1,0] = 1.0
    g = np.zeros((1,3))
    g[0,0] = 1 #Only the integral term x is being observed
    R = np.array([kv])


    #Initial subordiantor jump sizes and times
    jump_times = initial_jump_times
    jump_sizes = initial_jump_sizes


    #Initialise Kalman filter
    covariance_prior = np.zeros((3,3))
    covariance_prior[2,2] = 1.0
    grouped_jump_sizes,grouped_jump_times = group_jumps(jump_sizes,jump_times,time_axis)
    
    #Create the initial Kalman reference path for the MH step within Gibbs.
    #The previous_log_likelihood is stored for the latter MH step for updating the system structure
    previous_log_likelihood, individual_log_likelihoods,x_means,x_covariances = compute_log_likelihood_langevin(np.zeros((3,1)),covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,alphaw,betaw)
    prior_path = integrate_to_path(jump_sizes,jump_times,time_axis)

    #Initialize the samples
    sample_rate = exp_rate_post(jump_times, rate_alpha_prior, rate_beta_prior, num_samp=1)
    sample_measure = js_dist_post(jump_sizes,dir_alpha,dir_K)
    sample_path = prior_path #The sample path needs to be initialized to the prior path in this algorithm.
    #Containers of the samples
    sample_paths = []
    sample_measures = []
    sample_rates = []
    sample_thetas = [theta_prior]
    overall_acceptance_probabilities = []
    theta_acceptance_probabilities = np.zeros(num_iter)
    theta_accept_count = 0
    for iter in tqdm(range(num_iter),desc="Langevin System Inference Progress:"):

        #Now put the subordinator sampling and the theta sampling steps into 1 step. Note that the following is just proposal only

        #MH step for theta parameter sampling, and this now also determines the acceptance for the jump sizes and times proposed due to the joint sampling scheme
        proposed_theta = sample_thetas[-1] + np.random.randn()*theta_step_size
        #Sample jump sizes and times and update the Kalman reference path
        #Conditioning on the previous samples. This step now makes proposal only
        updated_grouped_jump_sizes,updated_grouped_jump_times,new_individual_log_likelihoods_1,x_means,x_covariances,overall_acceptance_probability = overlapping_block_updates(block_width,overlapping_width,grouped_jump_sizes,grouped_jump_times,time_axis,observations,proposed_theta,kv,alphaw,betaw,sample_rate,sample_measure,individual_log_likelihoods,x_means,x_covariances)

            #overall_acceptance_probabilities.append(overall_acceptance_probability)        
            #Update the Langevin matrix
        A[1, 1] = proposed_theta
        #This is the check function for the validity
        #new_log_likelihood, new_individual_log_likelihoods_2,_,_ = compute_log_likelihood_langevin(np.zeros((3,1)),covariance_prior,updated_grouped_jump_sizes,updated_grouped_jump_times,time_axis,observations,A,h,g,R,alphaw,betaw)
        new_log_likelihood = np.sum(new_individual_log_likelihoods_1)
        theta_acceptance_probability = new_log_likelihood-previous_log_likelihood
        theta_acceptance_probabilities[iter] = theta_acceptance_probability
        if np.log(np.random.rand())<= theta_acceptance_probability:# Acceptance case
            sample_thetas.append(proposed_theta)
            previous_log_likelihood = new_log_likelihood
            theta_accept_count += 1
            grouped_jump_sizes = updated_grouped_jump_sizes
            grouped_jump_times = updated_grouped_jump_times
            jump_sizes_array = [arr for arr in grouped_jump_sizes.copy() if arr.size > 0]
            jump_times_array = [arr for arr in grouped_jump_times.copy() if arr.size > 0]
            jump_sizes_array = np.concatenate(jump_sizes_array)
            jump_times_array = np.concatenate(jump_times_array)
            jump_sizes = np.zeros((1,len(jump_sizes_array)))
            jump_times = np.zeros((1,len(jump_times_array)))
            jump_sizes[0,:] = jump_sizes_array
            jump_times[0,:] = jump_times_array
            sample_path = integrate_to_path(jump_sizes,jump_times,time_axis)
            sample_paths.append(sample_path)

        else: #Rejection case
            sample_thetas.append(sample_thetas[-1])
            sample_paths.append(sample_path)
            #Simply keep the previous gouped jump sizes and times if the proposal is rejected




        #Sample rate parameter and the jump size distribution. These are exact posterior sampling i.e. Gibbs steps so always accepted
        sample_rate = exp_rate_post(jump_times, rate_alpha_prior, rate_beta_prior, num_samp=1)
        sample_measure = js_dist_post(jump_sizes,dir_alpha,dir_K)
        sample_measures.append(sample_measure)
        sample_rates.append(sample_rate)



        

    
    theta_acceptance_probabilities = np.exp(theta_acceptance_probabilities)
    overall_theta_acceptance_probability = theta_accept_count/num_iter
    #Return lists of samples    
    return sample_measures,sample_rates,sample_paths,sample_thetas,overall_theta_acceptance_probability














#The reduced system inference case
def langevin_lm_theta_inference_all_reduced_system(num_iter,block_width,overlapping_width,observations,time_axis,initial_jump_sizes,initial_jump_times,kv,rate_alpha_prior,rate_beta_prior,dir_alpha,dir_K,alphaw,betaw,theta_prior,theta_step_size):
    #Initialise the prior system structure
    #alpha_prior and beta_prior are prior on the exponential rate parameter
    A = np.zeros((2, 2))
    A[0, 1] = 1.0
    A[1, 1] = theta_prior
    h = np.zeros((2,1))
    h[1,0] = 1.0
    g = np.zeros((1,3))
    #g[0,0] = 1 #Only the integral term x is being observed
    g[0,1] = 1 #Change the observed state from x to dx/dt for the case of observing the L\'evy process directly
    R = np.array([kv])


    #Initial subordiantor jump sizes and times
    jump_times = initial_jump_times
    jump_sizes = initial_jump_sizes


    #Initialise Kalman filter
    covariance_prior = np.zeros((3,3))
    covariance_prior[2,2] = 1.0
    grouped_jump_sizes,grouped_jump_times = group_jumps(jump_sizes,jump_times,time_axis)
    
    #Create the initial Kalman reference path for the MH step within Gibbs.
    #The previous_log_likelihood is stored for the latter MH step for updating the system structure
    previous_log_likelihood, individual_log_likelihoods,x_means,x_covariances,alphaw_post,betaw_post = compute_log_likelihood_langevin_all(np.zeros((3,1)),covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,alphaw,betaw)
    prior_path = integrate_to_path(jump_sizes,jump_times,time_axis)

    #Initialize the samples
    sample_rate = exp_rate_post(jump_times, rate_alpha_prior, rate_beta_prior, num_samp=1)
    sample_measure = js_dist_post(jump_sizes,dir_alpha,dir_K)

    #Containers of the samples
    sample_paths = [prior_path]
    sample_measures = [sample_measure]
    sample_rates = [sample_rate]
    sample_thetas = [theta_prior]
    sample_alphaw_posts = [alphaw_post]
    sample_betaw_posts = [betaw_post]
    sample_x_means=[x_means]
    sample_x_covariances = [x_covariances]
    overall_acceptance_probabilities = []
    theta_acceptance_probabilities = np.zeros(num_iter)
    theta_accept_count = 0
    for iter in tqdm(range(num_iter),desc="Langevin System Inference Progress:"):
        #MH step for theta parameter sampling
        proposed_theta = sample_thetas[-1] + np.random.randn()*theta_step_size
            #Update the Langevin matrix
        A[1, 1] = proposed_theta
        #No need to update alphaw_post and betaw_post here, since we only want the likelihood
        new_log_likelihood, _,_,_ = compute_log_likelihood_langevin(np.zeros((3,1)),covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,alphaw,betaw)
        theta_acceptance_probability = new_log_likelihood-previous_log_likelihood
        theta_acceptance_probabilities[iter] = theta_acceptance_probability
        if np.log(np.random.rand())<= theta_acceptance_probability:# Acceptance case
            sample_thetas.append(proposed_theta)
            #previous_log_likelihood = new_log_likelihood
            theta_accept_count += 1
        else: #Rejection case
            sample_thetas.append(sample_thetas[-1])


        #Sample jump sizes and times and update the Kalman reference path
        #Conditioning on the previous samples update the hidden states
        #g is not passed to the function overlapping_block_update() here, causing problem. This function always assumes observing only the integral state in noise.
        grouped_jump_sizes,grouped_jump_times,new_individual_log_likelihoods,x_means,x_covariances,overall_acceptance_probability,alphaw_post,betaw_post = overlapping_block_updates_all(block_width,overlapping_width,grouped_jump_sizes,grouped_jump_times,time_axis,observations,sample_thetas[-1],kv,alphaw,betaw,sample_rate,sample_measure,individual_log_likelihoods,x_means,x_covariances,alphaw_post,betaw_post)
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
        #Sample rate parameter and the jump size distribution
        sample_rate = exp_rate_post(jump_times, rate_alpha_prior, rate_beta_prior, num_samp=1)
        sample_measure = js_dist_post(jump_sizes,dir_alpha,dir_K)
        sample_measures.append(sample_measure)
        sample_rates.append(sample_rate)


        sample_alphaw_posts.append(alphaw_post)
        sample_betaw_posts.append(betaw_post)


        

    
    theta_acceptance_probabilities = np.exp(theta_acceptance_probabilities)
    overall_theta_acceptance_probability = theta_accept_count/num_iter
    #Return lists of samples, each of which is a list of num_iter length    
    return sample_measures,sample_rates,sample_paths,sample_thetas,overall_theta_acceptance_probability,sample_x_means,sample_x_covariances,sample_alphaw_posts,sample_betaw_posts
























#Experiments


#Common simualtion rate of 50
def _check_langevin_lm_inference():
    #Inference Parameters
    num_iter = 1000
    block_width = 30
    overlapping_width = 10
    rate_alpha_prior = 2 #Gamma base distirbution for the Dirichlet process prior
    rate_beta_prior = 1
    dir_alpha = 2 #Can be understood as 10 observations from the base Gamma distribution
    dir_K = 1000
    alphaw = 0.1
    betaw = 0.1

    #Generate simulated observations
    observation_noise_level = 0.3
    T = 10.0
    time_axis = np.linspace(0,T,int(T)*10)
    beta=2
    #alpha = 1/beta
    C=2
    muw = 2.0
    sigmaw = 1.0
    kv = (observation_noise_level/sigmaw)**2
    theta = -1.0
    sub_jump_sizes,jump_times = gamma_process_jumps((beta,C),T,sim_rate=50)
    original_path = integrate_to_path(sub_jump_sizes,jump_times,time_axis)
    nvm_jump_sizes = nvm_process_jumps(sub_jump_sizes,muw,sigmaw)
    langevin_path = langevin_hidden_response(nvm_jump_sizes,jump_times,theta,time_axis) #(2,N) process path
    full_observations = langevin_observations(langevin_path,observation_noise_level*np.identity(2))

    observations = np.zeros((1,np.shape(full_observations)[1]))
    observations[0,:] = full_observations[0,:]

    #plt.figure()
    #plt.plot(time_axis,observations[0,:])
    #plt.show()


    #Generate Initilisation of the jump sizes
    wrong_beta = beta
    wrong_C = 1*C
    initial_jump_sizes,initial_jump_times = gamma_process_jumps((wrong_beta,wrong_C),T,sim_rate=50)
    grouped_jump_sizes,grouped_jump_times = group_jumps(initial_jump_sizes,initial_jump_times,time_axis)
    prior_path = integrate_to_path(initial_jump_sizes,initial_jump_times,time_axis)

    #Perform MCMC sampler for the Levy measure and hidden sample paths
    #initial_jump_sizes = sub_jump_sizes
    #initial_jump_times = jump_times
    #prior_path = original_path
    sample_measures,sample_rates,sample_paths,overall_acceptance_probabilities = langevin_lm_inference(num_iter,block_width,overlapping_width,observations,time_axis,initial_jump_sizes,initial_jump_times,theta,kv,rate_alpha_prior,rate_beta_prior,dir_alpha,dir_K,alphaw,betaw)



    #Plot the sample paths as visuable results
    plt.figure(figsize=(12, 6))  # Set a larger figure size for better visibility
    for i in range(num_iter):
        new_path = sample_paths[i]
        #Change alpha_value to an iterative sequence for gradually chaging transparency
        alpha_value = 0.3    # 0.3 + 0.7 * (i / iter_num)  # Gradually increase alpha from 0.3 to 1.0
        plt.plot(time_axis, new_path[0, :], color='orange', alpha=alpha_value)

    
    plt.plot(time_axis,prior_path[0,:],label = "Prior Path",color = "green")
    plt.plot(time_axis, original_path[0, :], label="Original Path", color='blue')
    plt.title(f"Subordinator Path Samples \n Block Width: {block_width} Overlapping Width: {overlapping_width} \n Prior Rate Ratio: {wrong_C/C} Iteration: {num_iter}")
    plt.xlabel("Time")
    plt.ylabel("Path Value")
    plt.legend() 
    plt.grid(True, linestyle='--', alpha=0.5)  # Add a grid for better readability

    # Display the plot
    plt.tight_layout()  # Ensure the plot layout is neat
    plt.show()

    









    #Visualisations of the Measures
    #Combining the rate and jump size distribution into the Levy measure
    for i in range(len(sample_measures)):
        sample_rate = sample_rates[i]
        sample_measure = sample_measures[i] #2 x K
        sample_measure[0,:] = sample_measure[0,:] * sample_rate
        sample_measures[i] = sample_measure
    # Set target number of plots (e.g., 100)
    target_outputs = 100
    thinned_samples = thin_samples(sample_measures, target_outputs)
    # Initial plot setup
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)  # Adjust the bottom to fit the slider
    initial_sample = thinned_samples[0]
    stem_container = ax.stem(initial_sample[1], initial_sample[0], linefmt='b-', markerfmt='bo', basefmt=" ")
    ax.set_title(f'Sample Levy Density 1 \n Block Width: {block_width} Overlapping Width: {overlapping_width} \n Prior Rate Ratio: {wrong_C/C} Iteration: {num_iter}')
    ax.set_xlabel('Jump Sizes')
    ax.set_ylabel('Rates')

    # Slider setup
    ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor='lightgray')
    slider = Slider(ax_slider, 'Sample', 1, len(thinned_samples), valinit=1, valstep=1)
    # Connect the slider to the update function
    # Update function for the slider
    def update(val):
        sample_index = int(slider.val) - 1
        sample = thinned_samples[sample_index]
        
        # Clear and redraw the plot
        ax.cla()
        ax.stem(sample[1], sample[0], linefmt='b-', markerfmt='bo', basefmt=" ")
        ax.set_title(f'Sample Levy Density {sample_index + 1} \n Block Width: {block_width} Overlapping Width: {overlapping_width} \n Prior Rate Ratio: {wrong_C/C} Iteration: {num_iter}')
        ax.set_xlabel('Jump Sizes')
        ax.set_ylabel('Rates')
        fig.canvas.draw_idle()
    slider.on_changed(update)
    plt.show()

    # Initial plot setup
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)  # Adjust the bottom to fit the slider

    # Initial plot
    initial_sample = thinned_samples[0]
    stem_container = ax.stem(initial_sample[1], initial_sample[0], linefmt='b-', markerfmt='bo', basefmt=" ")
    ax.set_title(f'Sample Levy Density 1 \n Block Width: {block_width} Overlapping Width: {overlapping_width} \n Prior Rate Ratio: {wrong_C/C} Iteration: {num_iter}')
    ax.set_xlabel('Jump Sizes')
    ax.set_ylabel('Rates')

    # Update function for animation
    def update(frame):
        sample = thinned_samples[frame]
        ax.cla()  # Clear the current plot
        ax.stem(sample[1], sample[0], linefmt='b-', markerfmt='bo', basefmt=" ")
        ax.set_title(f'Sample Levy Density {frame + 1} \n Block Width: {block_width} Overlapping Width: {overlapping_width} \n Prior Rate Ratio: {wrong_C/C} Iteration: {num_iter}')
        ax.set_xlabel('Jump Sizes')
        ax.set_ylabel('Rates')

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(thinned_samples), repeat=False)

    # Save the animation as a GIF
    ani.save('sliding_effect.gif', writer=PillowWriter(fps=2))  # Adjust 'fps' as needed

    plt.show()

    return






#This is mainly used for experimenting threshold
#There should be a probability threshold
def _langevin_lm_inference_histogram():
    #Inference Parameters
    num_iter = 100000
    block_width = 30
    overlapping_width = 10  #It seems to be actually better for the overlapping width to be 0?
    rate_alpha_prior = 2 #Gamma base distirbution for the Dirichlet process prior
    rate_beta_prior = 1
    dir_alpha = 2 #Can be understood as 10 observations from the base Gamma distribution
    dir_K = 1000
    alphaw = 0.1
    betaw = 0.1

    sim_rate = 50 #The parameter c in the latex notes

    #Generate simulated observations
    observation_noise_level = 0.3
    T = 10.0
    time_axis = np.linspace(0,T,int(T)*10)
    beta=2
    #alpha = 1/beta
    C=2
    muw = 2.0
    sigmaw = 1.0
    kv = (observation_noise_level/sigmaw)**2
    theta = -1.0
    sub_jump_sizes,jump_times = gamma_process_jumps((beta,C),T,sim_rate)
    original_path = integrate_to_path(sub_jump_sizes,jump_times,time_axis)
    nvm_jump_sizes = nvm_process_jumps(sub_jump_sizes,muw,sigmaw)
    langevin_path = langevin_hidden_response(nvm_jump_sizes,jump_times,theta,time_axis) #(2,N) process path
    full_observations = langevin_observations(langevin_path,observation_noise_level*np.identity(2))

    observations = np.zeros((1,np.shape(full_observations)[1]))
    observations[0,:] = full_observations[0,:]

    #plt.figure()
    #plt.plot(time_axis,observations[0,:])
    #plt.show()


    #Generate Initilisation of the jump sizes
    wrong_beta = beta
    wrong_C = 1*C
    initial_jump_sizes,initial_jump_times = gamma_process_jumps((wrong_beta,wrong_C),T,sim_rate)
    grouped_jump_sizes,grouped_jump_times = group_jumps(initial_jump_sizes,initial_jump_times,time_axis)
    prior_path = integrate_to_path(initial_jump_sizes,initial_jump_times,time_axis)

    #Perform MCMC sampler for the Levy measure and hidden sample paths
    #initial_jump_sizes = sub_jump_sizes
    #initial_jump_times = jump_times
    #prior_path = original_path
    sample_measures,sample_rates,sample_paths,overall_acceptance_probabilities = langevin_lm_inference(num_iter,block_width,overlapping_width,observations,time_axis,initial_jump_sizes,initial_jump_times,theta,kv,rate_alpha_prior,rate_beta_prior,dir_alpha,dir_K,alphaw,betaw)



    #Plot the sample paths as visuable results
    plt.figure(figsize=(12, 6))  # Set a larger figure size for better visibility
    for i in range(num_iter):
        new_path = sample_paths[i]
        #Change alpha_value to an iterative sequence for gradually chaging transparency
        alpha_value = 0.3    # 0.3 + 0.7 * (i / iter_num)  # Gradually increase alpha from 0.3 to 1.0
        plt.plot(time_axis, new_path[0, :], color='orange', alpha=alpha_value)

        

    plt.plot(time_axis,prior_path[0,:],label = "Prior Path",color = "green")
    plt.plot(time_axis, original_path[0, :], label="Original Path", color='blue')
    plt.title(f"Subordinator Path Samples \n Block Width: {block_width} Overlapping Width: {overlapping_width} \n Prior Rate Ratio: {wrong_C/C} Iteration: {num_iter}")
    plt.xlabel("Time")
    plt.ylabel("Path Value")
    plt.legend() 
    plt.grid(True, linestyle='--', alpha=0.5)  # Add a grid for better readability

    # Display the plot
    plt.tight_layout()  # Ensure the plot layout is neat
    plt.show()

    

    overall_rates_list = []
    overall_positions_list = []

    # Averaging the sample measures
    # Combining the rate and jump size distribution into the Levy measure
    for i in range(len(sample_measures)):
        sample_rate = sample_rates[i]
        sample_measure = sample_measures[i]  # 2 x K
        sample_measure[0, :] = sample_measure[0, :] * sample_rate  # K-dimensional array
        sample_measures[i] = sample_measure
        
        rates = sample_measure[0, :]
        positions = sample_measure[1, :]
        
        # Collect arrays in lists
        overall_rates_list.append(rates)
        overall_positions_list.append(positions)

    # Concatenate arrays outside of the loop for better performance
    rates = np.concatenate(overall_rates_list)
    positions = np.concatenate(overall_positions_list)
    x_axis = np.linspace(min(positions),max(positions),100)
    result_axis,results = regularized_projection_estimate(positions, rates, int(len(x_axis)/2), regularized_power=0)
    #Average for the posterior mean
    results = results/num_iter
    #Plotting the projection estimate
    plt.plot(result_axis,results)
    density_axis = np.linspace(0.2,max(result_axis),100)
    #density_axis = np.linspace(min(result_axis),max(result_axis),100)
    gamma_density = C * density_axis**(-1) * np.exp(-beta * density_axis)
    plt.plot(density_axis, gamma_density, 'r-', lw=2, label='Theoretical Gamma Lévy Density')

    plt.xlabel("Jump Size")
    plt.ylabel("Density")
    plt.title(f"Posterior Mean and Theoretical Lévy Density")
    plt.legend()
    plt.show()
    return





#This is the experiment function for the langevin sampler for both the Levy measure and the theta parameter for the Langevin system
#The proposal for theta is just a simple Gaussian random walk, GRW
#The main difference from the previous algorithm is the additional step of defining the theta prior and proposal structure.
def _langevin_measure_theta_sampler_GRW_check():

    #Inference Parameters
    num_iter = 500000
    block_width = 20
    overlapping_width = 10
    rate_alpha_prior = 2 #Gamma base distirbution for the Dirichlet process prior
    rate_beta_prior = 1
    dir_alpha = 2 #Can be understood as 10 observations from the base Gamma distribution
    dir_K = 1000
    alphaw = 0.1
    betaw = 0.1

    sim_rate = 50 #The parameter c in the latex notes

    #Generate simulated observations
    observation_noise_level = 0.01
    T = 10.0
    time_axis = np.linspace(0,T,int(T)*10)
    beta=0.2
    #alpha = 1/beta
    C=0.2
    muw = 1.0
    sigmaw = 1.0
    kv = (observation_noise_level/sigmaw)**2
    theta = -1.0
    sub_jump_sizes,jump_times = gamma_process_jumps((beta,C),T,sim_rate)
    original_path = integrate_to_path(sub_jump_sizes,jump_times,time_axis)
    nvm_jump_sizes = nvm_process_jumps(sub_jump_sizes,muw,sigmaw)
    langevin_path = langevin_hidden_response(nvm_jump_sizes,jump_times,theta,time_axis) #(2,N) process path
    full_observations = langevin_observations(langevin_path,observation_noise_level*np.identity(2))

    observations = np.zeros((1,np.shape(full_observations)[1]))
    observations[0,:] = full_observations[0,:]

    
    plt.figure()
    plt.plot(time_axis,observations[0,:])
    plt.show()


    #Defining the theta prior and the GRW proposal step size
    theta_prior = theta * 1
    theta_step_size = 0.1 #This is the factor applied to the standard deviation. Square it to apply it to the variance.

    #Generate Initilisation of the jump sizes
    #wrong_beta = beta
    wrong_C = 1*C
    #initial_jump_sizes,initial_jump_times = gamma_process_jumps((wrong_beta,wrong_C),T,sim_rate)
    #grouped_jump_sizes,grouped_jump_times = group_jumps(initial_jump_sizes,initial_jump_times,time_axis)
    #prior_path = integrate_to_path(initial_jump_sizes,initial_jump_times,time_axis)

    #Exact Prior Case
    initial_jump_sizes = sub_jump_sizes
    initial_jump_times = jump_times
    grouped_jump_sizes,grouped_jump_times = group_jumps(initial_jump_sizes,initial_jump_times,time_axis)
    prior_path = original_path
    sample_measures,sample_rates,sample_paths,sample_thetas,overall_theta_acceptance_probability = langevin_lm_theta_inference(num_iter,block_width,overlapping_width,observations,time_axis,initial_jump_sizes,initial_jump_times,kv,rate_alpha_prior,rate_beta_prior,dir_alpha,dir_K,alphaw,betaw,theta_prior,theta_step_size)



    #Plot the sample paths as visuable results
    plt.figure(figsize=(12, 6))  # Set a larger figure size for better visibility
    for i in range(num_iter):
        new_path = sample_paths[i]
        #Change alpha_value to an iterative sequence for gradually chaging transparency
        alpha_value = 0.3    # 0.3 + 0.7 * (i / iter_num)  # Gradually increase alpha from 0.3 to 1.0
        plt.plot(time_axis, new_path[0, :], color='orange', alpha=alpha_value)

        

    plt.plot(time_axis,prior_path[0,:],label = "Prior Path",color = "green")
    plt.plot(time_axis, original_path[0, :], label="Original Path", color='blue')
    plt.title(f"Subordinator Path Samples \n Block Width: {block_width} Overlapping Width: {overlapping_width} \n Prior Rate Ratio: {wrong_C/C} Iteration: {num_iter}")
    plt.xlabel("Time")
    plt.ylabel("Path Value")
    plt.legend() 
    plt.grid(True, linestyle='--', alpha=0.5)  # Add a grid for better readability

    # Display the plot
    plt.tight_layout()  # Ensure the plot layout is neat
    plt.show()

    

    overall_rates_list = []
    overall_positions_list = []

    # Averaging the sample measures
    # Combining the rate and jump size distribution into the Levy measure
    for i in range(len(sample_measures)):
        sample_rate = sample_rates[i]
        sample_measure = sample_measures[i]  # 2 x K
        sample_measure[0, :] = sample_measure[0, :] * sample_rate  # K-dimensional array
        sample_measures[i] = sample_measure
        
        rates = sample_measure[0, :]
        positions = sample_measure[1, :]
        
        # Collect arrays in lists
        overall_rates_list.append(rates)
        overall_positions_list.append(positions)

    # Concatenate arrays outside of the loop for better performance
    rates = np.concatenate(overall_rates_list)
    positions = np.concatenate(overall_positions_list)
    x_axis = np.linspace(min(positions),max(positions),100)
    result_axis,results = regularized_projection_estimate(positions, rates, int(len(x_axis)/2), regularized_power=0)
    #Average for the posterior mean
    results = results/num_iter
    #Plotting the projection estimate
    plt.plot(result_axis,results)
    density_axis = np.linspace(0.1,max(result_axis),100)
    gamma_density = C * density_axis**(-1) * np.exp(-beta * density_axis)
    plt.plot(density_axis, gamma_density, 'r-', lw=2, label='Theoretical Gamma Lévy Density')

    plt.xlabel("Jump Size")
    plt.ylabel("Density")
    plt.title(f"Posterior Mean and Theoretical Lévy Density")
    plt.legend()
    plt.show()

    # Theta
    plt.figure(figsize=(8, 6))  # Adjust figure size as needed
    sns.histplot(sample_thetas, kde=True, stat="density", linewidth=0)
    plt.axvline(np.mean(sample_thetas), color='r', linestyle='--', label=f'Mean: {np.mean(sample_thetas):.2f}')
    plt.axvline(np.percentile(sample_thetas, 2.5), color='g', linestyle=':', label='95% CI')
    plt.axvline(np.percentile(sample_thetas, 97.5), color='g', linestyle=':')
    plt.axvline(theta, color='blue', linestyle='-', label=f'True: {theta}')
    plt.title(f'Posterior distribution of $\\theta$ \n Acceptance Probability: {overall_theta_acceptance_probability}')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(sample_thetas)
    plt.title(f"$\\theta$ Trace")
    plt.show()

    return





#The joint sampler here is not quite appropriate, so do not use it.
#The joint update code check
def _langevin_measure_theta_joint_sampler_GRW_check():

    #Inference Parameters
    num_iter = 10000
    block_width = 10
    overlapping_width = 5
    rate_alpha_prior = 2 #Gamma base distirbution for the Dirichlet process prior
    rate_beta_prior = 1
    dir_alpha = 2 #Can be understood as 10 observations from the base Gamma distribution
    dir_K = 1000
    alphaw = 0.1
    betaw = 0.1

    sim_rate = 50 #The parameter c in the latex notes

    #Generate simulated observations
    observation_noise_level = 0.3
    T = 10.0
    time_axis = np.linspace(0,T,int(T)*10)
    beta=2
    #alpha = 1/beta
    C=2
    muw = 2.0
    sigmaw = 1.0
    kv = (observation_noise_level/sigmaw)**2


    #The important parameter in this function:
    theta = -2.0


    sub_jump_sizes,jump_times = gamma_process_jumps((beta,C),T,sim_rate)
    original_path = integrate_to_path(sub_jump_sizes,jump_times,time_axis)
    nvm_jump_sizes = nvm_process_jumps(sub_jump_sizes,muw,sigmaw)
    langevin_path = langevin_hidden_response(nvm_jump_sizes,jump_times,theta,time_axis) #(2,N) process path
    full_observations = langevin_observations(langevin_path,observation_noise_level*np.identity(2))

    observations = np.zeros((1,np.shape(full_observations)[1]))
    observations[0,:] = full_observations[0,:]

    #plt.figure()
    #plt.plot(time_axis,observations[0,:])
    #plt.show()


    #Defining the theta prior and the GRW proposal step size
    theta_prior = theta * 1
    theta_step_size = 0.2 #This is the factor applied to the standard deviation. Square it to apply it to the variance.

    #Generate Initilisation of the jump sizes
    wrong_beta = beta
    wrong_C = 1*C
    initial_jump_sizes,initial_jump_times = gamma_process_jumps((wrong_beta,wrong_C),T,sim_rate)
    grouped_jump_sizes,grouped_jump_times = group_jumps(initial_jump_sizes,initial_jump_times,time_axis)
    prior_path = integrate_to_path(initial_jump_sizes,initial_jump_times,time_axis)

    #Perform MCMC sampler for the Levy measure and hidden sample paths
    #initial_jump_sizes = sub_jump_sizes
    #initial_jump_times = jump_times
    #prior_path = original_path
    sample_measures,sample_rates,sample_paths,sample_thetas,overall_theta_acceptance_probability = langevin_lm_theta_joint_inference(num_iter,block_width,overlapping_width,observations,time_axis,initial_jump_sizes,initial_jump_times,kv,rate_alpha_prior,rate_beta_prior,dir_alpha,dir_K,alphaw,betaw,theta_prior,theta_step_size)



    #Plot the sample paths as visuable results
    plt.figure(figsize=(12, 6))  # Set a larger figure size for better visibility
    for i in range(num_iter):
        new_path = sample_paths[i]
        #Change alpha_value to an iterative sequence for gradually chaging transparency
        alpha_value = 0.3    # 0.3 + 0.7 * (i / iter_num)  # Gradually increase alpha from 0.3 to 1.0
        plt.plot(time_axis, new_path[0, :], color='orange', alpha=alpha_value)

        

    plt.plot(time_axis,prior_path[0,:],label = "Prior Path",color = "green")
    plt.plot(time_axis, original_path[0, :], label="Original Path", color='blue')
    plt.title(f"Subordinator Path Samples \n Block Width: {block_width} Overlapping Width: {overlapping_width} \n Prior Rate Ratio: {wrong_C/C} Iteration: {num_iter}")
    plt.xlabel("Time")
    plt.ylabel("Path Value")
    plt.legend() 
    plt.grid(True, linestyle='--', alpha=0.5)  # Add a grid for better readability

    # Display the plot
    plt.tight_layout()  # Ensure the plot layout is neat
    plt.show()

    

    overall_rates_list = []
    overall_positions_list = []

    # Averaging the sample measures
    # Combining the rate and jump size distribution into the Levy measure
    for i in range(len(sample_measures)):
        sample_rate = sample_rates[i]
        sample_measure = sample_measures[i]  # 2 x K
        sample_measure[0, :] = sample_measure[0, :] * sample_rate  # K-dimensional array
        sample_measures[i] = sample_measure
        
        rates = sample_measure[0, :]
        positions = sample_measure[1, :]
        
        # Collect arrays in lists
        overall_rates_list.append(rates)
        overall_positions_list.append(positions)

    # Concatenate arrays outside of the loop for better performance
    rates = np.concatenate(overall_rates_list)
    positions = np.concatenate(overall_positions_list)
    x_axis = np.linspace(min(positions),max(positions),100)
    result_axis,results = regularized_projection_estimate(positions, rates, int(len(x_axis)/2), regularized_power=0)
    #Average for the posterior mean
    results = results/num_iter
    #Plotting the projection estimate
    plt.plot(result_axis,results)
    density_axis = np.linspace(0.1,max(result_axis),100)
    gamma_density = C * density_axis**(-1) * np.exp(-beta * density_axis)
    plt.plot(density_axis, gamma_density, 'r-', lw=2, label='Theoretical Gamma Lévy Density')

    plt.xlabel("Jump Size")
    plt.ylabel("Density")
    plt.title(f"Posterior Mean and Theoretical Lévy Density")
    plt.legend()
    plt.show()

    # Theta
    plt.figure(figsize=(8, 6))  # Adjust figure size as needed
    sns.histplot(sample_thetas, kde=True, stat="density", linewidth=0)
    plt.axvline(np.mean(sample_thetas), color='r', linestyle='--', label=f'Mean: {np.mean(sample_thetas):.2f}')
    plt.axvline(np.percentile(sample_thetas, 2.5), color='g', linestyle=':', label='95% CI')
    plt.axvline(np.percentile(sample_thetas, 97.5), color='g', linestyle=':')
    plt.axvline(theta, color='blue', linestyle='-', label=f'True: {theta}')
    plt.title(f'Posterior distribution of $\\theta$ \n Acceptance Probability: {overall_theta_acceptance_probability}')
    plt.legend()
    plt.show()

    return








#Wrong Prior Check
def _rough_sampler_implementation():
    observation_noise_level = 0.1
    T = 10.0
    time_axis = np.linspace(0,T,int(T)*10)
    beta=2
    alpha = 5
    C=2
    muw = 2.0
    sigmaw = 1.0
    kv = (observation_noise_level/sigmaw)**2
    theta = -1.0
    sub_jump_sizes,jump_times = gamma_process_jumps((beta,C),T,sim_rate=50)
    original_path = integrate_to_path(sub_jump_sizes,jump_times,time_axis)
    nvm_jump_sizes = nvm_process_jumps(sub_jump_sizes,muw,sigmaw)
    langevin_path = langevin_hidden_response(nvm_jump_sizes,jump_times,theta,time_axis) #(2,N) process path
    full_observations = langevin_observations(langevin_path,observation_noise_level*np.identity(2))
    original_jump_sizes = sub_jump_sizes.copy()
    original_jump_times = jump_times.copy()
    observations = np.zeros((1,np.shape(full_observations)[1]))
    observations[0,:] = full_observations[0,:]

    
    #Assuming exact inference of the system parameters
    A = np.zeros((2, 2))
    A[0, 1] = 1.0
    A[1, 1] = theta
    h = np.zeros((2,1))
    h[1,0] = 1.0
    g = np.zeros((1,3))
    g[0,0] = 1 #Only the integral term x is being observed
    R = np.array([kv])
    alphaw = 0.1
    betaw = 0.1
    alpha_prior = 2            # Prior shape parameter for Gamma prior
    beta_prior = 1  
    K=100
    covariance_prior = np.zeros((3,3))
    covariance_prior[2,2] = 1.0


    #Generate jump sizes and times using some different parameters as inconsistent prior
    wrong_beta = beta
    wrong_C = C*1
    sub_jump_sizes,jump_times = gamma_process_jumps((wrong_beta,wrong_C),T,sim_rate=50)
    grouped_jump_sizes,grouped_jump_times = group_jumps(sub_jump_sizes,jump_times,time_axis)
    sample_rate = exp_rate_post(jump_times, alpha_prior, beta_prior, num_samp=1)
    sample_measure = js_dist_post(sub_jump_sizes,alpha,K)
    #Creating a wrong reference path from the wrong prior first
    log_likelihood, individual_log_likelihoods,x_means,x_covariances = compute_log_likelihood_langevin(np.zeros((3,1)),covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,alphaw,betaw)
    prior_path = integrate_to_path(sub_jump_sizes,jump_times,time_axis)
    #print(np.max(prior_path))

    # Plot the original path and the wrong prior path
    plt.figure(figsize=(12, 6))  # Set a larger figure size for better visibility
    



    # Set the number of new paths to plot
    iter_num = 50000

    
    new_paths = []
    overall_acceptance_probabilities = []
    # Generate and plot new paths with varying transparency
    for i in tqdm(range(iter_num),desc = "Progress"):
        grouped_jump_sizes,grouped_jump_times,new_individual_log_likelihoods,x_means,x_covariances,overall_acceptance_probability = overlapping_block_updates(30,10,grouped_jump_sizes,grouped_jump_times,time_axis,observations,theta,kv,alphaw,betaw,sample_rate,sample_measure,individual_log_likelihoods,x_means,x_covariances)
        overall_acceptance_probabilities.append(overall_acceptance_probability)
        new_jump_sizes_array = [arr for arr in grouped_jump_sizes if arr.size > 0]
        new_jump_times_array = [arr for arr in grouped_jump_times if arr.size > 0]
        new_jump_sizes_array = np.concatenate(new_jump_sizes_array)
        new_jump_times_array = np.concatenate(new_jump_times_array)
        new_jump_sizes = np.zeros((1,len(new_jump_sizes_array)))
        new_jump_times = np.zeros((1,len(new_jump_times_array)))
        new_jump_sizes[0,:] = new_jump_sizes_array
        new_jump_times[0,:] = new_jump_times_array
        new_path = integrate_to_path(new_jump_sizes,new_jump_times,time_axis)
        
        # Integrate to generate the new path
        new_path = integrate_to_path(new_jump_sizes, new_jump_times, time_axis)
        new_paths.append(new_path.copy())
        
        # Plot each new path with progressively denser opacity for newer paths
        alpha_value = 0.3 + 0.7 * (i / iter_num)  # Gradually increase alpha from 0.3 to 1.0
        plt.plot(time_axis, new_path[0, :], color='orange', alpha=alpha_value)
        sample_rate = exp_rate_post(new_jump_times, alpha_prior, beta_prior, num_samp=1)
        sample_measure = js_dist_post(new_jump_sizes,alpha,K)
    #print(np.max(new_path))
    # Add final styling
    plt.plot(time_axis, original_path[0, :], label="Original Path", color='blue')
    plt.plot(time_axis,prior_path[0,:],label = "Prior Path",color = "green")
    plt.title("Evolution of New Paths from the Original Path")
    plt.xlabel("Time")
    plt.ylabel("Path Value")
    plt.legend() 
    plt.grid(True, linestyle='--', alpha=0.5)  # Add a grid for better readability

    # Display the plot
    plt.tight_layout()  # Ensure the plot layout is neat
    plt.show()

    return





def main():
    #_check_langevin_lm_inference()
    #_rough_sampler_implementation()
    #_langevin_lm_inference_histogram()
    _langevin_measure_theta_sampler_GRW_check()

    #_langevin_measure_theta_joint_sampler_GRW_check()
    return

if __name__ == "__main__":
    main()
