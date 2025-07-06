import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from posteriors import*
from Common_Tools import*
from Levy_Generators import*
from Levy_State_Space import*









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
