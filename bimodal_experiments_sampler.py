#### This file contains the additional samplers to help the bimodal experiments
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
from scipy.stats import norm



#The tool for likelihood computation in the nvm process experiment
def compute_nvm_log_likelihood_single_site(sub_jump,nvm_jump,sigmaw2,muw_mean,muw_variance):
    """
    single-site update here
    """
    mean   = muw_mean * sub_jump
    var    = muw_variance * sub_jump**2 + sigmaw2 * sub_jump
    std    = np.sqrt(var)
    log_likelihood = norm.logpdf(nvm_jump, loc=mean, scale=std)
    return log_likelihood



def nvm_process_inference_sampler(num_iter,nvm_jumps,dir_K,dir_alpha_a,dir_alpha_b,base_gamma_a_mean,base_gamma_a_std,base_gamma_b_mean,base_gamma_b_std,sigmaw2,muw_mean,muw_variance,base_gamma_a_step_size,base_gamma_b_step_size,base_gamma_a=False,base_gamma_b=False,dir_alpha=False):
    """
    nvm_jumps: (1,Nj)

    """
    #The number of jumps is fixed in this setting
    Nj = np.shape(nvm_jumps)[1]
    
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


    #Drawing from a DP Prior
    sample_measure = js_dist_post(np.zeros((1, 0)),dir_alpha,dir_K,base_gamma_a,base_gamma_b) #Feed an empty jump sizes for prior sampling
    probabilities = sample_measure[0, :].copy()  # Shape (K,)
    locations = sample_measure[1, :]      # Shape (K,)


    #Drawing the prior subordinator jump sizes.
    #Note that we canot use the old multinomial darws and the  mapping scheme, since we have no jump times here and the assignment order then depends on the multinomial strutcure.
    sub_jump_sizes = np.random.choice(locations, size=Nj, p=probabilities) #(Nj,)
    sub_jump_sizes = sub_jump_sizes.reshape(1,-1)
    #Create the initial likelihood references
    previous_log_likelihoods = np.zeros(Nj)
    for i in range(Nj):
        sub_jump = sub_jump_sizes[0,i]
        nvm_jump = nvm_jumps[0,i]
        previous_log_likelihoods[i]= compute_nvm_log_likelihood_single_site(sub_jump,nvm_jump,sigmaw2,muw_mean,muw_variance)
    dir_alphas = [dir_alpha]
    base_gamma_as = [base_gamma_a]
    base_gamma_bs = [base_gamma_b]
    sample_measures = [sample_measure]
    
    for iter in tqdm(range(num_iter),desc="Langevin System Inference Progress:"):


        """
        Step 1
        jump density sampling
        """
        #Sample first the hyper-parameters
        dir_alpha = dir_alpha_post(sub_jump_sizes,dir_alpha,dir_alpha_a,dir_alpha_b)#shape-scale convention for this function
        base_gamma_a,base_gamma_b = dir_gamma_base_params_post(sub_jump_sizes,base_gamma_a,base_gamma_b,base_gamma_a_step_size,base_gamma_b_step_size,base_gamma_a_mean,base_gamma_a_std,base_gamma_b_mean,base_gamma_b_std)
        #Then sample the measure conditional on the hyper-parameters sampled
        sample_measure = js_dist_post(sub_jump_sizes,dir_alpha,dir_K,base_gamma_a,base_gamma_b)
        probabilities = sample_measure[0, :].copy()  # Shape (K,)
        locations = sample_measure[1, :]      # Shape (K,)
        sample_measures.append(sample_measure)
        dir_alphas.append(dir_alpha)
        base_gamma_as.append(base_gamma_a)
        base_gamma_bs.append(base_gamma_b)



        """
        Step 2
        jump sizes sampling
        """
        proposed_sub_jump_sizes = np.random.choice(locations, size=Nj, p=probabilities) #(Nj,) array also
        #Iterate over the single-site updates
        for i in range(Nj):
            proposed_sub_jump = proposed_sub_jump_sizes[i]
            nvm_jump = nvm_jumps[0,i]
            new_log_likelihood = compute_nvm_log_likelihood_single_site(proposed_sub_jump,nvm_jump,sigmaw2,muw_mean,muw_variance)
            acceptance_probability = new_log_likelihood-previous_log_likelihoods[i]
            if np.log(np.random.rand())<= acceptance_probability:# Acceptance case
                sub_jump_sizes[0,i] = proposed_sub_jump
                previous_log_likelihoods[i] = new_log_likelihood

        

    return np.array(sample_measures),np.array(dir_alphas),np.array(base_gamma_as),np.array(base_gamma_bs)

