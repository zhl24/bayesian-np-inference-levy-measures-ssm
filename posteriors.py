import numpy as np
from Common_Tools import*
from Levy_Generators import*
from Levy_State_Space import*
from numba import jit
import matplotlib.pyplot as plt
from tqdm import tqdm




def exp_rate_post(jump_times,alpha,beta,num_samp=1):
    #shape rate convention here
    #First convert the (1,c) jump time matrix into c dimensional array for simplicity
    jump_times = jump_times[0,:]
    sorted_jump_times = np.sort(jump_times)  # Ensure jump times are sorted
    sorted_jump_times = np.concatenate(([0.0], sorted_jump_times))   
    epochs = np.diff(sorted_jump_times)  #1 dimension removed due to the difference operation
    #The number of observations is c now
    # Posterior parameters for Gamma
    alpha = alpha + len(epochs)
    beta = beta + np.sum(epochs)
    #Note that the np.random.gamma takes the scale parameter but not the more popular beta parameter
    scale = 1/beta #Use the scale parameter for the numpy module
    rate_samples = np.random.gamma(alpha,scale,num_samp)
    return rate_samples



#The Gibbs kernel for sampling the DP alpha parameter using the auxilliary variable approach.
def dir_alpha_post(jump_sizes,previous_dir_alpha,dir_alpha_a,dir_alpha_b):
    """
    dir_alpha sampling via the auxilliary variable approach
    Note that we assume shape-rate convention throughout for Gamma distribution, which is the opposite to the numpy module convention.
    Inputs:
        jump_sizes: (1,c) matrix
        previous_dir_alpha: float, alpha value from the previous iteration
        dir_alpha_a,dir_alpha_b: prior Gamma(a,b) parameters on alpha
    """
    #get the number of unique components and components
    n = np.shape(jump_sizes)[1]
    k = np.unique(jump_sizes).size
    #Sample first the auxilliary x variable from the Beta distribution
    x = np.random.beta(previous_dir_alpha+1,n)
    #Sample then for the dir_alpha conditonal on x
    mixture_ratio = (dir_alpha_a+k-1.0)/(1.0/dir_alpha_b-np.log(x))/n
    pi = mixture_ratio/(1+mixture_ratio)
    #The mixture of Gamma sampling
    if np.random.rand() < pi:
        dir_alpha = np.random.gamma(dir_alpha_a+k,1.0/(1.0/dir_alpha_b-np.log(x)))
    else:
        dir_alpha = np.random.gamma(dir_alpha_a+k-1.0,1.0/(1.0/dir_alpha_b-np.log(x)))
    return dir_alpha





def dir_gamma_base_params_post(jump_sizes,base_gamma_a,base_gamma_b,base_gamma_a_step_size,base_gamma_b_step_size,base_gamma_a_mean,base_gamma_a_std,base_gamma_b_mean,base_gamma_b_std):
    """
    DP base measure parameters update via MH in Gibbs. Gibbs sampling for the 2 parameters separately for easier exploration
    Inputs:
        jump_sizes: (1,c) matrix
        base_gamma_a,base_gamma_b:prior Gamma(a,b) parameter samples for DP base density from the last iteration
        base_gamma_a_step_size,base_gamma_b_step_size: N(last_sample,step_size) GRW proposal for the MH
        base_gamma_a_mean,base_gamma_a_std,base_gamma_b_mean,base_gamma_b_std: Gaussian prior parameters for the base parameters
    """
    
    unique_jump_sizes = np.unique(jump_sizes) #Frequency/weights does not matter, only the positions
    #Compute the log likelihood for the initial parameters: prior + likelihood
    previous_log_lik = gaussian_log_likelihood(base_gamma_a,base_gamma_a_mean,base_gamma_a_std)+gaussian_log_likelihood(base_gamma_b,base_gamma_b_mean,base_gamma_b_std)+log_gamma_joint(unique_jump_sizes,base_gamma_a,base_gamma_b) 
    #Gibbs sampling
    proposed_base_gamma_a = np.random.normal(base_gamma_a,base_gamma_a_step_size)
    #Simply take it as 0 probability for proposed values in the non-positive regimes. More proper approach via the truncated normal
    if proposed_base_gamma_a >0:
        log_lik = gaussian_log_likelihood(proposed_base_gamma_a,base_gamma_a_mean,base_gamma_a_std)+gaussian_log_likelihood(base_gamma_b,base_gamma_b_mean,base_gamma_b_std)+ log_gamma_joint(unique_jump_sizes,proposed_base_gamma_a,base_gamma_b)
        if np.log(np.random.rand()) < log_lik-previous_log_lik: #Acceptance case
            base_gamma_a = proposed_base_gamma_a
            previous_log_lik = log_lik
        
    proposed_base_gamma_b = np.random.normal(base_gamma_b,base_gamma_b_step_size)
    #Same 0 probability idea as above
    if proposed_base_gamma_b >0:
        log_lik = gaussian_log_likelihood(base_gamma_a,base_gamma_a_mean,base_gamma_a_std)+gaussian_log_likelihood(proposed_base_gamma_b,base_gamma_b_mean,base_gamma_b_std) + log_gamma_joint(unique_jump_sizes,base_gamma_a,proposed_base_gamma_b) 
        if np.log(np.random.rand()) < log_lik-previous_log_lik: #Acceptance case
            base_gamma_b = proposed_base_gamma_b

    return base_gamma_a,base_gamma_b


def js_dist_post(jump_sizes,alpha,K,base_gamma_a=1.0,base_gamma_b=1.0):
    #Gamma(1,1) base distirbution is by default (quite a similar shape to levy densities), following the shape-scale convention
    #The jump sizes are directly observations from the distribution, and with shape (1,c)
    #alpha is the strength parameter for the Dirichlet process
    #K is the number of components truncated, and it should be greater than the effective number of components

    #Step 1: Sample stick-breaking weights in a vectorized manner
    jump_sizes = jump_sizes[0,:]
    #jump_sizes = jump_sizes[jump_sizes > 0.001]
    #Also note that the 0.0 jump sizes need to be removed
    M = len(jump_sizes) #The number of observations needed latter
    beta = np.random.beta(1, alpha + M, size=K - 1)  # K - 1 Beta samples
    beta = np.append(beta, 1.0) #No normalization needed after this step
    weights = np.zeros(K)
    weights[0] = beta[0] #The first weight is directly the beta sample so treated specially
    weights[1:] = beta[1:] * (1 - beta[:-1]).cumprod(axis=0)
    #weights /= weights.sum()
    # Step 2: Sample locations for each component
    cluster_locations = np.zeros(K)
    prior_prob = alpha / (alpha + M)
    for k in range(K):
        # Probability check to decide whether to sample from the base distribution or empirical data
        if np.random.rand() < prior_prob:
            # Sample from the base distribution H, assumed to be Gamma(1,1)
            cluster_locations[k] = np.random.gamma(base_gamma_a, base_gamma_b)  # NumPy's Gamma uses (shape, scale)
        else:
            # Sample from the empirical distribution of observed jump sizes
            cluster_locations[k] = np.random.choice(jump_sizes)

    # Step 3: Sort locations and reorder weights based on sorted locations
    sorted_indices = np.argsort(cluster_locations)
    sorted_weights = weights[sorted_indices]
    sorted_locations = cluster_locations[sorted_indices]
    # Combine weights and locations into a sample measure
    sample_measure = np.vstack((sorted_weights, sorted_locations))
    return sample_measure
#The sample measure should be essentially a matrix, with one axis being the probabilities and one axis being the delta positions


# Update function for each selected segment, e.g. blocks. Simple proposal
def segment_update(grouped_jump_sizes,grouped_jump_times,time_intervals,base_time,rate,sample_measure):
    #It is important to note that time points correspond to the ends of each time interval, essentially the t in the formulae.
    #Compound Poisson way of generating jump sizes and times over a fixed frame
    N = len(time_intervals) #The time intervals is an N dimensional array, and N is the total number of intervals
    #Extract information from the sample jump size distirbution first
    probabilities = sample_measure[0, :].copy()  # Shape (K,)
    locations = sample_measure[1, :]      # Shape (K,)
    for i in range(N): #Iteratively update each interval
        T = time_intervals[i] #The duration of the interval
        #Generate the number of jumps in the interval using a Poisson process
        poisson_rate = rate * T 
        Nj = np.random.poisson(lam=poisson_rate.item())
        #Simulate the jump times
        new_jump_times = np.random.uniform(low=0, high=T, size=Nj) #(Nj,) array
        new_jump_times += base_time
        base_time += time_intervals[i]
        #Simulate the jump sizes from multinomial
        counts = np.random.multinomial(Nj, probabilities)
        new_jump_sizes = np.repeat(locations, counts) #(Nj,) array also
        #Replace the corresponding jump sizes and times in the interval
        grouped_jump_sizes[i] = new_jump_sizes
        grouped_jump_times[i] = new_jump_times

    return grouped_jump_sizes, grouped_jump_times









#An alternative version of the algorithm above with the sigmaw posterior also updated and returned
def overlapping_block_updates_all(block_size,overlapping_size,grouped_jump_sizes,grouped_jump_times,time_axis,observations,theta,kv,alphaw,betaw,sample_rate,sample_measure,individual_log_likelihoods,previous_x_means,previous_x_covariances,alphaw_post,betaw_post):
    #Initialize the grouping and then assign the blocks first
    #grouped_jump_sizes,grouped_jump_times = group_jumps(jump_sizes,jump_times,time_axis) #N-1 dimensional list of arrays
    sequence_length = len(grouped_jump_sizes) #N-1 value
    block_indices = compute_overlapping_block_indices(block_size,overlapping_size,sequence_length) #(2,n) matrix
    start_indices = block_indices[0,:]
    end_indices = block_indices[1,:]
    time_intervals = np.diff(time_axis) #N-1
    #Define the Langevin dynamics and the emission strutcure
    A = np.zeros((2, 2))
    A[0, 1] = 1.0
    A[1, 1] = theta
    h = np.zeros((2,1))
    h[1,0] = 1.0
    g = np.zeros((1,3))
    g[0,0] = 1 #Only the integral term x is being observed
    R = np.array([kv])
    #We are 100% sure about hidden state starting from 0
        #Initialise the mean and covariance for the initial position
        #previous_X_mean = np.zeros((3,1))
        #previous_X_uncertainty = np.zeros((3,3))
        #previous_X_uncertainty[2,2] = 1.0 #Iniital uncertainty about muw is needed
    #Start the overlapping block update algorithm
    Nb = len(start_indices) #The number of blocks
    acceptance_probabilities = []
    accepted_count = 0
    #Assign the prior x means and uncertaintys
    previous_X_mean = previous_x_means[0,:,:].copy()
    previous_X_uncertainty = previous_x_covariances[0,:,:].copy()
    for i in range(Nb):
        #Copy the grouping reuslts for latter proposal use
        updated_jump_sizes = grouped_jump_sizes.copy()
        updated_jump_times = grouped_jump_times.copy()
        start_index = start_indices[i]
        end_index = end_indices[i]
        #block_observations = observations[:,start_index:].copy() #All the remaining observations need to be considered
        #The initial positions are not updated
        
        #The log likelihood of the whole interval to be updated. Ni-1 length. Log likelihood is also an interval term
        previous_log_likelihood = np.sum(individual_log_likelihoods) # Excluding the first point which is common and not updated
        base_time = time_axis[start_index]# already have 1 index lag due to the differential operation, so the start index is just the base time index
        #Extract the grouped jump sizes and times in the corresponding block
        jump_sizes_block = grouped_jump_sizes[start_index:end_index+1].copy()#Ni-1 intervals, excluding the end index. Automatic -1 
        jump_times_block = grouped_jump_times[start_index:end_index+1].copy()
        #Prior sampling here
        updated_grouped_jump_sizes,updated_grouped_jump_times = segment_update(jump_sizes_block,jump_times_block,time_intervals[start_index:end_index+1],base_time,sample_rate,sample_measure)
        
        updated_jump_sizes[start_index:end_index+1] = updated_grouped_jump_sizes.copy()
        updated_jump_times[start_index:end_index+1] = updated_grouped_jump_times.copy()

        #Compute the acceptance probability using the whole sequence, and propose the hidden states and posterior parameters also (not accepted yet). This is running the conditional inference here
        log_likelihood,individual_log_likelihoods_block,updated_x_means,updated_x_covariances,alphaw_post_proposed,betaw_post_proposed = compute_log_likelihood_langevin_all(previous_X_mean,previous_X_uncertainty,updated_jump_sizes,updated_jump_times,time_axis,observations,A,h,g,R,alphaw,betaw)
        log_acceptance_probability = log_likelihood - previous_log_likelihood
        acceptance_probabilities.append(log_acceptance_probability)
        #Acceptance case
        if np.log(np.random.rand()) <= log_acceptance_probability:
            accepted_count+=1
            #print("accepted!")
            individual_log_likelihoods = individual_log_likelihoods_block #The block has been changed to the whole sequence, so it doesn't matter.
            grouped_jump_sizes[start_index:end_index+1] = updated_grouped_jump_sizes
            grouped_jump_times[start_index:end_index+1] = updated_grouped_jump_times
            #The updated x means and covariances are interval objects, since the initial position is always nt updated. But we have included the inital position in the return to make them time objects
            previous_x_means = updated_x_means
            previous_x_covariances = updated_x_covariances
            alphaw_post = alphaw_post_proposed
            betaw_post = betaw_post_proposed
    
    overall_acceptance_probability = accepted_count/Nb
    return grouped_jump_sizes,grouped_jump_times,individual_log_likelihoods,previous_x_means,previous_x_covariances,overall_acceptance_probability,alphaw_post,betaw_post









