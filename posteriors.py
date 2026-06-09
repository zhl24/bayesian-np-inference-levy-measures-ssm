import numpy as np
from scipy.linalg import expm #This is the automatic matrix expnent solver
import math
from scipy.special import logsumexp
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
    Note that we assume shape-scale convention throughout for Gamma distribution, the same as numpy.
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


# The Gibbs kernel for sampling GaP alpha parameter directly trhough the Gamma distribution
def GaP_alpha_post(jump_sizes,jump_times,beta,GaP_alpha_a,GaP_alpha_b,num_samp = 1):
    """
    GaP_alpha sampling. Note that the alpha here now is both the alpha for the DP distributed component and the Gamma distributed normalizatipn constant component.
    Note that we assume shape-rate convention throughout for Gamma distribution, which is the opposite to the numpy module convention.
    Inputs:
        jump_sizes: (1,c) matrix
        jump_times: (1,c) matrix
        beta: The Gamma prior parameter on the normalization constant
        GaP_alpha_a,GaP_alpha_b: prior Gamma(a,b) parameters on alpha
        num_samp: number of alpha samples needed, 1 by default.
    """
    #get the number of unique components and components
    M = np.shape(jump_sizes)[1]
    k = np.unique(jump_sizes).size
    #Convert the jump times into exponential samples
    jump_times = jump_times[0,:]
    sorted_jump_times = np.sort(jump_times)  # Ensure jump times are sorted
    sorted_jump_times = np.concatenate(([0.0], sorted_jump_times))   #Put a zero there to keep the first sample
    epochs = np.diff(sorted_jump_times)  #1 dimension removed due to the difference operation (M+1) -> M

    shape = GaP_alpha_a + k
    rate = GaP_alpha_b - np.log((beta/(beta+np.sum(epochs))))
    GaP_alpha = np.random.gamma(shape,1/rate,num_samp) #Note that the nimpy gamma module uses the scake parameter
    return GaP_alpha


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


#shape-scale convention here
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






#The full update function for the third posterior using overlapped block update
#This is just a particular implementation using all the previously defined functions
#Need an initial Kalman filtered path as the reference for the original probability for the MH algorithm. Note that this is the MH within Gibbs but not Gibbs, so history or original reference is needed. 
#The Kalman prior should just be the initial position with extremely small covariance.
def overlapping_block_updates(block_size,overlapping_size,grouped_jump_sizes,grouped_jump_times,time_axis,observations,theta,kv,alphaw,betaw,sample_rate,sample_measure,individual_log_likelihoods,previous_x_means,previous_x_covariances):
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

        #Compute the acceptance probability using the whole sequence
        log_likelihood,individual_log_likelihoods_block,updated_x_means,updated_x_covariances = compute_log_likelihood_langevin(previous_X_mean,previous_X_uncertainty,updated_jump_sizes,updated_jump_times,time_axis,observations,A,h,g,R,alphaw,betaw)
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
    
    overall_acceptance_probability = accepted_count/Nb
    return grouped_jump_sizes,grouped_jump_times,individual_log_likelihoods,previous_x_means,previous_x_covariances,overall_acceptance_probability




def overlapping_block_updates_conditional_on_sigmaw2(block_size,overlapping_size,grouped_jump_sizes,grouped_jump_times,time_axis,observations,theta,sigmav2,sigmaw2,sample_rate,sample_measure,individual_log_likelihoods,previous_x_means,previous_x_covariances,compute_smoothing = False):
    """
    Change again:
    kv to sigmawv2 for absolute observation noise variance.
    alphaw and betaw to sigamw2 for conditioning. Note again that it is the variance convention for both.
    """

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
    R = np.array([sigmav2])
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

        #Compute the acceptance probability using the whole sequence
        log_likelihood,individual_log_likelihoods_block,updated_x_means,updated_x_covariances = compute_log_likelihood_langevin_conditional_on_sigmaw2(previous_X_mean,previous_X_uncertainty,updated_jump_sizes,updated_jump_times,time_axis,observations,A,h,g,R,sigmaw2)
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
    
    overall_acceptance_probability = accepted_count/Nb

    if compute_smoothing:
        #Make a final pass for computing the smoothing distribution based on the final jump sizes and times.
        log_likelihood,individual_log_likelihoods_block,updated_x_means,updated_x_covariances,smoothed_means, smoothed_covariances = compute_log_likelihood_langevin_conditional_on_sigmaw2(previous_X_mean,previous_X_uncertainty,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,sigmaw2,compute_smoothing=compute_smoothing)
        return grouped_jump_sizes,grouped_jump_times,individual_log_likelihoods,previous_x_means,previous_x_covariances,overall_acceptance_probability, smoothed_means, smoothed_covariances
    else:
        return grouped_jump_sizes,grouped_jump_times,individual_log_likelihoods,previous_x_means,previous_x_covariances,overall_acceptance_probability



def overlapping_block_updates_conditional_on_sigmaw2_variable_system_structure(block_size,overlapping_size,grouped_jump_sizes,grouped_jump_times,time_axis,observations,sigmaw2,sample_rate,sample_measure,individual_log_likelihoods,previous_x_means,previous_x_covariances,A,h,g,R):
    """
    The only difference to the algorithm above is just it now needs also the system structure inputs,
    and it therefore does not need the sigmav2 and theta inputs.
    """

    #Initialize the grouping and then assign the blocks first
    #grouped_jump_sizes,grouped_jump_times = group_jumps(jump_sizes,jump_times,time_axis) #N-1 dimensional list of arrays
    sequence_length = len(grouped_jump_sizes) #N-1 value
    block_indices = compute_overlapping_block_indices(block_size,overlapping_size,sequence_length) #(2,n) matrix
    start_indices = block_indices[0,:]
    end_indices = block_indices[1,:]
    time_intervals = np.diff(time_axis) #N-1
   
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

        #Compute the acceptance probability using the whole sequence
        log_likelihood,individual_log_likelihoods_block,updated_x_means,updated_x_covariances = compute_log_likelihood_langevin_conditional_on_sigmaw2(previous_X_mean,previous_X_uncertainty,updated_jump_sizes,updated_jump_times,time_axis,observations,A,h,g,R,sigmaw2)
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
    
    overall_acceptance_probability = accepted_count/Nb
    return grouped_jump_sizes,grouped_jump_times,individual_log_likelihoods,previous_x_means,previous_x_covariances,overall_acceptance_probability



#The specific nvm process case
def nvm_overlapping_block_updates_conditional_on_sigmaw2(block_size,overlapping_size,grouped_jump_sizes,grouped_jump_times,time_axis,observations,sigmav2,sigmaw2,sample_rate,sample_measure,individual_log_likelihoods,previous_x_means,previous_x_covariances):
    """
    Change again:
    kv to sigmawv2 for absolute observation noise variance.
    alphaw and betaw to sigamw2 for conditioning. Note again that it is the variance convention for both.
    """

    #Initialize the grouping and then assign the blocks first
    #grouped_jump_sizes,grouped_jump_times = group_jumps(jump_sizes,jump_times,time_axis) #N-1 dimensional list of arrays
    sequence_length = len(grouped_jump_sizes) #N-1 value
    block_indices = compute_overlapping_block_indices(block_size,overlapping_size,sequence_length) #(2,n) matrix
    start_indices = block_indices[0,:]
    end_indices = block_indices[1,:]
    time_intervals = np.diff(time_axis) #N-1
    A = np.zeros((1, 1))
    h = np.ones((1,1))
    g = np.zeros((1,2))
    g[0,0] = 1 #Only the integral term x is being observed
    R = np.array([sigmav2])

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

        #Compute the acceptance probability using the whole sequence
        log_likelihood,individual_log_likelihoods_block,updated_x_means,updated_x_covariances = compute_log_likelihood_nvm_conditional_on_sigmaw2(previous_X_mean,previous_X_uncertainty,updated_jump_sizes,updated_jump_times,time_axis,observations,g,R,sigmaw2)
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
    
    overall_acceptance_probability = accepted_count/Nb
    return grouped_jump_sizes,grouped_jump_times,individual_log_likelihoods,previous_x_means,previous_x_covariances,overall_acceptance_probability




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



#IGSDP Prior Samples
def draw_IGSDP_Prior_Measures(rate_alpha,rate_beta,base_gamma_a,base_gamma_b,alpha_gamma_a,alpha_gamma_b,dir_K,num_samples):
    """
    All Gamma distributions here are assumed to be following the shape-rate convention

    This function draws sample measures from IGSDP(rate_alpha,rate_beta,alpha,H)
    a Gamma(base_gamma_a,base_gamma_b) is used as the base measure
    a Gamma(alpha_gamma_a,alpha_gamma_b) is put as a prior on the concentration parameter alpha

    Outputs: 
        sample_measurews: (num_samples,2,K). First row for probabilities, second row for positions
        sample_rates: (num_samples)
    """

    sample_rates = np.random.gamma(rate_alpha,1/rate_beta,num_samples)
    sample_measures = np.zeros((num_samples,2,dir_K))
    #Drawing the prior measures
    for i in range(num_samples):
        alpha = np.random.gamma(alpha_gamma_a,1/alpha_gamma_b)
        if alpha <= 1e-8:
            alpha = 1e-8
        sample_measures[i,:,:] = js_dist_post(np.zeros((1, 0)),alpha,dir_K,base_gamma_a,1/base_gamma_b)


    return sample_measures, sample_rates








#Check Functions

def _test_rate_posterior():
    true_lambda = 2.0          # Known true rate parameter for synthetic data
    num_observations = 500      # Number of observations to generate
    alpha_prior = 2            # Prior shape parameter for Gamma prior
    beta_prior = 1             # Prior rate parameter for Gamma prior
    num_samples = 5000         # Number of posterior samples to draw
    # Step 1: Generate synthetic data from Exponential distribution with known rate
    jump_times = np.cumsum(np.random.exponential(1/true_lambda, num_observations))
    jump_times = jump_times[np.newaxis, :]  # Reshape to (1, num_observations) as expected by exp_rate_post
    
    # Step 2: Perform inference using the exp_rate_post function
    posterior_samples = exp_rate_post(jump_times, alpha_prior, beta_prior, num_samples)
    
    # Step 3: Plot posterior samples and compare with true value
    plt.figure(figsize=(10, 6))
    plt.hist(posterior_samples, bins=30, density=True, alpha=0.6, color='skyblue', label="Posterior samples")
    plt.axvline(true_lambda, color='red', linestyle='--', label=f"True rate ($\lambda$ = {true_lambda})")
    plt.xlabel("Rate parameter (lambda)")
    plt.ylabel("Density")
    plt.title("Posterior Distribution of Rate Parameter")
    plt.legend()
    plt.show()
    return


#Try DP inference on exact observations for the jump sizes first
def _test_js_posterior():
    #Generate the jump process first
    T = 10.0
    time_axis = np.linspace(0,T,int(T)*10)
    beta=2
    C=2
    sub_jump_sizes,jump_times = gamma_process_jumps((beta,C),T,sim_rate=30)

    alpha = 10
    K=100
    alpha_prior = 2            # Prior shape parameter for Gamma prior
    beta_prior = 1  
    posterior_samples = exp_rate_post(jump_times, alpha_prior, beta_prior, num_samp=1)

    sample_measure = js_dist_post(sub_jump_sizes,alpha,K)
    probabilities = sample_measure[0,:]
    locations = sample_measure[1,:]

    plot_measure(locations,probabilities,posterior_samples)
    return



def _explicit_truncation_experimentation():
        #Generate the jump process first
    T = 10.0
    time_axis = np.linspace(0,T,int(T)*10)
    beta=2
    C=2
    jump_sizes,jump_times = gamma_process_jumps((beta,C),T,sim_rate=200)
    truncation_matrix = jump_sizes > 0.001
    jump_sizes = jump_sizes[truncation_matrix].reshape(1,-1)
    jump_times  =jump_times[truncation_matrix].reshape(1,-1)
    alpha = 10
    K=100
    alpha_prior = 2            # Prior shape parameter for Gamma prior
    beta_prior = 1  
    posterior_samples = exp_rate_post(jump_times, alpha_prior, beta_prior, num_samp=1)
    sample_measure = js_dist_post(jump_sizes,alpha,K)

    probabilities = sample_measure[0,:]
    locations = sample_measure[1,:]
    
    plot_measure(locations,probabilities,posterior_samples)
    return


#original_process_path = integrate_to_path(sub_jump_sizes,jump_times,time_axis) 

def _segment_update_test():
    T = 10.0
    time_axis = np.linspace(0,T,int(T)*10)
    beta=2
    C=2
    sub_jump_sizes,jump_times = gamma_process_jumps((beta,C),T,sim_rate=100)

    alpha = 10
    K=100
    alpha_prior = 2            # Prior shape parameter for Gamma prior
    beta_prior = 1  
    rate = exp_rate_post(jump_times, alpha_prior, beta_prior, num_samp=1)

    sample_measure = js_dist_post(sub_jump_sizes,alpha,K)
    original_process_path = integrate_to_path(sub_jump_sizes,jump_times,time_axis)
    #probabilities = sample_measure[0,:]
    #locations = sample_measure[1,:]
    grouped_jump_sizes,grouped_jump_times = group_jumps(sub_jump_sizes,jump_times,time_axis)
    time_intervals = np.diff(time_axis)
    grouped_jump_sizes,grouped_jump_times = segment_update(grouped_jump_sizes,grouped_jump_times,time_intervals,0.0,rate,sample_measure)
    new_jump_sizes = np.concatenate(grouped_jump_sizes)
    new_jump_times = np.concatenate(grouped_jump_times)
    jump_times = np.zeros((1,len(new_jump_times)))
    jump_times[0,:] = new_jump_times
    jump_sizes = np.zeros((1,len(new_jump_sizes)))
    jump_sizes[0,:] = new_jump_sizes
    new_process_path = integrate_to_path(jump_sizes,jump_times,time_axis)
    plt.plot(time_axis,original_process_path[0,:])
    plt.plot(time_axis,new_process_path[0,:])
    plt.show()
    return



def _check_overlapping_block_updates():
    observation_noise_level = 0.3
    T = 10.0
    time_axis = np.linspace(0,T,int(T)*10)
    beta=2
    alpha = 1/beta
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
    grouped_jump_sizes,grouped_jump_times = group_jumps(sub_jump_sizes,jump_times,time_axis)
    original_grouped_jump_sizes = grouped_jump_sizes.copy()
    original_grouped_jump_times = grouped_jump_times.copy()
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
    sample_rate = exp_rate_post(jump_times, alpha_prior, beta_prior, num_samp=1)
    sample_measure = js_dist_post(sub_jump_sizes,alpha,K)
    covariance_prior = np.zeros((3,3))
    covariance_prior[2,2] = 1.0
    #Creating a reference path first
    log_likelihood, individual_log_likelihoods,x_means,x_covariances,overall_acceptance_probability = compute_log_likelihood_langevin(np.zeros((3,1)),covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,alphaw,betaw)
    
    #Update also the reference path
    grouped_jump_sizes,grouped_jump_times = group_jumps(sub_jump_sizes,jump_times,time_axis)
    grouped_jump_sizes,grouped_jump_times,new_individual_log_likelihoods,x_means,x_covariances,overall_acceptance_probability = overlapping_block_updates(40,30,grouped_jump_sizes,grouped_jump_times,time_axis,observations,theta,kv,alphaw,betaw,sample_rate,sample_measure,individual_log_likelihoods,x_means,x_covariances)
    new_jump_sizes_array = [arr for arr in grouped_jump_sizes if arr.size > 0]
    new_jump_times_array = [arr for arr in grouped_jump_times if arr.size > 0]
    new_jump_sizes = np.zeros((1,len(new_jump_sizes_array)))
    new_jump_times = np.zeros((1,len(new_jump_times_array)))
    new_jump_sizes[0,:] = new_jump_sizes_array
    new_jump_times[0,:] = new_jump_times_array
    new_path = integrate_to_path(new_jump_sizes,new_jump_times,time_axis)


    #Subordinator path plot
    plt.figure()
    plt.plot(time_axis,original_path[0,:],label="original path")
    plt.plot(time_axis,new_path[0,:],label = "new path")
    plt.legend()
    plt.show()


    #Since the jump sizes and times are known already, another possible check is on the path corresponding to the jumps sampled, whcih should be more efficient.
    
    #Langevin system path plot
    original_langevin_path = langevin_path #(2,N) process path, the original langevin path responsieble for the observations
    new_nvm_jump_sizes = nvm_process_jumps(new_jump_sizes,muw,sigmaw)
    new_langevin_path = langevin_hidden_response(new_nvm_jump_sizes,new_jump_times,theta,time_axis)#(2,N) also
    #plt.figure()
    #plt.plot(time_axis,original_langevin_path[0,:],label = "Original x")
    #plt.plot(time_axis,new_langevin_path[0,:],label = "New x")
    #plt.legend()
    #plt.show()

    plt.figure()
    plt.plot(time_axis,original_langevin_path[0,:],label = "Original x")
    plt.plot(time_axis,observations[0,:],label = "Observations")
    plt.plot(time_axis,new_langevin_path[0,:],label = "New x")
    plt.legend()
    plt.show()

    return




#Exact Prior Check
def _iterative_check_overlapping_block_updates():
    observation_noise_level = 0.3
    T = 10.0
    time_axis = np.linspace(0,T,int(T)*10)
    beta=2
    alpha = 1/beta
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
    sample_rate = exp_rate_post(jump_times, alpha_prior, beta_prior, num_samp=1)
    sample_measure = js_dist_post(sub_jump_sizes,alpha,K)
    covariance_prior = np.zeros((3,3))
    covariance_prior[2,2] = 1.0


    #Generate jump sizes and times using some different parameters as inconsistent prior
    #wrong_beta = beta
    #wrong_C = C
    #sub_jump_sizes,jump_times = gamma_process_jumps((wrong_beta,wrong_C),T,sim_rate=50)
    grouped_jump_sizes,grouped_jump_times = group_jumps(sub_jump_sizes,jump_times,time_axis)
    #Creating a wrong reference path from the wrong prior first
    log_likelihood, individual_log_likelihoods,x_means,x_covariances = compute_log_likelihood_langevin(np.zeros((3,1)),covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,alphaw,betaw)
    prior_path = integrate_to_path(sub_jump_sizes,jump_times,time_axis)
    #print(np.max(prior_path))

    # Plot the original path and the wrong prior path
    plt.figure(figsize=(12, 6))  # Set a larger figure size for better visibility
    
    # Set the number of new paths to plot
    iter_num = 10000
    new_paths = []
    overall_acceptance_probabilities = []
    # Generate and plot new paths with varying transparency
    for i in tqdm(range(iter_num),desc = "Progress"):
        grouped_jump_sizes,grouped_jump_times,new_individual_log_likelihoods,x_means,x_covariances,overall_acceptance_probability = overlapping_block_updates(10,5,grouped_jump_sizes,grouped_jump_times,time_axis,observations,theta,kv,alphaw,betaw,sample_rate,sample_measure,individual_log_likelihoods,x_means,x_covariances)
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
        alpha_value = 0.3    # 0.3 + 0.7 * (i / iter_num)  # Gradually increase alpha from 0.3 to 1.0
        plt.plot(time_axis, new_path[0, :], color='orange', alpha=alpha_value)
    #print(np.max(new_path))
    plt.plot(time_axis, original_path[0, :], label="Original Path", color='blue')
    plt.plot(time_axis,prior_path[0,:],label = "Prior Path",color = "green")
    # Add final styling
    plt.title("Evolution of New Paths from the Original Path")
    plt.xlabel("Time")
    plt.ylabel("Path Value")
    plt.legend() 
    plt.grid(True, linestyle='--', alpha=0.5)  # Add a grid for better readability

    # Display the plot
    plt.tight_layout()  # Ensure the plot layout is neat
    plt.show()

    return






#Wrong Prior Check
def _iterative_consistency_check_overlapping_block_updates():
    observation_noise_level = 0.3
    T = 10.0
    time_axis = np.linspace(0,T,int(T)*10)
    beta=2
    alpha = 1/beta
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
    sample_rate = exp_rate_post(jump_times, alpha_prior, beta_prior, num_samp=1)
    sample_measure = js_dist_post(sub_jump_sizes,alpha,K)
    covariance_prior = np.zeros((3,3))
    covariance_prior[2,2] = 1.0


    #Generate jump sizes and times using some different parameters as inconsistent prior
    wrong_beta = beta
    wrong_C = C*10
    sub_jump_sizes,jump_times = gamma_process_jumps((wrong_beta,wrong_C),T,sim_rate=50)
    grouped_jump_sizes,grouped_jump_times = group_jumps(sub_jump_sizes,jump_times,time_axis)
    #Creating a wrong reference path from the wrong prior first
    log_likelihood, individual_log_likelihoods,x_means,x_covariances = compute_log_likelihood_langevin(np.zeros((3,1)),covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,alphaw,betaw)
    prior_path = integrate_to_path(sub_jump_sizes,jump_times,time_axis)
    #print(np.max(prior_path))

    # Plot the original path and the wrong prior path
    plt.figure(figsize=(12, 6))  # Set a larger figure size for better visibility
    



    # Set the number of new paths to plot
    iter_num = 10000
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
        alpha_value = 0.3    # 0.3 + 0.7 * (i / iter_num)  # Gradually increase alpha from 0.3 to 1.0
        plt.plot(time_axis, new_path[0, :], color='orange', alpha=alpha_value)
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





def _check_kalman():
    observation_noise_level = 0.3
    T = 10.0
    time_axis = np.linspace(0,T,int(T)*10)
    beta=1
    alpha = 1/beta
    C=1
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
    grouped_jump_sizes,grouped_jump_times = group_jumps(sub_jump_sizes,jump_times,time_axis)
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
    sample_rate = exp_rate_post(jump_times, alpha_prior, beta_prior, num_samp=1)
    sample_measure = js_dist_post(sub_jump_sizes,alpha,K)
    covariance_prior = np.zeros((3,3))
    covariance_prior[2,2] = 1.0
    log_likelihood, individual_log_likelihoods,x_means,x_covariances = compute_log_likelihood_langevin(np.zeros((3,1)),covariance_prior,grouped_jump_sizes,grouped_jump_times,time_axis,observations,A,h,g,R,alphaw,betaw)
    xs = x_means[:,0,0]
    xdots = x_means[:,1,0]
    muws = x_means[:,2,0]
    start_index = 40
    end_index = 52
    previous_log_likelihood = np.sum(individual_log_likelihoods[start_index+1:end_index+1]) 
    previous_X_mean = x_means[start_index+1,:,:]
    previous_X_uncertainty = x_covariances[start_index+1,:,:]

    wrong_beta = 0.1
    wrong_C = 0.001
    wrong_sub_jump_sizes,wrong_jump_times = gamma_process_jumps((wrong_beta,wrong_C),T,sim_rate=50)
    wrong_grouped_jump_sizes,wrong_grouped_jump_times = group_jumps(wrong_sub_jump_sizes,wrong_jump_times,time_axis)
    wrong_log_likelihood, wrong_individual_log_likelihoods,wrong_x_means,wrong_x_covariances = compute_log_likelihood_langevin(previous_X_mean,previous_X_uncertainty,wrong_grouped_jump_sizes[start_index:],wrong_grouped_jump_times[start_index:],time_axis[start_index:],observations[:,start_index+1:end_index+2],A,h,g,R,alphaw,betaw)

    #plt.plot(time_axis,xs,label = "x")
    #plt.plot(time_axis,xdots,label = "xdot")
    #plt.plot(time_axis,observations[0,:],label = "observations")
    #plt.legend()
    #plt.show()

    return

#The following is the actions to be run when the file is called directly
def main():
    #_test_js_posterior()
    #_explicit_truncation_experimentation()
    #_segment_update_test()
    #_check_overlapping_block_updates()
    #_check_kalman()
    #_iterative_check_overlapping_block_updates()
    _iterative_consistency_check_overlapping_block_updates()
if __name__ == "__main__":
    main()