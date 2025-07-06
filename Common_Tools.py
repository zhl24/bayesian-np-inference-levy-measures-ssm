import numpy as np
from scipy.linalg import expm #This is the automatic matrix expnent solver
from scipy.special import gammaln #log gamma function
from scipy.special import gamma #This one is claimed to be highly optimized
from scipy.stats import norm
from scipy.stats import t as student_t
from scipy.stats import pearsonr
import math
from scipy.special import logsumexp
from Common_Tools import*
from numba import jit
import matplotlib.pyplot as plt
from Levy_Generators import*
from Filters import Kalman_transit,Kalman_correct
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from scipy.stats import norm
from scipy.stats import invgamma

#This converts the (d,N) continuous observations into (d,T) discrete observations, leading to information loss
def integrate_to_path(jump_sizes,jump_times,evaluation_points):
    d,N = np.shape(jump_sizes) #Note that for index d-1 is the correct one
    
    T = np.size(evaluation_points) #A more adaptable version of len

    #Simple 2-pointer algorithm to speed up the computation
        #Sort the matrices first
    sort_indices = np.argsort(jump_times, axis=1) #Sort the jump time matrix in the chronological order i.e. along the time axis
    sorted_jump_times = np.take_along_axis(jump_times,sort_indices,axis = 1)
    sorted_jump_sizes = np.take_along_axis(jump_sizes,sort_indices,axis=1)
        #Variable Initialization
    p = 0 #The slow pointer on the continuous observations
    jump_size = sorted_jump_sizes[:,p]
    jump_time = sorted_jump_times[:,p]
    process_paths = np.zeros((d,T))
    previous_path_value = 0
    for i in range(T): #The fast pointer on the discrete time axis
        evaluation_point = evaluation_points[i]
        process_paths[:,i]+= previous_path_value #An incremental process
        while jump_time <= evaluation_point and p<=N-2:
            process_paths[:,i] += jump_size
            p+=1
            jump_size = sorted_jump_sizes[:,p]
            jump_time = sorted_jump_times[:,p]
        previous_path_value = process_paths[:,i]
    return process_paths.copy() #The output is a matrix of dimension (d,T)



#The time matrix is a 2-D matrix. Due to the use of broadcasting, even for a single time point it should be encapsulted into a matrix of (1,1).
def langevin_matrix_exponent(theta,time_matrix):
#The time matrix is of dimension (c,N)
    M1 = np.zeros((2,2))
    M1[0,1] = 1/theta
    M1[1,1] = 1

    M2 = np.zeros((2,2))
    M2[0,0] = 1
    M2[0,1] = -1/theta

    expt_series = np.exp(theta * time_matrix)  #A matrix of the exponential factors (c,N)

    #Broadcasting to prepare for the operations
    expt_series = expt_series[:,:,np.newaxis,np.newaxis] #For the scalar coefficients to be repeated into (c,N,1,1)
    M1 = M1[np.newaxis,np.newaxis,:,:]
    M2 = M2[np.newaxis,np.newaxis,:,:]
    #result = expt_series * M1 +M2

    return  expt_series * M1 +M2



def plot_measure(locations, probabilities,rate="Unknown"):
    # Plotting the Measure (Locations and Probabilities)
    plt.figure(figsize=(10, 6))
    plt.stem(locations, probabilities, basefmt=" ")
    plt.xlabel("Locations")
    plt.ylabel("Probabilities")
    plt.title(f"Dirichlet Process Sample Measure, rate={rate}")
    plt.grid(True)
    plt.show()

    # Plotting the Sorted Probabilities (Effective Number of Clusters)
    sorted_probabilities = np.sort(probabilities)[::-1]  # Sort in descending order
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(sorted_probabilities) + 1), sorted_probabilities)
    plt.xlabel("Cluster Rank")
    plt.ylabel("Probability")
    plt.title("Distribution of Ordered Probabilities")
    plt.grid(True, axis='y')
    plt.show()


#There is an if statement in the following function, so it will be most suited for using it in the outter loop i.e. indices should be fixed
#There is a potential solution to optimize this code. Check the description below.
def compute_overlapping_block_indices(block_size,overlapping_size,sequence_length):
    #3 cases to be taken care of:   1<n_start<n_end  n_start<1<n_end      n_start<n_end<1    
    n_start = np.floor((sequence_length-block_size)/(block_size-overlapping_size)).astype(int)
    n_end = np.floor((sequence_length-1)/(block_size-overlapping_size)).astype(int)
    #n+1 is the expected number of blocks
    if n_end < 1: #Treat n as 0 for all integer n less than 1, could be optimized
        block_indices = np.zeros((2,1),dtype=int)
        block_indices[0,0] = 0 #Start index of block
        block_indices[1,0] = sequence_length - 1 #End index of block
    elif n_start<1<=n_end:
        block_indices = np.zeros((2,2),dtype=int)
        block_indices[0,0] = 0
        block_indices[0,1] = sequence_length - block_size
        block_indices[1,0] = n_end
        block_indices[1,1] = sequence_length - 1
    else:

        block_indices = np.zeros((2,(n_end+1)),dtype=int)
        start_index_sequence = np.flip(np.arange(n_start+1,dtype = int))
        end_index_sequence = np.flip(np.arange(n_end+1,dtype= int))#A sequence from n to 0. Note that just like range, arange excludes the end index also
        block_indices[1,:] = sequence_length + end_index_sequence *(overlapping_size-block_size)-1  #End indices of blocks
        #The case of complete divisions
        if n_start == n_end:
            #print("Hi")
            block_indices[0,:] = sequence_length + start_index_sequence * (overlapping_size-block_size) - block_size #Start indices of blocks
            block_indices = block_indices[:,1:]
        #Incomplete division case
        else:
            #print("Yo")
            block_indices[0,-(n_start+1):] = sequence_length + start_index_sequence * (overlapping_size-block_size) - block_size #Start indices of blocks
            block_indices[0,:n_end-n_start] = np.zeros(n_end-n_start,dtype=int)
            if block_indices[0,0] == block_indices[1,0]:
                block_indices = block_indices[:,1:]
    return block_indices


def group_jumps(jump_sizes,jump_times,time_axis):
    c = np.shape(jump_sizes)[1] 
    N = np.size(time_axis) #A more adaptable version of len
    grouped_jump_sizes = [None]*(N-1)
    grouped_jump_times = [None]*(N-1)
    #Use again a double pointer algorithm
    sort_indices = np.argsort(jump_times, axis=1) #Sort the jump time matrix in the chronological order i.e. along the time axis
    sorted_jump_times = np.take_along_axis(jump_times,sort_indices,axis = 1)
    sorted_jump_sizes = np.take_along_axis(jump_sizes,sort_indices,axis=1)
    sp = 0 
    for i in range(N-1): #The fast pointer iterates over each group
        time = time_axis[i+1] #+1 for the fact that interval is being used here
        jt_collection = []
        js_collection = []
        while sorted_jump_times[0,sp] < time and sp<c-1: 
            jt_collection.append(sorted_jump_times[0,sp])
            js_collection.append(sorted_jump_sizes[0,sp])
            sp +=1
        grouped_jump_sizes[i] = np.array(js_collection)
        grouped_jump_times[i] = np.array(jt_collection)
    return grouped_jump_sizes,grouped_jump_times




#The following code is 
#Note that the following design considers the possibility of irregular time intervals for the discrete observations
#Likelihood Computation by running the kalman filter, but note that information form the previous time point is also needed
#We need the distribution of the hidden states for the previous point
#This is probably also the most computationally intensive step in the whole framework
def compute_log_likelihood_langevin(previous_X_mean,previous_X_uncertainty,grouped_jump_sizes,grouped_jump_times,time_points,observations,A,h,g,R,alphaw,betaw):
    M = 1 #Dimension of the observations
    #The time points should be the ends of the intervals, not the beginning, such than 1 to 1 correspondence of jumps to time points can be made
    T = len(time_points)# Return the lenght of the outtermost list for the total number of time points
    time_differences = np.zeros((1,T-1))
    time_differences[0,:]= np.diff(time_points) #Still T dimensional array since increments to all observations are needed

    theta = A[1,1]
    interval_matrix_exps = langevin_matrix_exponent(theta,time_differences)#(1,T-1,2,2)
    #Initialize the transition matrix and transition noise covariance for each time step
    transition_matrix = np.zeros((3,3))
    transition_matrix[2,2] = 1
    transition_noise_covariance = np.zeros((3,3))
    log_likelihood = 0.0
    individual_log_likelihoods = np.zeros(T-1) #Also an interval term
    previous_log_likelihood = 0.0
    accumulated_E = 0.0
    accumulated_F = 0.0
    x_means = np.zeros((T,3,1))
    x_covariances = np.zeros((T,3,3))
    x_means[0,:,:] = previous_X_mean
    x_covariances[0,:,:] = previous_X_uncertainty
    #note that the ccomputation cannot be fully vectorised due to the mismatch in the dimensions of the elelmetns, and hence a for loop has to be used.
    for t in range(T-1): #Iterate over each time point
        #Computing the transition matrix first
        interval_matrix_exp = interval_matrix_exps[0,t,:,:] #a (2,2) matrix exponent
        observation = observations[0,t+1] #The observation at time point t, a float value
        time_point = time_points[t+1] #The time point of each observation corresponding to each inteval
        jump_sizes = grouped_jump_sizes[t]  #An array of jump times, (c_t,)
        jump_times = grouped_jump_times[t]  #An array of jump sizes, (c_t,)

        differential_jump_times = np.zeros((1,len(jump_sizes)))
        differential_jump_times[0,:] = time_point - jump_times #(1,c_t) array of floats again
        differential_jump_time_matrix_exp = langevin_matrix_exponent(theta,differential_jump_times) #(1,c_t,2,2)
            #Broadcast for common multiplication
        jump_sizes = jump_sizes[np.newaxis,:,np.newaxis,np.newaxis]    
        jump_transition_exp_matrix = differential_jump_time_matrix_exp@h #(1,c_t,2,1)
            #This transpose action with specified axes causes a re-arrangement of the matrix elements into a shape of (1,c,1,2), and the remaining is just simple broadcasting product
        outer_products = jump_transition_exp_matrix @ jump_transition_exp_matrix.swapaxes(-1, -2)#(1,c,2,2), the second one is manual transpose by swapping the axes
        jump_transition_exp_matrix = jump_sizes *jump_transition_exp_matrix #(1,c_t,2,1)
        jump_transition_exp_matrix = np.sum(jump_transition_exp_matrix, axis=1).reshape(2, 1) #(1,c_t,2,1) summed in the time c_t axis to form (1,2,1) matrix.
            #Modify only the 6 elements
        transition_matrix[:2, :2] = interval_matrix_exp
        
        # Place the (2,1) column vector
        transition_matrix[:2, 2] = jump_transition_exp_matrix[:, 0]

        #Compute the marginalised transition noise covariance matrix then
        
        marginalised_noise_covariance = np.sum(jump_sizes*outer_products,axis = 1).reshape((2,2)) #(2,2) matrix
        transition_noise_covariance[:2,:2] = marginalised_noise_covariance #(3,3)

        #Run the Kalman filter since we have computed the transition matrix and the transition noise covariance matrix
        previous_X_mean, previous_X_uncertainty = Kalman_transit(previous_X_mean, previous_X_uncertainty, transition_matrix, transition_noise_covariance) #This could be the main position of problem, since the noise mean passed is a row vector here
        previous_X_mean, previous_X_uncertainty , log_det_F,Ei = Kalman_correct(previous_X_mean, previous_X_uncertainty, observation, g, R)#Need direct knowledge of the emission structure
        x_means[t+1,:,:] = previous_X_mean
        x_covariances[t+1,:,:] = previous_X_uncertainty

        #Accumulate the log marginal
        accumulated_F = accumulated_F - 0.5 * log_det_F   #Note that this term is already negative
        accumulated_E = accumulated_E  + Ei

        log_likelihood=-M*t/2*np.log(2*np.pi) + accumulated_F + alphaw * np.log(betaw) - (alphaw+t/2)*np.log(betaw + accumulated_E/2) + gammaln(t/2 + alphaw) - gammaln(alphaw)
        individual_log_likelihoods[t] = log_likelihood - previous_log_likelihood
        previous_log_likelihood = log_likelihood
    return log_likelihood, individual_log_likelihoods, x_means, x_covariances





#This one is the algorithm above but with all inference results returned
def compute_log_likelihood_langevin_all(previous_X_mean,previous_X_uncertainty,grouped_jump_sizes,grouped_jump_times,time_points,observations,A,h,g,R,alphaw,betaw):
    M = 1 #Dimension of the observations
    #The time points should be the ends of the intervals, not the beginning, such than 1 to 1 correspondence of jumps to time points can be made
    T = len(time_points)# Return the lenght of the outtermost list for the total number of time points
    time_differences = np.zeros((1,T-1))
    time_differences[0,:]= np.diff(time_points) #Still T dimensional array since increments to all observations are needed

    theta = A[1,1]
    interval_matrix_exps = langevin_matrix_exponent(theta,time_differences)#(1,T-1,2,2)
    #Initialize the transition matrix and transition noise covariance for each time step
    transition_matrix = np.zeros((3,3))
    transition_matrix[2,2] = 1
    transition_noise_covariance = np.zeros((3,3))
    log_likelihood = 0.0
    individual_log_likelihoods = np.zeros(T-1) #Also an interval term
    previous_log_likelihood = 0.0
    accumulated_E = 0.0
    accumulated_F = 0.0
    x_means = np.zeros((T,3,1))
    x_covariances = np.zeros((T,3,3))
    x_means[0,:,:] = previous_X_mean
    x_covariances[0,:,:] = previous_X_uncertainty
    alphaw_post = alphaw
    betaw_post = betaw
    #note that the ccomputation cannot be fully vectorised due to the mismatch in the dimensions of the elelmetns, and hence a for loop has to be used.
    for t in range(T-1): #Iterate over each time point
        #Computing the transition matrix first
        interval_matrix_exp = interval_matrix_exps[0,t,:,:] #a (2,2) matrix exponent
        observation = observations[0,t+1] #The observation at time point t, a float value
        time_point = time_points[t+1] #The time point of each observation corresponding to each inteval
        jump_sizes = grouped_jump_sizes[t]  #An array of jump times, (c_t,)
        jump_times = grouped_jump_times[t]  #An array of jump sizes, (c_t,)
        differential_jump_times = np.zeros((1,len(jump_sizes)))
        differential_jump_times[0,:] = time_point - jump_times #(1,c_t) array of floats again
        differential_jump_time_matrix_exp = langevin_matrix_exponent(theta,differential_jump_times) #(1,c_t,2,2)
            #Broadcast for common multiplication
        
        jump_sizes = jump_sizes[np.newaxis,:,np.newaxis,np.newaxis]    
        jump_transition_exp_matrix = differential_jump_time_matrix_exp@h #(1,c_t,2,1)
            #This transpose action with specified axes causes a re-arrangement of the matrix elements into a shape of (1,c,1,2), and the remaining is just simple broadcasting product
        outer_products = jump_transition_exp_matrix @ jump_transition_exp_matrix.swapaxes(-1, -2)#(1,c,2,2), the second one is manual transpose by swapping the axes
        jump_transition_exp_matrix = jump_sizes *jump_transition_exp_matrix #(1,c_t,2,1)
        jump_transition_exp_matrix = np.sum(jump_transition_exp_matrix, axis=1).reshape(2, 1) #(1,c_t,2,1) summed in the time c_t axis to form (1,2,1) matrix.
            #Modify only the 6 elements
        transition_matrix[:2, :2] = interval_matrix_exp

        # Place the (2,1) column vector
        transition_matrix[:2, 2] = jump_transition_exp_matrix[:, 0]

        #Compute the marginalised transition noise covariance matrix then
        
        marginalised_noise_covariance = np.sum(jump_sizes*outer_products,axis = 1).reshape((2,2)) #(2,2) matrix
        transition_noise_covariance[:2,:2] = marginalised_noise_covariance #(3,3)

        #Run the Kalman filter since we have computed the transition matrix and the transition noise covariance matrix
        previous_X_mean, previous_X_uncertainty = Kalman_transit(previous_X_mean, previous_X_uncertainty, transition_matrix, transition_noise_covariance) #This could be the main position of problem, since the noise mean passed is a row vector here
        previous_X_mean, previous_X_uncertainty , log_det_F,Ei = Kalman_correct(previous_X_mean, previous_X_uncertainty, observation, g, R)#Need direct knowledge of the emission structure
        x_means[t+1,:,:] = previous_X_mean
        x_covariances[t+1,:,:] = previous_X_uncertainty

        #Accumulate the log marginal
        accumulated_F = accumulated_F - 0.5 * log_det_F   #Note that this term is already negative
        accumulated_E = accumulated_E  + Ei

        log_likelihood=-M*t/2*np.log(2*np.pi) + accumulated_F + alphaw * np.log(betaw) - (alphaw+t/2)*np.log(betaw + accumulated_E/2) + gammaln(t/2 + alphaw) - gammaln(alphaw)
        individual_log_likelihoods[t] = log_likelihood - previous_log_likelihood
        previous_log_likelihood = log_likelihood

        alphaw_post += 0.5
        betaw_post += Ei/2
    #Ei is a (1,1) matrix element, and hence a .item() is added for the beta value to extract the float element 
    return log_likelihood, individual_log_likelihoods, x_means, x_covariances,alphaw_post,betaw_post.item()


def log_gamma_joint(samples: np.ndarray, shape: float, scale: float) -> float:
    """
    Compute the log joint likelihood of `samples` under a Gamma(shape, scale) distribution.
    
    Parameters
    ----------
    samples : np.ndarray
        Array of observed values (must be > 0), any shape.
    shape : float
        The 'a' (shape) parameter of the Gamma distribution.
    scale : float
        The 'b' (scale) parameter of the Gamma distribution.
        
    Returns
    -------
    float
        The sum of log-pdf over all entries in `samples`.
        Returns -np.inf if any sample is non-positive.
    """
    x = np.ravel(samples)
    if np.any(x <= 0):
        return -np.inf
    a = shape
    b = scale
    # log pdf: (a-1)*log(x) - x/b - a*log(b) - logΓ(a)
    log_pdf = (a - 1) * np.log(x) - x / b - a * np.log(b) - gammaln(a)
    return np.sum(log_pdf)





def gaussian_log_likelihood(x, mu, std):
    """
    Compute the univariate Gaussian log‐likelihood.

    Parameters
    ----------
    x : float
    mu : float
        Mean of the Gaussian.
    std: float std
    """
    return -0.5 * np.log(2 * np.pi * std**2) - (x - mu)**2 / (2 * std**2)


    


# Thinning function to reduce the number of samples
def thin_samples(samples, target_count):
    if len(samples) <= target_count:
        return samples  # No need to thin if the number of samples is already less than the target
    step = max(1, len(samples) // target_count)
    return samples[::step][:target_count]









#The Gaussian kernel written in numpy for efficiency
#This can be used to compute multiple dimensions, see the NVM_ground_truth_measure function for example use
#Be careful with the vectorized operations, since they may use up the computer's memory. The code has to be half-for loop but not fully vectorized
def gaussian_pdf(x, mean, std):
    """
    Compute the Gaussian PDF using NumPy.

    Args:
        x (ndarray): Points at which to evaluate the PDF.
        mean (float or ndarray): Mean of the Gaussian distribution.
        std (float or ndarray): Standard deviation of the Gaussian distribution.

    Returns:
        ndarray: PDF values at the given points.
    """
    coef = 1.0 / (np.sqrt(2 * np.pi) * std)
    exponent = -0.5 * ((x - mean) / std)**2
    return coef * np.exp(exponent)



#This is designed in particular for the rates and positions structure of the overall combined sample measure (2,KxIter_Num)
#The inputs are all basic arrays.
#x_axis is the axis of evaluation for the pdf
#Mixture of Gaussian for the NVM ground truth with exact muw and sigmaw values
def NVM_ground_truth_measure(rates, positions, x_axis, muw, sigmaw):
    """
    Smooth a measure Q(x) = sum_i w_i z_i into Q'(x) = sum_i w_i N(x; mu_w*z_i, sigma_w^2*z_i) using vectorization.

    Args:
        rates/weights (array-like): Weights w_i of the measure.
        positions (array-like): Original positions z_i of the atoms.
        x (array-like): Points at which to evaluate the smoothed measure.
        mu_w (float): Scaling factor for the mean of the kernel.
        sigma_w (float): Scaling factor for the standard deviation of the kernel.

    Returns:
        ndarray: Smoothed measure evaluated at the points x.
    """
    #No need for data type conversion if designed properly
    #positions = np.array(positions)
    #weights = np.array(weights)
    #x = np.array(x)

    # Compute kernel parameters. Boradcast into matrices
    kernel_means = muw * positions[:, None]  # Shape: (N, 1)
    kernel_stds = sigmaw * np.sqrt(positions[:, None])  # Shape: (N, 1)

    # Evaluate kernels at all points in x. Each row represents a function
    kernels = gaussian_pdf(x_axis[None, :], mean=kernel_means, std=kernel_stds)  # Shape: (N, M)
    ineffective_component_indices = np.where(np.all(kernels == 0, axis=1))[0] #This gives an array of indices of the components that give no contribution to the resultant mixture density.
    all_indices = np.arange(kernels.shape[0])  # All row indices
    effective_component_indices = np.setdiff1d(all_indices, ineffective_component_indices)  # Rows not in zero_row_indices
    effective_kernels = kernels[effective_component_indices,:]
    #Then, only the effective components should be used for computing the explicit mixture density, and the remaining of them should be used as delta indices
    # Weighted sum of kernels
    smoothed_density = np.sum(rates[effective_component_indices, None] * effective_kernels, axis=0)

    return smoothed_density




#Ineffective components considered. This returns a discrete DP multinomial measure
def NVM_ground_truth_measure_with_ineffective_components(rates, positions, x_axis, muw, sigmaw):
    """
    Smooth a measure Q(x) = sum_i w_i z_i into Q'(x) = sum_i w_i N(x; mu_w*z_i, sigma_w^2*z_i) using vectorization.

    Args:
        rates/weights (array-like): Weights w_i of the measure.
        positions (array-like): Original positions z_i of the atoms.
        x (array-like): Points at which to evaluate the smoothed measure.
        mu_w (float): Scaling factor for the mean of the kernel.
        sigma_w (float): Scaling factor for the standard deviation of the kernel.

    Returns:
        ndarray: Smoothed measure evaluated at the points x.
    """
    #No need for data type conversion if designed properly
    #positions = np.array(positions)
    #weights = np.array(weights)
    #x = np.array(x)

    # Compute kernel parameters. Boradcast into matrices
    kernel_means = muw * positions[:, None]  # Shape: (N, 1)
    kernel_stds = sigmaw * np.sqrt(positions[:, None])  # Shape: (N, 1)

    # Evaluate kernels at all points in x. Each row represents a function
    kernels = gaussian_pdf(x_axis[None, :], mean=kernel_means, std=kernel_stds)  # Shape: (N, M)
    ineffective_component_indices = np.where(np.all(kernels == 0, axis=1))[0] #This gives an array of indices of the components that give no contribution to the resultant mixture density.
    all_indices = np.arange(kernels.shape[0])  # All row indices
    effective_component_indices = np.setdiff1d(all_indices, ineffective_component_indices)  # Rows not in zero_row_indices
    effective_kernels = kernels[effective_component_indices,:]
    #Then, only the effective components should be used for computing the explicit mixture density, and the remaining of them should be used as delta indices
    # Weighted sum of kernels
    smoothed_density = np.sum(rates[effective_component_indices, None] * effective_kernels, axis=0)
    discrete_measure = np.stack([rates[ineffective_component_indices], positions[ineffective_component_indices]], axis=0)
    return smoothed_density, discrete_measure


#Clustering trick for dimension reduction
def aggregate_measure(weights, locations,decimal_precision=10):
    """
    Inputs: A crude discrete measure from DP
        weights: (N,) array
        locations: (N,) array
    Outputs: A discrete measure with reduced dimensions
        aggregated_weights: (K,) K<=N
        unique_locs: (K,) K<=N
    """
    # Round locations to avoid floating-point matching issues
    rounded = np.round(locations, decimals=decimal_precision)  #精确到小数点后十位
    unique_locs, inverse_idx = np.unique(rounded, return_inverse=True)
    aggregated_weights = np.zeros_like(unique_locs, dtype=weights.dtype)
    np.add.at(aggregated_weights, inverse_idx, weights)
    return aggregated_weights, unique_locs










#The student-t kernel
def student_t_pdf(x_axis, loc, scale, df): #note that the scale parameter here is the square root of the parameter used in the notes
    """
    Compute the Student-t PDF.
    """
    coef = gamma((df + 1) / 2) / (gamma(df / 2) * np.sqrt(df * np.pi) * scale)
    exponent = -((df + 1) / 2)
    return coef * (1 + ((x_axis - loc)**2) / (df * scale**2))**exponent


#The NVM measure inference code using the mixture of student-t distributions
def NVM_measure_inference(rates, positions,x_axis, mu,kw, alphaw_post,betaw_post):
    """
    Evaluate the Student-t mixture density.

    Args:
        x_Axis (ndarray): Points at which to evaluate the mixture.
        rates (array-like): Weights w_i of the mixture components.
        positions (array-like): Atom positions z_i.
        mu (float): The muw mean inferred
        kw (float): The muw marginalzied variance inferred
        alphaw_post (float): The IG posterior parameter for the sigma_w^2 inference
        betaw_post (float): The IG posterior parameter for the sigma_w^2 inference
        
        

    Returns:
        ndarray: Mixture density evaluated at the points x.
    """

    # Compute parameters for each component
    locs = mu * positions
    scales = np.sqrt(betaw_post * (positions + positions**2 * kw) / alphaw_post)
    dfs = 2 * alphaw_post

    

    # Compute the Student-t PDFs for all components and evaluation points
    pdfs = student_t_pdf(x_axis[None, :], locs[:,None], scales[:,None], dfs)  # Shape: (N, M)
    
    ineffective_component_indices = np.where(np.all(pdfs == 0, axis=1))[0] #This gives an array of indices of the components that give no contribution to the resultant mixture density.
    all_indices = np.arange(pdfs.shape[0])  # All row indices
    effective_component_indices = np.setdiff1d(all_indices, ineffective_component_indices)  # Rows not in zero_row_indices
    effective_kernels = pdfs[effective_component_indices,:]
    #Then, only the effective components should be used for computing the explicit mixture density, and the remaining of them should be used as delta indices
    # Weighted sum of PDFs across components
    mixture_density = np.sum(rates[effective_component_indices, None] * effective_kernels, axis=0)  # Shape: (M,)

    return mixture_density








#Distribution Average Functions
#This function averages sequences of Gaussian distributions
#The inputs are expected to be a list of matrices, say a list 20000 elements of (100,3,1) for the means and a list of 20000 elements of (100,3,3) covariances,
#where 100 is the number of points on the time axis and the list length is the number of iterations made
def Gaussian_average(means,covariances):
    average_means = np.zeros_like(means[0])
    average_covariances = np.zeros_like(covariances[0])
    num_iter = len(means)
    for i in range(num_iter): #Iterate over each iteration sample
        average_means += means[i]
        average_covariances += covariances[i] + means[i]@means[i].transpose(0,2,1)

    average_means = average_means/num_iter
    average_covariances = average_covariances/num_iter
    average_covariances -= average_means@average_means.transpose(0,2,1)
    return  average_means,average_covariances


#This function averages over Inverse Gamma distributions
def IG_average(alphas,betas):
    alphas = np.array(alphas)
    betas = np.array(betas)
    means = betas/(alphas-1)
    means_squared = means**2
    mean = np.mean(means)

    variances = means_squared/(alphas-2)
    variance = np.mean(variances + means_squared) - mean**2
    return mean, variance




#This is the integration function to check the rates. not the quad integral thing for generating the ground truth
def integrate_function(x_axis, function_values):
    """
    Numerically integrates a function over the given x-axis using the trapezoidal rule.

    Args:
        x_axis (array-like): Array of x values (must be sorted in ascending order).
        function_values (array-like): Array of function values corresponding to x_axis.

    Returns:
        float: The integral of the function over the given x-axis.
    """
    if len(x_axis) != len(function_values):
        raise ValueError("x_axis and function_values must have the same length.")
    
    integral = np.trapz(function_values, x_axis)
    return integral





#The tools for the upper tail functions

#This function integrate the Dirichlet measure to compute the upper tail integral funciton values
def DP_upper_tail_projection(positions,weights,x_axis):
    sorted_indices = np.argsort(positions)
    sorted_positions = positions[sorted_indices]
    sorted_weights = weights[sorted_indices]
    # Compute cumulative sum of weights (from largest position to smallest)
    cumulative_weights = np.cumsum(sorted_weights[::-1])[::-1]  # First reverse the weights for it to fit in the cumsum function. Second reverse the cumulative sum to fit in the upper tail function convention. vectorized computation
    # Compute the upper tail for each x in x_axis
    upper_tail_values = np.zeros_like(x_axis, dtype=float) #initialization first
    for i, x in enumerate(x_axis):
        # Find the index of the first position greater than or equal to x
        idx = np.searchsorted(sorted_positions, x, side="left")
        if idx < len(cumulative_weights):
            upper_tail_values[i] = cumulative_weights[idx]
        else:
            upper_tail_values[i] = 0.0  # No weights beyond this x

    return upper_tail_values




#This function computes the tail functions for the posterior NVM density inferred
def NVM_measure_tail_functions(rates, positions, x_axis, mu, kw, alphaw_post, betaw_post):
    """
    Compute the CDF and Survival Function (SF) for a mixture of Student-t distributions.

    Args:
        x_axis (ndarray): Points at which to evaluate the CDF and SF.
        rates (array-like): Weights w_i of the mixture components.
        positions (array-like): Atom positions z_i.
        mu (float): The muw mean inferred.
        kw (float): The muw marginalized variance inferred.
        alphaw_post (float): The IG posterior parameter for sigma_w^2 inference.
        betaw_post (float): The IG posterior parameter for sigma_w^2 inference.

    Returns:
        tuple: (CDF values, SF values) evaluated at the points x_axis.
    """
    if 0 in x_axis:
        raise ValueError("The input x_axis contains 0, which is not allowed. Please modify your input.")
    #Separate the axis into the positive and negative halves first
    negative_half = x_axis[x_axis < 0]  
    positive_half = x_axis[x_axis > 0]
    # Compute parameters for each component
    locs = mu * positions  # Means
    scales = np.sqrt(betaw_post * (positions + positions**2 * kw) / alphaw_post)  # Scales
    dfs = 2 * alphaw_post  # Degrees of freedom

    # Compute the Student-t CDFs for all components and evaluation points
    cdfs = student_t.cdf(negative_half[None, :], df=dfs, loc=locs[:, None], scale=scales[:, None])  # Shape: (N/2, M). CDF for the negative axis
    sfs = student_t.sf(positive_half[None, :], df=dfs, loc=locs[:, None], scale=scales[:, None])  #sf for the positive axis
    # Weighted sum of CDFs (mixture CDF)
    smoothed_cdf = np.sum(rates[:, None] * cdfs, axis=0)

    # Compute the SF as 1 - CDF
    smoothed_sf = np.sum(rates[:, None] * sfs, axis=0)

    return smoothed_cdf, smoothed_sf







#This function computes the overall mse of the function and the accumulated mse
def compute_mse_and_accumulated(ground_truth, estimated, x_axis):
    """
    Compute MSE and accumulated MSE along a non-uniform axis.
    
    Parameters:
    - ground_truth: Array of true values evaluated on x_axis.
    - estimated: Array of estimated values evaluated on x_axis.
    - x_axis: Non-uniform axis (1D array).
    
    Returns:
    - mse: Mean Squared Error (float).
    - accumulated_mse: Array of accumulated MSE values, starting from the largest x_axis.
    """

    # Compute weights for non-uniform axis. This is only useful for weighted mse
    #weights = np.diff(x_axis)  # Interval widths between points

    # Compute squared errors
    squared_errors = (ground_truth[:-1] - estimated[:-1])**2

    # MSE
    #wmse = np.sum(weights * squared_errors) / np.sum(weights)
    mse = np.mean(squared_errors) #This computes the mse directly
    # Compute accumulated MSE in reverse
    accumulated_mse = np.zeros_like(squared_errors)
    count = 1
    accumulated_mse[-1] = squared_errors[-1]  # Initialize the last value
    for i in range(len(squared_errors) - 2, -1, -1):  # Reverse accumulation
        accumulated_mse[i] = count/(count+1)*accumulated_mse[i + 1] + 1/(count+1)*squared_errors[i]
        count+=1

    return mse, accumulated_mse





#Autocorrelation Time Analysis Tools

#Entropy computation as the summary statistics for the DP sample measures
def compute_entropy(probabilities, base=np.e):
    """
    Compute the Shannon entropy for a given set of discrete probabilities.
    
    Parameters:
        probabilities (array-like): A 1D array of probabilities (must sum to 1 or will be normalized).
        base (float): The logarithm base. Default is natural log (base e).
                      Use base=2 for bits or base=10 for shannons.           
    Returns:
        entropy (float): The computed Shannon entropy.
    """    
    # Remove zero probabilities to avoid log(0)
    non_zero_probs = probabilities[probabilities > 0]
    # Compute entropy
    entropy = -np.sum(non_zero_probs * np.log(non_zero_probs) / np.log(base))
    return entropy

#The 1D Samples ACF and IACT Computation
def compute_acf_and_iact(samples, max_lag=50):
    """
    Compute the autocorrelation function (ACF) and integrated autocorrelation time (IACT)
    for 1D MCMC samples.
    
    Parameters:
        samples (ndarray): 1D array of MCMC samples.
        max_lag (int): Maximum lag to compute the autocorrelation.
    
    Returns:
        acf_values (ndarray): Autocorrelation function at each lag.
        iact (float): Integrated autocorrelation time.
    """
    N = len(samples)
    mean = np.mean(samples)
    centered_samples = samples - mean

    # Compute autocovariance for each lag
    acf_values = []
    var = np.mean(centered_samples ** 2)  # Variance (lag-0 autocovariance)
    for lag in range(max_lag + 1):
        cov = np.mean(centered_samples[:N - lag] * centered_samples[lag:])
        acf_values.append(cov / var)  # Normalize by variance

    # Compute IACT
    iact = 1 + 2 * np.sum(acf_values[1:])  # Sum autocorrelations except lag-0

    return np.array(acf_values), iact



#The normalized autocovariance function for time series samples
def compute_functional_acf_and_act(time_series_samples, time_axis, max_lag=50):
    """
    Compute the normalized functional autocorrelation function (ACF) and 
    the integrated autocorrelation time (IACT) for time series samples with irregular axis spacing.
    
    Parameters:
        time_series_samples (ndarray): MCMC samples of time series, shape (N, T),
                                       where N = number of samples, T = time points.
        axis_spacing (ndarray): 1D array of axis spacing, length T (e.g., time intervals).
        max_lag (int): Maximum lag to compute the autocorrelation.
    
    Returns:
        acf_values (ndarray): Normalized autocorrelation function at each lag.
        act (float): Integrated autocorrelation time.
    """
    axis_spacing = np.diff(time_axis)
    time_series_samples = time_series_samples[:,:-1] #Remove the sample at the last time point for dimensionality fit
    num_samples, time_points = np.shape(time_series_samples)
    
    # Compute weights from spacing
    weights = axis_spacing

    # Center the time series samples by subtracting the mean series
    mean_series = np.mean(time_series_samples, axis=0)
    centered_samples = time_series_samples - mean_series

    # Compute normalization constant (lag-0 autocorrelation with weights)
    norm_factor = np.mean([np.sum(weights * ts * ts) for ts in centered_samples])

    # Compute ACF at each lag
    acf_values = []
    for lag in tqdm(range(max_lag + 1), desc="ACF Computation Progress"):
        correlations = []
        for i in range(num_samples - lag):
            # Weighted inner product
            corr = np.sum(weights * centered_samples[i] * centered_samples[i + lag])
            correlations.append(corr)
        acf_values.append(np.mean(correlations) / norm_factor)  # Normalize by lag-0

    # Compute integrated autocorrelation time (IACT)
    act = 1 + 2 * np.sum(acf_values[1:])  # Sum over all lags except lag-0

    return np.array(acf_values), act


#Plotting tool
def plot_acf_arrays(acf_arrays, iact_values, labels, max_lag):
    """
    Visualize ACF results for multiple variables.

    Parameters:
        acf_arrays (list of ndarray): List of ACF arrays for each variable.
        iact_values (list of float): List of IACT values for each variable.
        labels (list of str): Labels corresponding to each variable.
        max_lag (int): Maximum lag used in ACF computation.
    """
    num_variables = len(acf_arrays)
    lags = range(max_lag + 1)

    # Set up subplots
    fig, axes = plt.subplots(num_variables, 1, figsize=(8, 2.5 * num_variables), sharex=True)

    # Ensure axes is iterable even for a single variable
    if num_variables == 1:
        axes = [axes]

    # Plot each ACF with its IACT
    for i in range(num_variables):
        axes[i].stem(lags, acf_arrays[i], linefmt='b-', markerfmt='bo', basefmt=" ")
        axes[i].set_title(f"{labels[i]} (IACT = {iact_values[i]:.2f})")
        axes[i].set_ylabel("ACF")
        axes[i].grid()

    axes[-1].set_xlabel("Lag")
    plt.tight_layout()
    plt.show()





#Plotting tool for mixture of distributions:


def plot_mixture_of_gaussians(means,variances,weights,x_axis,true_value,title,show_individuals=False,show_ground_truth = True):
    # Compute the individual Gaussian PDFs and the mixture
    pdf_components = []
    for mu, var in zip(means, variances):
        pdf = norm.pdf(x_axis, loc=mu, scale=np.sqrt(var))  # Gaussian PDF
        pdf_components.append(pdf)

    # Compute the mixture PDF
    mixture_pdf = np.sum(weights[:, None] * np.array(pdf_components), axis=0)

    # Plotting
    plt.figure(figsize=(10, 6))

    if show_individuals:
        # Plot individual Gaussians
        for i, pdf in enumerate(pdf_components):
            plt.plot(x_axis, pdf)

    # Plot the mixture
    plt.plot(x_axis, mixture_pdf, label=title, color='blue', linewidth=2)
    # Highlight the true parameter value
    if show_ground_truth:
        plt.axvline(true_value, color='red', linestyle='--', label=f'True Value: {true_value}', linewidth=2)

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    #plt.grid()
    plt.show()
    return


def plot_mixture_of_inverse_gamma(alphas, betas, weights, x_axis, true_value, title, show_individuals=False,show_ground_truth = True):
    """
    Plot a mixture of Inverse Gamma distributions with the individual components and a true parameter value.

    Parameters:
    - alphas: array-like, shape parameters of the Inverse Gamma components
    - betas: array-like, scale parameters of the Inverse Gamma components
    - weights: array-like, weights of the Inverse Gamma components
    - x_axis: array-like, x-axis values for the plot
    - true_value: float, the true parameter value to highlight
    - title: str, the title of the plot
    - show_individuals: bool, whether to show the individual components of the mixture
    """
    # Compute the individual Inverse Gamma PDFs and the mixture
    pdf_components = []
    for alpha, beta in zip(alphas, betas):
        pdf = invgamma.pdf(x_axis, a=alpha, scale=beta)  # Inverse Gamma PDF
        pdf_components.append(pdf)

    # Compute the mixture PDF
    pdf_components = np.array(pdf_components)
    mixture_pdf = np.sum(weights[:, None] * pdf_components, axis=0)

    # Plotting
    plt.figure(figsize=(10, 6))

    if show_individuals:
        # Plot individual Inverse Gamma components
        for i, pdf in enumerate(pdf_components):
            plt.plot(x_axis, pdf)

    # Plot the mixture
    plt.plot(x_axis, mixture_pdf, label=title, color='blue', linewidth=2)

    # Highlight the true parameter value
    if show_ground_truth:
        plt.axvline(true_value, color='red', linestyle='--', label=f'True Value: {true_value}', linewidth=2)

    # Add labels and legend
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    return


