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
from Filters import*
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from scipy.stats import norm
from scipy.stats import invgamma

#This converts the (d,N) continuous observations into (d,T) discrete observations, leading to information loss
def integrate_to_path(jump_sizes,jump_times,evaluation_points):
    d,N = np.shape(jump_sizes) #Note that for index d-1 is the correct one
    
    T = np.size(evaluation_points) #A more adaptable version of len

    # Handle case when there are no jumps
    if N == 0:
        return np.zeros((d,T))

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
# def compute_overlapping_block_indices(block_size,overlapping_size,sequence_length):
#     #3 cases to be taken care of:   1<n_start<n_end  n_start<1<n_end      n_start<n_end<1    
#     n_start = np.floor((sequence_length-block_size)/(block_size-overlapping_size)).astype(int)
#     n_end = np.floor((sequence_length-1)/(block_size-overlapping_size)).astype(int)
#     #n+1 is the expected number of blocks
#     if n_end < 1: #Treat n as 0 for all integer n less than 1, could be optimized
#         block_indices = np.zeros((2,1),dtype=int)
#         block_indices[0,0] = 0 #Start index of block
#         block_indices[1,0] = sequence_length - 1 #End index of block
#     elif n_start<1<=n_end:
#         block_indices = np.zeros((2,2),dtype=int)
#         block_indices[0,0] = 0
#         block_indices[0,1] = sequence_length - block_size
#         block_indices[1,0] = n_end
#         block_indices[1,1] = sequence_length - 1
#     else:

#         block_indices = np.zeros((2,(n_end+1)),dtype=int)
#         start_index_sequence = np.flip(np.arange(n_start+1,dtype = int))
#         end_index_sequence = np.flip(np.arange(n_end+1,dtype= int))#A sequence from n to 0. Note that just like range, arange excludes the end index also
#         block_indices[1,:] = sequence_length + end_index_sequence *(overlapping_size-block_size)-1  #End indices of blocks
#         #The case of complete divisions
#         if n_start == n_end:
#             #print("Hi")
#             block_indices[0,:] = sequence_length + start_index_sequence * (overlapping_size-block_size) - block_size #Start indices of blocks
#             block_indices = block_indices[:,1:]
#         #Incomplete division case
#         else:
#             #print("Yo")
#             block_indices[0,-(n_start+1):] = sequence_length + start_index_sequence * (overlapping_size-block_size) - block_size #Start indices of blocks
#             block_indices[0,:n_end-n_start] = np.zeros(n_end-n_start,dtype=int)
#             if block_indices[0,0] == block_indices[1,0]:
#                 block_indices = block_indices[:,1:]
#     return block_indices

def compute_overlapping_block_indices(block_size, overlapping_size, sequence_length):
    """
    Return a 2×K array:
      row 0: start indices of each block
      row 1: end   indices of each block
    with block length = block_size and stride = block_size - overlapping_size.
    Guaranteed:
      - first block starts at 0
      - last  block ends   at sequence_length - 1 (by appending a tail block if needed)
    """
    if sequence_length <= 0:
        return np.zeros((2,0), dtype=int)

    step = block_size - overlapping_size
    if step <= 0:
        raise ValueError("Require block_size > overlapping_size (strictly).")

    # 如果块比序列还长，就只给一块覆盖全部（可按需求改成 raise）
    if block_size >= sequence_length:
        return np.array([[0], [sequence_length - 1]], dtype=int)

    # 常规：从 0 开始每次跨 step
    starts = np.arange(0, sequence_length - block_size + 1, step, dtype=int)
    ends   = starts + block_size - 1

    # 若最后一块没对齐到末尾，就补一块以末尾对齐
    last_start = sequence_length - block_size
    if starts[-1] != last_start:
        starts = np.append(starts, last_start)
        ends   = np.append(ends, sequence_length - 1)

    return np.vstack([starts, ends])


def group_jumps(jump_sizes,jump_times,time_axis):
    c = np.shape(jump_sizes)[1] 
    N = np.size(time_axis) #A more adaptable version of len
    grouped_jump_sizes = [None]*(N-1)
    grouped_jump_times = [None]*(N-1)
    
    # Handle case when there are no jumps
    if c == 0:
        for i in range(N-1):
            grouped_jump_sizes[i] = np.array([])
            grouped_jump_times[i] = np.array([])
        return grouped_jump_sizes, grouped_jump_times
    
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





def compute_log_likelihood_langevin_conditional_on_sigmaw2(previous_X_mean,previous_X_uncertainty,grouped_jump_sizes,grouped_jump_times,time_points,observations,A,h,g,R,sigmaw2,compute_smoothing = False):
    """
    sigmaw2 should be some fixed float value for the NVM volatility parameter. Note that this is variance not the standard deviation.
    Note also that we need R to be [Cv] but not [kv] now for the conditional case.
    """
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

    if compute_smoothing:
        predicted_means = np.zeros((T, 3, 1))
        predicted_covariances = np.zeros((T, 3, 3))
        transition_matrices = np.zeros((T - 1, 3, 3))


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

        #We no longer use the normalized noise covariance here
        
        noise_covariance = sigmaw2 * np.sum(jump_sizes*outer_products,axis = 1).reshape((2,2)) #(2,2) matrix
        transition_noise_covariance[:2,:2] = noise_covariance #(3,3)

        #Run the Kalman filter since we have computed the transition matrix and the transition noise covariance matrix
        previous_X_mean, previous_X_uncertainty = Kalman_transit(previous_X_mean, previous_X_uncertainty, transition_matrix, transition_noise_covariance) #This could be the main position of problem, since the noise mean passed is a row vector here
        if compute_smoothing:
            transition_matrices[t, :, :] = transition_matrix
            predicted_means[t + 1, :, :] = previous_X_mean
            predicted_covariances[t + 1, :, :] = previous_X_uncertainty
        previous_X_mean, previous_X_uncertainty , log_det_F,Ei = Kalman_correct(previous_X_mean, previous_X_uncertainty, observation, g, R)#Need direct knowledge of the emission structure
        x_means[t+1,:,:] = previous_X_mean
        x_covariances[t+1,:,:] = previous_X_uncertainty

        #Accumulate the log marginal
        accumulated_F = accumulated_F - 0.5 * log_det_F   #Note that this term is already negative
        accumulated_E = accumulated_E  + Ei

        log_likelihood=-M*t/2*np.log(2*np.pi) + accumulated_F - 0.5 * accumulated_E
        individual_log_likelihoods[t] = log_likelihood - previous_log_likelihood
        previous_log_likelihood = log_likelihood
        
    if compute_smoothing:
        smoothed_means, smoothed_covariances = rts_smoother(
            filtered_means=x_means,
            filtered_covariances=x_covariances,
            predicted_means=predicted_means,
            predicted_covariances=predicted_covariances,
            transition_matrices=transition_matrices,
        )
        return log_likelihood, individual_log_likelihoods, x_means, x_covariances, smoothed_means, smoothed_covariances
    else:
        return log_likelihood, individual_log_likelihoods, x_means, x_covariances




#This one is the algorithm above but with all inference results returned, just with sigmaw2 posterior distribution returned.
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
    return log_likelihood, individual_log_likelihoods, x_means, x_covariances,alphaw_post,betaw_post




#Likelihood computation for the case of just NVM Process without system strutcures
def compute_log_likelihood_nvm_conditional_on_sigmaw2(previous_X_mean,previous_X_uncertainty,grouped_jump_sizes,grouped_jump_times,time_points,observations,g,R,sigmaw2):
    """
    sigmaw2 should be some fixed float value for the NVM volatility parameter. Note that this is variance not the standard deviation.
    Note also that we need R to be [Cv] but not [kv] now for the conditional case.
    """
    M = 1 #Dimension of the observations
    #The time points should be the ends of the intervals, not the beginning, such than 1 to 1 correspondence of jumps to time points can be made
    T = len(time_points)# Return the lenght of the outtermost list for the total number of time points
    time_differences = np.zeros((1,T-1))
    time_differences[0,:]= np.diff(time_points) #Still T dimensional array since increments to all observations are needed

    
    #Initialize the transition matrix and transition noise covariance for each time step
    transition_matrix = np.zeros((2,2))
    transition_matrix[1,1] = 1
    transition_matrix[0,0] = 1
    transition_noise_covariance = np.zeros((2,2))
    log_likelihood = 0.0
    individual_log_likelihoods = np.zeros(T-1) #Also an interval term
    previous_log_likelihood = 0.0
    accumulated_E = 0.0
    accumulated_F = 0.0
    x_means = np.zeros((T,2,1))
    x_covariances = np.zeros((T,2,2))
    x_means[0,:,:] = previous_X_mean
    x_covariances[0,:,:] = previous_X_uncertainty
    #note that the ccomputation cannot be fully vectorised due to the mismatch in the dimensions of the elelmetns, and hence a for loop has to be used.
    for t in range(T-1): #Iterate over each time point
        #Computing the transition matrix first
        observation = observations[0,t+1] #The observation at time point t, a float value
        time_point = time_points[t+1] #The time point of each observation corresponding to each inteval
        jump_sizes = grouped_jump_sizes[t]  #An array of jump times, (c_t,)
        jump_times = grouped_jump_times[t]  #An array of jump sizes, (c_t,)
        jump_sum = np.sum(jump_sizes)
        
        # Place the jump sum
        transition_matrix[0, 1] = jump_sum

        #We no longer use the normalized noise covariance here
        
        noise_covariance = sigmaw2 * jump_sum #Noise variance to be more precise
        transition_noise_covariance[0,0] = noise_covariance 

        #Run the Kalman filter since we have computed the transition matrix and the transition noise covariance matrix
        previous_X_mean, previous_X_uncertainty = Kalman_transit(previous_X_mean, previous_X_uncertainty, transition_matrix, transition_noise_covariance) #This could be the main position of problem, since the noise mean passed is a row vector here
        previous_X_mean, previous_X_uncertainty , log_det_F,Ei = Kalman_correct(previous_X_mean, previous_X_uncertainty, observation, g, R)#Need direct knowledge of the emission structure
        x_means[t+1,:,:] = previous_X_mean
        x_covariances[t+1,:,:] = previous_X_uncertainty

        #Accumulate the log marginal
        accumulated_F = accumulated_F - 0.5 * log_det_F   #Note that this term is already negative
        accumulated_E = accumulated_E  + Ei

        log_likelihood=-M*t/2*np.log(2*np.pi) + accumulated_F - 0.5 * accumulated_E
        individual_log_likelihoods[t] = log_likelihood - previous_log_likelihood
        previous_log_likelihood = log_likelihood
    return log_likelihood, individual_log_likelihoods, x_means, x_covariances





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





#Histogram tool, modified from the previous frequentist code base for the new data type
#Here is basically reconstrucitng the counts from the Levy measure and then do the histogram again
def regularized_projection_estimate(data, counts, resolution, regularized_power=0):
    #Both data and counts are simple 1D arrays but with the same dimension. Counts contain the numbers of counts for each data type
    
    #Double Pointer Algorithm for Histogramming
    x_max = np.max(data)
    x_min = np.min(data)
    hist_axis = np.linspace(x_min, x_max, resolution + 1)
    bin_width = hist_axis[1] - hist_axis[0]  #Assuming uniform histogram. If not, change bin_width for each iteration
    #print(bin_width)
    data_sorted_indices = np.argsort(data)
    data = data[data_sorted_indices]
    counts = counts[data_sorted_indices]
    data_pointer = 0
    axis_pointer = 0
    hist_counts = np.zeros(len(hist_axis) - 1, dtype=float)
    while data_pointer < len(data) and axis_pointer < len(hist_axis) - 1:
        if data[data_pointer] <= hist_axis[axis_pointer + 1]:
            hist_counts[axis_pointer] += counts[data_pointer]* (data[data_pointer] ** regularized_power) /  bin_width
            data_pointer += 1
        else:
            axis_pointer += 1
    
    bin_centers = (hist_axis[:-1] + hist_axis[1:]) / 2
    results = np.zeros(resolution * len(bin_centers))
    x_axis = np.linspace(x_min, x_max, len(results))
    
    # 填充 results 数组
    for i in range(len(bin_centers)):
        for j in range(resolution):
            if i * resolution + j < len(results):
                results[i * resolution + j] = hist_counts[i]

    restoration_array = x_axis ** (-regularized_power)
    results = results * restoration_array
    return x_axis, results



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






#The log domain computation version


def stable_gaussian_pdf(x, mean, std):
    """
    Compute the numerically stable Gaussian PDF using NumPy.

    Args:
        x (ndarray): Points at which to evaluate the PDF.
        mean (float or ndarray): Mean of the Gaussian distribution.
        std (float or ndarray): Standard deviation of the Gaussian distribution.

    Returns:
        ndarray: PDF values at the given points.
    """
    # Prevent division by zero
    if np.any(std <= 0):
        raise ValueError("Standard deviation must be positive.")

    # Compute the scaled squared distance
    z = (x - mean) / std
    exp_term = -0.5 * z**2

    # Numerically stable coefficient
    log_coef = -0.5 * np.log(2 * np.pi) - np.log(std)

    # Combine log terms and exponentiate
    log_pdf = log_coef + exp_term
    return log_pdf


def stable_NVM_ground_truth_measure(rates, positions, x_axis, muw, sigmaw):
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
    log_kernels = stable_gaussian_pdf(x_axis[None, :], mean=kernel_means, std=kernel_stds)  # Shape: (N, M)

    # Weighted sum of kernels
    weighted_kernels = np.exp(np.log(rates[:, None]) + log_kernels)
    smoothed_density = np.sum(weighted_kernels, axis=0)

    return smoothed_density




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




#Ineffective components considered. This returns a discrete DP multinomial measure
def NVM_measure_inference_with_ineffective_components(rates, positions,x_axis, mu,kw, alphaw_post,betaw_post):
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
    discrete_measure = np.stack([rates[ineffective_component_indices], positions[ineffective_component_indices]], axis=0)
    return mixture_density, discrete_measure






#Log domain computation functions
def stable_student_t_pdf(x_axis, loc, scale, df):
    """
    Compute the numerically stable Student-t PDF.
    
    Args:
        x_axis (array-like): Points at which to evaluate the PDF.
        loc (float or array-like): Location parameter (mean of the distribution).
        scale (float or array-like): Scale parameter (std of the distribution).
        df (float): Degrees of freedom.

    Returns:
        ndarray: Evaluated Student-t PDF values.
    """
    if np.any(scale <= 0):
        raise ValueError("Scale parameter must be positive.")

    # Compute logarithm of the coefficient
    log_coef = (
        gammaln((df + 1) / 2) 
        - gammaln(df / 2) 
        - 0.5 * np.log(df * np.pi) 
        - np.log(scale)
    )

    # Compute the log-exponent term
    log_exponent = -((df + 1) / 2) * np.log1p(((x_axis - loc)**2) / (df * scale**2))

    # Combine terms for the log-PDF and return the exponentiated result
    log_pdf = log_coef + log_exponent
    return log_pdf


def stable_NVM_measure_inference(rates, positions, x_axis, mu, kw, alphaw_post, betaw_post):
    """
    Evaluate the numerically stable Student-t mixture density.

    Args:
        x_axis (ndarray): Points at which to evaluate the mixture.
        rates (array-like): Weights w_i of the mixture components.
        positions (array-like): Atom positions z_i.
        mu (float): Mean parameter.
        kw (float): Variance parameter.
        alphaw_post (float): IG posterior alpha parameter for variance.
        betaw_post (float): IG posterior beta parameter for variance.

    Returns:
        ndarray: Mixture density evaluated at the points x.
    """
    if len(rates) != len(positions):
        raise ValueError("Rates and positions must have the same length.")

    # Compute parameters for each component
    locs = mu * positions
    scales = np.sqrt(betaw_post * (positions + positions**2 * kw) / alphaw_post)
    dfs = 2 * alphaw_post

    # Compute the Student-t PDFs for all components and evaluation points
    log_pdfs = stable_student_t_pdf(x_axis[None, :], locs[:, None], scales[:, None], dfs)  # Shape: (N, M)
    weighted_kernels = np.exp(np.log(rates[:, None])+log_pdfs)
    # Weighted sum of PDFs across components
    mixture_density = np.sum(weighted_kernels, axis=0)  # Shape: (M,)

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
#The version with variable sigmaw2
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

#The posterior mean esitmate
def NVM_measure_tail_functions_conditional_on_sigmaw2(
    rates, positions, x_axis,
    muw_mu, muw_sigma2, sigmaw2):
    """
    Tail functions for the NVM Lévy density when sigma_w^2 is fixed.

    Parameters
    ----------
    rates : (K,) array_like
        Stick-breaking weights W_i.
    positions : (K,) array_like
        Atom positions z_i.
    x_axis : (N,) ndarray
        Evaluation points (must not contain 0).
    muw_mu, muw_sigma2 : float
        Mean and variance of the Gaussian prior/posterior for mu_w.
    sigmaw2 : float
        Fixed variance parameter sigma_w^2.

    Returns
    -------
    smoothed_cdf : ndarray, shape (N_neg,)
        Mixture CDF on the negative half of x_axis.
    smoothed_sf  : ndarray, shape (N_pos,)
        Mixture survival function on the positive half of x_axis.
    """
    if np.any(x_axis == 0):
        raise ValueError("x_axis contains 0; remove or shift zero.")

    # split axis
    neg = x_axis[x_axis < 0]   # shape (N_neg,)
    pos = x_axis[x_axis > 0]   # shape (N_pos,)

    # component-wise Gaussian parameters
    means = muw_mu * positions                              # (K,)
    variances = muw_sigma2 * positions**2 + sigmaw2 * positions  # (K,)
    stds = np.sqrt(variances)                               # (K,)

    # broadcast: (K,1) vs (1,N_half) -> (K,N_half)
    cdf_components = norm.cdf(neg[None, :], loc=means[:, None], scale=stds[:, None])
    sf_components  = norm.sf (pos[None, :], loc=means[:, None], scale=stds[:, None])

    # weight by mixture weights
    smoothed_cdf = np.sum(rates[:, None] * cdf_components, axis=0)  # (N_neg,)
    smoothed_sf  = np.sum(rates[:, None] * sf_components,  axis=0)  # (N_pos,)

    return smoothed_cdf, smoothed_sf




#The inference scheme under the collapsed Gibbs sampler. This is for estimating the entire distribution.
def NVM_measure_tail_functions_sampled_on_sigmaw2(
    rates, positions, x_axis,
    muw_mu, muw_sigma2, sigmaw2,
    random_state=None
):
    """
    Tail functions for one posterior draw of the NVM Lévy density
    when sigma_w^2 is fixed.

    Procedure:
      1. Sample mu_w ~ N(muw_mu, muw_sigma2)
      2. Form the induced NVM Lévy measure draw
      3. Compute lower-tail CDF on x<0 and upper-tail SF on x>0

    Parameters
    ----------
    rates : (K,) array_like
        Stick-breaking / atom weights W_i.
    positions : (K,) array_like
        Atom positions z_i.
    x_axis : (N,) ndarray
        Evaluation points (must not contain 0).
    muw_mu, muw_sigma2 : float
        Mean and variance of the Gaussian posterior for mu_w.
    sigmaw2 : float
        Fixed variance parameter sigma_w^2.
    random_state : None, int, or np.random.Generator
        Random seed or generator.

    Returns
    -------
    smoothed_cdf : ndarray, shape (N_neg,)
        Mixture CDF on the negative half of x_axis.
    smoothed_sf : ndarray, shape (N_pos,)
        Mixture survival function on the positive half of x_axis.
    muw_sample : float
        The sampled mu_w draw used in this posterior sample.
    """
    if np.any(x_axis == 0):
        raise ValueError("x_axis contains 0; remove or shift zero.")

    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    rates = np.asarray(rates)
    positions = np.asarray(positions)
    x_axis = np.asarray(x_axis)

    # sample posterior draw for mu_w
    muw_sample = rng.normal(loc=muw_mu, scale=np.sqrt(muw_sigma2))

    # split axis
    neg = x_axis[x_axis < 0]
    pos = x_axis[x_axis > 0]

    # component-wise Gaussian parameters for this sampled draw
    means = muw_sample * positions                    # (K,)
    variances = sigmaw2 * positions                  # (K,)
    stds = np.sqrt(variances)                        # (K,)

    cdf_components = norm.cdf(
        neg[None, :], loc=means[:, None], scale=stds[:, None]
    )
    sf_components = norm.sf(
        pos[None, :], loc=means[:, None], scale=stds[:, None]
    )

    smoothed_cdf = np.sum(rates[:, None] * cdf_components, axis=0)
    smoothed_sf  = np.sum(rates[:, None] * sf_components, axis=0)

    return smoothed_cdf, smoothed_sf, muw_sample


#This function computes the upper tial function for the projection estimator in the Lebesgue domain:
def proj_lebesgue_upper_tail(jump_sizes,x_axis,T):
    """
    Input jump_sizes is an (1,N) matrix
    Input x_axis is an (T,) array containing the evaluation points
    Input T is a float of the time length
    """
    jump_sizes = jump_sizes[0,:]
    jump_sizes = np.sort(jump_sizes)
    jump_sizes = np.insert(jump_sizes, 0, 0) #Insert x_0
    jump_sizes = np.flip(jump_sizes) #Flip the order for the evaluation of the upper tail fucntion
    flipped_axis = np.flip(x_axis) #Flip also the x_axis

    n = len(jump_sizes)
    m = len(x_axis)
    result = np.zeros(m)
    inherited_part = 0.0
    # Double pointer trick for efficient computation
    jump_pointer = 0  # Pointer for sorted jump sizes, starts from the largest jump
    for i in range(m):  # Iterate over the x_axis from the descending order
        x = flipped_axis[i]
        # Find the appropriate interval for x using the double-pointer technique
        result[i] = inherited_part
        while jump_pointer < n and jump_sizes[jump_pointer] > x:
            result[i]+=1/T
            jump_pointer += 1
        #Do not increment the jump in this part to let the partial case be taken as the complete case in the next iteration
        inherited_part = result[i]
        if jump_pointer < n:
            lb_jump = jump_sizes[jump_pointer]
            if jump_pointer > 0:
                ub_jump = jump_sizes[jump_pointer-1]
                result[i] += (ub_jump-x)/(ub_jump - lb_jump)/T
            else:
                # Handle edge case when jump_pointer is 0
                result[i] += 0  # No contribution when at the beginning
    result = np.flip(result)
    return result




#This function computes the upper tail function for the projection estimator in the regularized domain:
def proj_regularized_upper_tail(jump_sizes,x_axis,T):
    """
    Input jump_sizes is an (1,N) matrix
    Input x_axis is an (T,) array containing the evaluation points
    Input T is a float of the time length
    """
    jump_sizes = jump_sizes[0,:]
    jump_sizes = np.sort(jump_sizes)
    jump_sizes = np.insert(jump_sizes, 0, 0) #Insert x_0
    jump_sizes = np.flip(jump_sizes) #Flip the order for the evaluation of the upper tail fucntion
    flipped_axis = np.flip(x_axis) #Flip also the x_axis

    n = len(jump_sizes)
    m = len(x_axis)
    result = np.zeros(m)
    inherited_part = 0.0
    # Double pointer trick for efficient computation
    jump_pointer = 0  # Pointer for sorted jump sizes, starts from the largest jump
    for i in range(m):  # Iterate over the x_axis from the descending order
        x = flipped_axis[i]
        # Find the appropriate interval for x using the double-pointer technique
        result[i] = inherited_part
        while jump_pointer < n and jump_sizes[jump_pointer] > x:
            current_jump = jump_sizes[jump_pointer]
            if jump_pointer + 1 < n:
                next_jump = jump_sizes[jump_pointer+1]
                if next_jump > 0:  # Avoid division by zero
                    result[i]+=1/T * current_jump/next_jump
                else:
                    result[i]+=1/T  # Fallback when next_jump is 0
            else:
                # At the end of array, just add the standard contribution
                result[i]+=1/T
            jump_pointer += 1
        #Do not increment the jump in this part to let the partial case be taken as the complete case in the next iteration
        inherited_part = result[i]
        if jump_pointer < n:
            lb_jump = jump_sizes[jump_pointer]
            if jump_pointer > 0:
                ub_jump = jump_sizes[jump_pointer-1]
                result[i] += ub_jump/x * (ub_jump-x)/(ub_jump - lb_jump)/T
            else:
                # Handle edge case when jump_pointer is 0
                result[i] += 0  # No contribution when at the beginning
    result = np.flip(result)
    return result




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

def compute_acf_and_iact_via_fft(samples, max_lag=50):
    """
    Compute the autocorrelation function (ACF) and integrated autocorrelation time (IACT)
    for 1D MCMC samples using FFT-based autocovariance.
    """
    samples = np.asarray(samples)
    N = len(samples)
    mean = np.mean(samples)
    centered = samples - mean

    # Next power of 2 for zero-padding (for FFT speed and to avoid circular convolution)
    nfft = 1 << (2 * N - 1).bit_length()
    
    # FFT of centered samples
    f = np.fft.fft(centered, n=nfft)
    # Autocovariance via inverse FFT of power spectrum
    acov = np.fft.ifft(f * np.conjugate(f)).real[:max_lag+1]
    acov /= np.arange(N, N - max_lag - 1, -1)  # unbiased normalization

    # Normalize to get autocorrelation
    acf_values = acov / acov[0]

    # Integrated autocorrelation time (IACT)
    iact = 1 + 2 * np.sum(acf_values[1:])

    return acf_values, iact


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


def compute_functional_acf_and_act_via_fft(time_series_samples, time_axis, max_lag=50):
    """
    FFT-accelerated functional ACF/IACT for irregular time axis (weights = diff(time)).
    Works along the sample axis N (treat each time point/column as a series across samples).
    
    Inputs:
      time_series_samples: array (N, T)
      time_axis: array (T,)
      max_lag: int, <= N-1
    
    Returns:
      acf_values: array (max_lag+1,)
      act: float
    """
    # weights from irregular spacing: length T-1
    w = np.diff(time_axis)
    X = time_series_samples[:, :-1]          # (N, T-1), align with weights
    N, Tm1 = X.shape

    if max_lag >= N:
        max_lag = N-1

    # center each time column by mean over samples
    Xc = X - X.mean(axis=0, keepdims=True)   # (N, T-1)

    # lag-0 normalization factor: mean over samples of weighted squared norm
    # = sum_t w_t * (1/N * sum_i Xc[i,t]^2)
    norm_factor = np.sum(w * (Xc**2).mean(axis=0))
    if norm_factor == 0:
        # all-constant input -> ACF undefined; return zeros with act=1
        return np.r_[1.0, np.zeros(max_lag)], 1.0

    # ---- FFT-based unbiased autocov across samples for each column ----
    # Zero-pad到 >= 2N 的长度实现线性相关；取前 N 个滞后
    L = 1 << (2*N-1).bit_length()   # next power of two >= 2N
    # FFT along sample axis (axis=0), for all columns并行
    F = np.fft.rfft(Xc, n=L, axis=0)
    # Power spectrum -> IFFT gives (circular) autocorrelation
    acf_circ = np.fft.irfft(F * np.conj(F), n=L, axis=0).real   # shape (L, T-1)
    # 线性相关的前 N 个滞后
    acf_lin = acf_circ[:N, :]                                   # (N, T-1)

    # 无偏（unbiased）归一：每个滞后除以有效样本数 (N - lag)
    counts = (N - np.arange(N)).reshape(-1, 1)                   # (N, 1)
    acf_unbiased = acf_lin / counts                              # (N, T-1)

    # 汇总为“functional”自相关：对列按 w 加权求和
    # acf_functional[lag] = sum_t w_t * acf_unbiased[lag, t]
    acf_weighted = acf_unbiased @ w                              # (N,)

    # 取所需滞后，并按 lag-0 归一
    acf_values = (acf_weighted[:max_lag+1] / norm_factor).copy()

    # Integrated autocorrelation time
    act = 1.0 + 2.0 * np.sum(acf_values[1:])

    return acf_values, float(act)

#Plotting tool
def plot_acf_arrays(acf_arrays, iact_values, labels, max_lag,show=True,file_path=None):
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
    if show:
        plt.show()
    if file_path is not None:
        plt.savefig(file_path)






#Plotting tool for mixture of distributions:


def plot_mixture_of_gaussians(means,variances,weights,x_axis,true_value,title,show_individuals=False,show_ground_truth = True,show=True,file_path=None,show_mixture_mean = False):
    # Compute the individual Gaussian PDFs and the mixture
    pdf_components = []
    for mu, var in zip(means, variances):
        pdf = norm.pdf(x_axis, loc=mu, scale=np.sqrt(var))  # Gaussian PDF
        pdf_components.append(pdf)

    # Compute the mixture PDF
    mixture_pdf = np.sum(weights[:, None] * np.array(pdf_components), axis=0)
    mixture_mean = float(np.sum(weights * means))

    # Plotting
    plt.figure(figsize=(8, 6))

    if show_individuals:
        # Plot individual Gaussians
        for i, pdf in enumerate(pdf_components):
            plt.plot(x_axis, pdf)

    # Plot the mixture
    plt.plot(x_axis, mixture_pdf, label=title, color='blue', linewidth=2)
    # Highlight the true parameter value
    if show_ground_truth:
        plt.axvline(true_value, color='red', linestyle='--', label=f'True Value', linewidth=2)
    if show_mixture_mean:
        plt.axvline(mixture_mean, color='green', linestyle='-.', label=f'Mixture Mean', linewidth=2)

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    #plt.grid()
    if show:
        plt.show()
    if file_path is not None:
        plt.savefig(file_path)
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




def plot_mixture_of_NIG_marginals(
    means,
    variances,
    alphas,
    betas,
    weights,
    x_axis_sigmaw2,
    x_axis_muw=None,
    true_muw=None,
    true_sigmaw2=None,
    num_samples_per_muw=10,
    random_state=None,
):
    """
    Plot mixture marginals induced by a mixture of Normal-Inverse-Gamma components.

    Parameters
    ----------
    means : array-like, shape (K,)
        Posterior means m_k for mu_w | sigma_w^2, component k.
    variances : array-like, shape (K,)
        Conditional variance multipliers v_k in:
            mu_w | sigma_w^2, k ~ N(m_k, v_k * sigma_w^2)
    alphas : array-like, shape (K,)
        Inverse-Gamma shape parameters alpha_k for sigma_w^2 | component k.
    betas : array-like, shape (K,)
        Inverse-Gamma scale parameters beta_k for sigma_w^2 | component k.
    weights : array-like, shape (K,)
        Mixture weights.
    x_axis_sigmaw2 : array-like
        Grid for plotting sigma_w^2 marginal density.
    x_axis_muw : array-like or None
        Grid for plotting mu_w marginal density. If None, an automatic grid is built.
    true_muw : float or None
        True mu_w value for reference.
    true_sigmaw2 : float or None
        True sigma_w^2 value for reference.
    num_samples_per_muw : int
        Number of sigma_w^2 samples per component for MC marginalization of mu_w.
    random_state : int or None
        Random seed.
    """

    rng = np.random.default_rng(random_state)

    means = np.asarray(means, dtype=float)          # (K,)
    variances = np.asarray(variances, dtype=float)  # (K,)
    alphas = np.asarray(alphas, dtype=float)        # (K,)
    betas = np.asarray(betas, dtype=float)          # (K,)
    weights = np.asarray(weights, dtype=float)      # (K,)
    x_axis_sigmaw2 = np.asarray(x_axis_sigmaw2, dtype=float)

    weights = weights / np.sum(weights)
    K = len(weights)

    if not (len(means) == len(variances) == len(alphas) == len(betas) == K):
        raise ValueError("means, variances, alphas, betas, and weights must all have the same length.")

    # ---------------------------------------------------------
    # 1) sigma_w^2 marginal: vectorized mixture of Inverse-Gamma
    # ---------------------------------------------------------
    # Broadcast to shape (K, G_sigma)
    pdf_components_sigmaw2 = invgamma.pdf(
        x_axis_sigmaw2[None, :],
        a=alphas[:, None],
        scale=betas[:, None],
    )

    mixture_pdf_sigmaw2 = np.sum(weights[:, None] * pdf_components_sigmaw2, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis_sigmaw2, mixture_pdf_sigmaw2, color="blue", label="Posterior", linewidth=2)

    if true_sigmaw2 is not None:
        plt.axvline(
            true_sigmaw2,
            color="red",
            linestyle="--",
            label=f"True Value: {true_sigmaw2}",
            linewidth=2,
        )

    plt.title(r"$\sigma_w^2$ Inference")
    plt.xlabel(r"$\sigma_w^2$")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    # ---------------------------------------------------------
    # 2) mu_w marginal: MC marginalization over sigma_w^2, fully vectorized
    # ---------------------------------------------------------
    if x_axis_muw is None:
        sigmaw2_means = np.where(
            alphas > 1,
            betas / (alphas - 1),
            betas / np.maximum(alphas, 1e-8)
        )
        approx_std_per_component = np.sqrt(np.maximum(variances * sigmaw2_means, 1e-12))
        left = np.min(means - 5.0 * approx_std_per_component)
        right = np.max(means + 5.0 * approx_std_per_component)
        x_axis_muw = np.linspace(left, right, 500)
    else:
        x_axis_muw = np.asarray(x_axis_muw, dtype=float)

    # Sample sigma_w^2 for every component at once
    # Shape: (K, S)
    sigmaw2_samples = invgamma.rvs(
        a=alphas[:, None],
        scale=betas[:, None],
        size=(K, num_samples_per_muw),
        random_state=rng,
    )

    # Conditional stds for mu_w | sigma_w^2
    # Shape: (K, S)
    conditional_stds = np.sqrt(np.maximum(variances[:, None] * sigmaw2_samples, 1e-12))

    # Vectorized Gaussian pdf evaluation
    # x_axis_muw: (G,)
    # means[:, None, None]: (K,1,1)
    # conditional_stds[:, :, None]: (K,S,1)
    # Result: (K, S, G)
    centered = x_axis_muw[None, None, :] - means[:, None, None]
    scales = conditional_stds[:, :, None]

    cond_pdfs = (
        np.exp(-0.5 * (centered / scales) ** 2)
        / (np.sqrt(2.0 * np.pi) * scales)
    )

    # Average over MC samples within each component -> (K, G)
    pdf_components_muw = np.mean(cond_pdfs, axis=1)

    # Weight over mixture components -> (G,)
    mixture_pdf_muw = np.sum(weights[:, None] * pdf_components_muw, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis_muw, mixture_pdf_muw, color="blue", label="Posterior", linewidth=2)

    if true_muw is not None:
        plt.axvline(
            true_muw,
            color="red",
            linestyle="--",
            label=f"True Value: {true_muw}",
            linewidth=2,
        )

    plt.title(r"$\mu_w$ Inference")
    plt.xlabel(r"$\mu_w$")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    return {
        "x_axis_sigmaw2": x_axis_sigmaw2,
        "mixture_pdf_sigmaw2": mixture_pdf_sigmaw2,
        "x_axis_muw": x_axis_muw,
        "mixture_pdf_muw": mixture_pdf_muw,
        "pdf_components_sigmaw2": pdf_components_sigmaw2,
        "pdf_components_muw": pdf_components_muw,
        "sigmaw2_samples": sigmaw2_samples,
    }


def plot_gamma_mixture(shapes, scales, weights, xmax=None, n_points=1200):
    """
    Plot the jump–size distribution for a mixture of Gamma components.

    Parameters
    ----------
    shapes   : array-like (K,)   – shape parameters α_k
    scales   : array-like (K,)   – scale parameters θ_k  (mean is α_k*θ_k)
    weights  : array-like (K,)   – mixture weights w_k   (will be normalised)
    xmax     : float or None     – right-hand x-limit; pick a heuristic if None
    n_points : int               – grid resolution for the PDF plot
    """
    shapes  = np.asarray(shapes,  dtype=float)
    scales  = np.asarray(scales,  dtype=float)
    weights = np.asarray(weights, dtype=float)

    if not (shapes.shape == scales.shape == weights.shape):
        raise ValueError("shapes, scales, and weights must have the same length")

    weights /= weights.sum()              # ensure they sum to 1

    # If xmax not supplied, choose something that covers most mass of the heaviest tail
    if xmax is None:
        xmax = 5.0 * np.max(shapes * scales)

    x = np.linspace(0.0, xmax, n_points)
    mixture_pdf = np.zeros_like(x)

    # Plot each weighted component (dashed) and build the mixture PDF
    for α, θ, w in zip(shapes, scales, weights):
        comp_pdf = gamma(a=α, scale=θ).pdf(x)
        mixture_pdf += w * comp_pdf
        plt.plot(x, w * comp_pdf, '--', label=fr"$w={w},\,\Gamma({α:.2g},\,{θ:.2g})$")

    # Plot the overall mixture (solid)
    plt.plot(x, mixture_pdf, lw=2, label="Mixture")
    plt.xlabel("jump size")
    plt.ylabel("density")
    plt.title("Gamma–mixture jump-size distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()


def upper_tail_from_deltas(grid, locs, masses):
    """
    Compute ν([x,∞)) on a grid given a discrete measure ∑ w_i δ_{loc_i}.
    grid : (M,) nondecreasing array
    locs : (K,)
    masses: (K,)   (already includes the rate factor)
    returns tail : (M,)
    """
    grid = np.asarray(grid)
    locs = np.asarray(locs)
    masses = np.asarray(masses)

    # sort by location
    order = np.argsort(locs)
    locs_sorted = locs[order]
    w_sorted    = masses[order]

    # right-cumulative sum
    rcum = np.cumsum(w_sorted[::-1])[::-1]   # same length as locs_sorted

    # for each grid point, find first index with loc >= grid
    idx = np.searchsorted(locs_sorted, grid, side='left')

    tail = np.zeros_like(grid, dtype=float)
    in_range = idx < locs_sorted.size
    tail[in_range] = rcum[idx[in_range]]
    # (where idx == K, tail stays 0)
    return tail



def return_gamma_shape_rate_from_mean_variance(mean,variance):
    alpha = mean**2/variance
    beta = mean/variance
    return alpha,beta





def thin_uniform(arr, N):
    """
    Uniformly subsample an array along axis 0 without randomness.

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (num_samples, dim)
    N : int
        Number of samples to keep (htinned_num)

    Returns
    -------
    np.ndarray
        Thinned array of shape (N, dim)
    """
    num_samples = arr.shape[0]
    if N >= num_samples:
        return arr.copy()  # nothing to thin

    indices = np.linspace(0, num_samples - 1, N, dtype=int)
    return arr[indices]


#Computing other possible summaries of the sample array
def approximate_mode(sample_array):
    """
    Approximate mode (MAP-like) sample from posterior realizations.
    Selects the sample with smallest L2 distance to the pointwise median.
    """
    median_curve = np.median(sample_array, axis=0)
    distances = np.linalg.norm(sample_array - median_curve, axis=1)
    mode_idx = np.argmin(distances)
    return sample_array[mode_idx]
def summarize_sample_array(arr):
    """Compute mean, median, and approximate mode curves for (num_samples, dim) array."""
    mean_curve = np.mean(arr, axis=0)
    median_curve = np.median(arr, axis=0)
    mode_curve = approximate_mode(arr)
    return mean_curve, median_curve, mode_curve




#Cheking functions






def _check_group_times():
    T = 10.0
    time_axis = np.linspace(0,T,int(T)*10)
    beta=1
    alpha = 1/beta
    theta = -1.0
    C=1.0
    muw = 0.0
    sigmaw = 1.0
    sub_jump_sizes,jump_times = gamma_process_jumps((beta,C),T,sim_rate=200)
    grouped_jump_sizes,grouped_jump_times = group_jumps(sub_jump_sizes,jump_times,time_axis)
    return





#The following is the actions to be run when the file is called directly
def main():
    #langevin_matrix_exponent(1,np.random.rand(2,2))
    indices = compute_overlapping_block_indices(20,10,100)
    #_check_group_times()
    return

if __name__ == "__main__":
    main()
