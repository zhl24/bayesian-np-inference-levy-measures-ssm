import numpy as np
from scipy.linalg import expm #This is the automatic matrix expnent solver
import math
from scipy.special import logsumexp
from Common_Tools import*
from numba import jit
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import gamma
from numpy.random import default_rng as _np_default_rng  # 取个别名，绕过你自己的 monkey-patch

def gamma_process_jumps(gamma_parameters,T,sim_rate=100):
    #Gamman parameters (beta,C)
    beta = gamma_parameters[0]
    C = gamma_parameters[1]
    c = sim_rate * int(T) #The total number of simulations

    #Jump size generation. Vectorised generation of Poisson epochs to be converted into jump sizes
    random_intervals = np.random.exponential(1/T,size = (1,c)) #(1,c) row matrix of floats
        #Cumulative sum will be carried out along the corresponding axis
    poisson_epochs = np.cumsum(random_intervals,axis = 1) # (1,c) row matrix of floats
    jump_sizes = 1/(beta*(np.exp(poisson_epochs/C)-1))   #(1,c) row matrix of floats
    acceptance_probabilities = (1+beta*jump_sizes)*np.exp(-beta*jump_sizes)
    probability_checks = np.random.uniform(size=(1,c))
    acceptance_matrix = acceptance_probabilities>=probability_checks #(1,c) matrix of true and false
    #acceptance_matrix = acceptance_matrix.astype(int) #True for 1 and False for 0, this astype operation is just for numpy array
    jump_sizes = jump_sizes[acceptance_matrix] #Only the accepted jumps have non-zero sizes.
    jump_times = np.random.uniform(size=(1,c))*T # (1,c) matrix of floats
    jump_times = jump_times[acceptance_matrix]

    jump_sizes = jump_sizes.reshape(1, -1)
    jump_times = jump_times.reshape(1, -1)
    return jump_sizes,jump_times







def tempered_stable_process_jumps(ts_parameters,T,sim_rate=100):
    #Tempered stable prcocess parameters (beta,alpha,C)
    beta = ts_parameters[0] #This is the same as the Gamma inverse scale parameter
    alpha = ts_parameters[1] #When alpha=0, the process is just a Gamma process
    C = ts_parameters[2] #The rate parameter of the process
    c = sim_rate * int(T) #The total number of simulations over the whole time frame

     #Jump size generation. Vectorised generation of Poisson epochs to be converted into jump sizes
    random_intervals = np.random.exponential(1/T,size = (1,c)) #(1,c) row matrix of floats
        #Cumulative sum will be carried out along the corresponding axis
    poisson_epochs = np.cumsum(random_intervals,axis = 1) # (1,c) row matrix of floats
    jump_sizes = (alpha*poisson_epochs/C)**(-1/alpha)   #(1,c) row matrix of floats
    acceptance_probabilities = np.exp(-beta * jump_sizes)

    probability_checks = np.random.uniform(size=(1,c))
    acceptance_matrix = acceptance_probabilities>=probability_checks #(1,c) matrix of true and false
    #acceptance_matrix = acceptance_matrix.astype(int) #True for 1 and False for 0, this astype operation is just for numpy array
    jump_sizes = jump_sizes[acceptance_matrix] #Only the accepted jumps have non-zero sizes.
    jump_times = np.random.uniform(size=(1,c))*T # (1,c) matrix of floats
    jump_times = jump_times[acceptance_matrix]

    jump_sizes = jump_sizes.reshape(1, -1)
    jump_times = jump_times.reshape(1, -1)
    return jump_sizes,jump_times






def nvm_process_jumps(subordinator_jump_sizes,muw,sigmaw):
    d,c = np.shape(subordinator_jump_sizes)# Here d is always 1 since we are considering the 1 dimensional Levy prcoess here only. c is the total number of jumps
    jump_sizes = muw*subordinator_jump_sizes + sigmaw*np.sqrt(subordinator_jump_sizes)*np.random.randn(d,c) #Note the special way that randn takes the dimension argument
    return jump_sizes





def inferred_process_jumps(sample_rate,sample_measure,T):
    #The data types here follow the standard convention as before
    """
    sample_rate: float Poisson rate for the compound Poisson process inferred
    sample_measure: (2,K) matrix, with the first row being the probabilities and second row being the corresponding random values
    T: float for the total simulation interval
    """
    probabilities = sample_measure[0, :].copy()  # Shape (K,)
    locations = sample_measure[1, :]      # Shape (K,)

    poisson_rate = sample_rate * T 
    Nj = np.random.poisson(lam=poisson_rate.item())
    jump_times = np.random.uniform(low=0, high=T, size=Nj) #(Nj,) array
    counts = np.random.multinomial(Nj, probabilities)
    jump_sizes = np.repeat(locations, counts) #(N_j,) array
    jump_sizes = jump_sizes.reshape(1, -1) #(1,N_j) matrix
    jump_times = jump_times.reshape(1, -1) #(1,N_j) matrix
    return jump_sizes,jump_times


#A compound Poisson generator using the DP distirbution for the jump sizes.
def compound_poisson_DP(rate,sample_measure,T):
    """
    Inputs:
        rate: float, Poisson rate
        sample_measure: (2,K) sample from a DP
        T: float, time duration for the generation
    Outputs:
        jump_sizes: (1,c)
        jump_times: (1,c)
    """
    probabilities = sample_measure[0, :].copy()  # Shape (K,)
    locations = sample_measure[1, :]      # Shape (K,)

    #Generate the number of jumps in the interval using a Poisson process
    poisson_rate = rate * T 
    Nj = np.random.poisson(lam=poisson_rate.item())
    #Simulate the jump times
    jump_times = np.random.uniform(low=0, high=T, size=Nj) #(Nj,) array
    #Simulate the jump sizes from multinomial
    counts = np.random.multinomial(Nj, probabilities)
    jump_sizes = np.repeat(locations, counts) #(Nj,) array also

    return jump_sizes.reshape(1,-1), jump_times.reshape(1,-1)




def compound_poisson_gamma_mixture(rate,
                                   shapes,
                                   scales,
                                   weights,
                                   T,
                                   return_sorted=True,
                                   seed=None):
    """
    Simulate a compound Poisson process with Gamma–mixture jump sizes.

    Parameters
    ----------
    rate   : float
        Poisson rate λ.
    shapes : array_like, shape (K,)
        Shape parameters α_k of the K Gamma components.
    scales : array_like, shape (K,)
        Scale parameters θ_k (>0) of the K Gamma components.
    weights: array_like, shape (K,)
        Mixture weights w_k (w_k ≥ 0, sum to 1).
    T      : float
        Observation window length.
    return_sorted : bool, optional (default True)
        If True, jump_times are returned in ascending order.
    seed   : int or None, optional
        RNG seed for reproducibility.

    Returns
    -------
    jump_sizes : ndarray, shape (1, N_jumps)
    jump_times : ndarray, shape (1, N_jumps)
    """
    # 用原生 default_rng，不会递归
    rng = _np_default_rng(seed)

    # 1) Number of jumps
    Nj = rng.poisson(rate * T)
    if Nj == 0:
        return np.empty((1, 0)), np.empty((1, 0))

    # 2) Choose mixture components for each jump
    comp_idx = rng.choice(len(weights), size=Nj, p=weights)

    # 3) Sample jump sizes from the selected Gamma components
    jump_sizes = rng.gamma(shape=np.asarray(shapes)[comp_idx],
                           scale=np.asarray(scales)[comp_idx])

    # 4) Sample jump times
    jump_times = rng.uniform(0.0, T, size=Nj)

    if return_sorted:
        order = np.argsort(jump_times)
        jump_times = jump_times[order]
        jump_sizes = jump_sizes[order]

    return jump_sizes.reshape(1, -1), jump_times.reshape(1, -1)




#Some check function for the distirbution of the Gamma process generator
def _check_gamma_process_jumps(check_type):
    #0 for density check
    #1 for path check
    if check_type==0:
        from scipy.stats import gamma  #The gamma random variable
        N = 10000
        T = 10.0
        beta_series = np.linspace(0.5,5,3)
        alpha_series = 1/beta_series  #To adjust to the public generator convention
        C_series = np.linspace(0.5,2,3)

        fig, axes = plt.subplots(3, 3, figsize=(12, 12))

        for i in tqdm(range(len(alpha_series)),desc = "Progress"):
            for j in range(len(C_series)):
                C = C_series[j]
                alpha = alpha_series[i]
                beta = beta_series[i]
                samples = []
                #Take samples at time t=1 to check if the empirical distirbution matches the gamma distribution with the corresponding parameters
                for k in range(N): 
                    jump_sizes,jump_times = gamma_process_jumps((beta,C),T,sim_rate=200)
                    sample = integrate_to_path(jump_sizes,jump_times,[1.0]) #This is a matrix of dimension (1,1)
                    samples.append(sample[0,0])
                axes[i,j].hist(samples, bins=50, density=True, alpha=0.6, color='b', label="Gamma Process Samples")
                
                # Generate x values
                x = np.linspace(min(samples), max(samples), 1000)
                
                # Plotting gamma distribution
                axes[i,j].plot(x, gamma.pdf(x, C, scale=alpha), 'r', label="Gamma Distribution")
                axes[i,j].set_title("Gamma Distribution \n with Shape Parameter C:{} \nScale Parameter alpha (1/beta):{}".format(round(C,3), round(alpha,3)))

        plt.tight_layout()
        plt.savefig("Gamma_Validity_Check.png")
        plt.show()
    elif check_type == 1:
        T = 10.0
        time_axis = np.linspace(0,T,int(T)*10)
        beta=1
        alpha = 1/beta
        C=10
        jump_sizes,jump_times = gamma_process_jumps((beta,C),T,sim_rate=200)
        process_path = integrate_to_path(jump_sizes,jump_times,time_axis)
        plt.plot(time_axis,process_path[0,:])
        plt.show()
    return




def _check_ts_process():
    T = 10.0
    time_axis = np.linspace(0,T,int(T)*10)
    beta=0.2
    alpha = 0.5
    C=0.2
    jump_sizes,jump_times = tempered_stable_process_jumps((beta,alpha,C),T)
    process_path = integrate_to_path(jump_sizes,jump_times,time_axis)
    plt.figure()
    plt.plot(time_axis,process_path[0,:])
    plt.show()
    return




def _check_gamma_nvm_process():
    T = 10.0
    time_axis = np.linspace(0,T,int(T)*10)
    beta=1
    alpha = 1/beta
    C=1
    muw = 0.0
    sigmaw = 1.0
    sub_jump_sizes,jump_times = gamma_process_jumps((beta,C),T,sim_rate=200)
    nvm_jump_sizes = nvm_process_jumps(sub_jump_sizes,muw,sigmaw)
    process_path = integrate_to_path(nvm_jump_sizes,jump_times,time_axis)
    plt.plot(time_axis,process_path[0,:])
    plt.show()
    return




#The following is the actions to be run when the file is called directly
def main():
    #_check_gamma_process_jumps(check_type=0)
    #_check_gamma_nvm_process()
    _check_ts_process()


if __name__ == "__main__":
    main()


