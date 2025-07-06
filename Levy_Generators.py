import numpy as np
from scipy.special import logsumexp
from Common_Tools import*
from numba import jit
import matplotlib.pyplot as plt
from tqdm import tqdm


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









