"""Linear Lévy state-space models via shot-noise representations.

Builds hidden-state paths and noisy observations for the Langevin model driven
by a Lévy (NVM) noise process, using the series representation of the driving
jumps.

Part of the code accompanying:
    Lin, B. Z. & Godsill, S. (2025). Bayesian Non-Parametric Inference for
    Lévy Measures in State-Space Models. arXiv:2505.22587.
"""
import numpy as np
from scipy.linalg import expm #This is the automatic matrix expnent solver
import math
from scipy.special import logsumexp
from Common_Tools import*
from Levy_Generators import *
from numba import jit



#Langevin System
def langevin_hidden_response(jump_sizes,jump_times,theta,time_axis,initial_position = np.zeros((2,1))):
    N = len(time_axis) #The number of time points
    #Define the driving noise contribution matix
    h = np.zeros((2,1))
    h[1,0] = 1

    #Compute the initial transient
    initial_position = initial_position[np.newaxis,np.newaxis,:,:] #Broadcast for vectorised operation
        #Broadcast the time axis array into a matrix of (1,N) for the use of the matrix exponent function
    initial_transient = langevin_matrix_exponent(theta,time_axis[np.newaxis,:]) @ initial_position #(1,N,2,2) array of exponential matrices @ (1,1,2,1) broadcasted matrix into (1,N,2,1)
    
    #Need a matrix to hold the evolution of the differential time for each jump time. N time points and c jump times
        #Create a matrix of (c,N), each row is the evolution of differential jump time for each jump
    differential_jump_times = time_axis[np.newaxis,:] - np.transpose(jump_times) #Automatic broadcasted elementwise operations (1,N)-(c,1)=>(c,N)
    #Below are the matrices of jump evolution with along the discrete time axis
    exp_jumps = langevin_matrix_exponent(theta,differential_jump_times) #(c,N,2,2)

    #Below is the most difficult operation in the whole function
        #(c,N,2,2)@(1,1,2,1)=>(c,N,2,1), then elementwise product with scalar matrix(1,c)
        #(1,c) has to be transposed into (c,1), and then it should be broadcasted into (c,1,1,1) for automatic repeatition according to each jump size
    system_jumps = (exp_jumps @ h[np.newaxis,np.newaxis,:,:]) * np.transpose(jump_sizes)[:,:,np.newaxis,np.newaxis] #(c,N,2,1)
    #integrate the system jumps for each time instant to determine the discrete system path
    hidden_process_path = np.zeros((2,N)) #The result is the (d,N) matrix of process path
    for i in range(N): #Have to iterate over each time instant to use the integrate_to_path function. Result is the path at the time instant i
        jump_matrix = system_jumps[:,i,:,0] #This value taking removes the indexed dimensions, so the result is (c,2) = (c,d)
        jump_path = integrate_to_path(np.transpose(jump_matrix),jump_times,[time_axis[i]]) # (d,1) matrix
        hidden_process_path[:,i] = jump_path[:,0] #Array to array, so need to convert the column matrix into an array without shape information
    hidden_process_path += np.transpose(initial_transient[0,:,:,0]) #(d,N) output
    return hidden_process_path


#Always d=2 also, by the definition of Langevin system strutcure.
def langevin_observations(hidden_process_path,C=np.identity(2),H=np.identity(2),mu=np.zeros((2,1))):
    #C and mu are the mean and covariance of the Gaussian random noise
    #H is the emission structure matrix
    
    #Get the state dimension and the number of time points first
    d,N = np.shape(hidden_process_path)

    #Cholesky decomposition to obtain the Gaussian noise from the multivariate Gaussian first
    L = np.linalg.cholesky(C)  #Also (d,d)=(2,2)
    crude_noise = np.random.randn(N,d) #Standard normal noise of (N,d)
    target_noise = (crude_noise @ L.T + mu.T).T #Target multivariate Gaussian noise of (N,d).T = (d,N)
    
    #Observations
    observations = H @ hidden_process_path + target_noise
    return observations






#Check Functions:
def _langevin_test(with_observation_noise=False):
    if with_observation_noise:
        observation_noise_level = 0.1
        T = 10.0
        time_axis = np.linspace(0,T,int(T)*10)
        beta=1
        alpha = 1/beta
        C=0.1
        muw = 0.0
        sigmaw = 1.0
        theta = -1.0
        sub_jump_sizes,jump_times = gamma_process_jumps((beta,C),T,sim_rate=200)
        nvm_jump_sizes = nvm_process_jumps(sub_jump_sizes,muw,sigmaw)
        langevin_path = langevin_hidden_response(nvm_jump_sizes,jump_times,theta,time_axis) #(2,N) process path
        noisy_langevin_path = langevin_observations(langevin_path,observation_noise_level*np.identity(2))
        x_path = noisy_langevin_path[0,:]
        xdot_path = noisy_langevin_path[1,:]
        plt.plot(time_axis,xdot_path)
        plt.show()
    else:
        #Generate the jump process first
        T = 10.0
        time_axis = np.linspace(0,T,int(T)*10)
        beta=1
        alpha = 1/beta
        theta = -1.0
        C=0.1
        muw = 0.0
        sigmaw = 1.0
        sub_jump_sizes,jump_times = gamma_process_jumps((beta,C),T,sim_rate=200)
        nvm_jump_sizes = nvm_process_jumps(sub_jump_sizes,muw,sigmaw)

        langevin_path = langevin_hidden_response(nvm_jump_sizes,jump_times,theta,time_axis) #(2,N) process path
        x_path = langevin_path[0,:]
        xdot_path = langevin_path[1,:]
        plt.plot(time_axis,xdot_path)
        plt.show()
    return








#The following is the actions to be run when the file is called directly
def main():
    _langevin_test(False)#True to add noise, False to remove noise

if __name__ == "__main__":
    main()
