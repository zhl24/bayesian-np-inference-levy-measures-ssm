import numpy as np
from scipy.linalg import expm #This is the automatic matrix expnent solver
import math
from scipy.special import logsumexp
from Common_Tools import*
from Levy_Generators import *
from numba import jit

#The same Kalman filters used in IIB. If the inputs are equally valid, they should give the correct results, so no need for additional test.
#The main point is to ensure the correct input format when using these 2 algorithms.
def Kalman_transit(X, P, f, Q):
    # Perform matrix multiplication
    X_new = f @ X
    P_new = f @ P @ f.T + Q
    return X_new, P_new



    
#We again need the current states as the first two inputs, but now we need an obervation. g is the emission matrix, mv and R are 
#the observation mean and noises which determine most of the Kalman filtering difficulties
#@jit(nopython=True)

def Kalman_correct(X, P, Y, g, R): #The log_marginal returned would be used as the particle weight in marginalised particle filtering scheme.

    Ino = Y - g @X  # Innovation term, just the predicton error
    S = g @ P @ g.T + R  # Innovation covariance

    K = P @ g.T /S 

    n = np.shape(P)[0]
    I = np.identity(n)
    log_cov_det = np.log(S)  # Use S for log marginal likelihood
    cov_inv = 1/S
    
    return X + K @ Ino, (I - K @ g) @ P, log_cov_det, np.dot((Ino).T, np.dot(cov_inv, (Ino)))