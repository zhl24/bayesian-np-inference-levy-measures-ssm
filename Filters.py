import numpy as np
from scipy.linalg import expm #This is the automatic matrix expnent solver
import math
from scipy.special import logsumexp
from Common_Tools import*
from Levy_Generators import *
from numba import jit

# #The same Kalman filters used in IIB. If the inputs are equally valid, they should give the correct results, so no need for additional test.
# #The main point is to ensure the correct input format when using these 2 algorithms.
# def Kalman_transit(X, P, f, Q):
#     # Perform matrix multiplication
#     X_new = f @ X
#     P_new = f @ P @ f.T + Q
#     return X_new, P_new



    
# #We again need the current states as the first two inputs, but now we need an obervation. g is the emission matrix, mv and R are 
# #the observation mean and noises which determine most of the Kalman filtering difficulties
# #@jit(nopython=True)

# def Kalman_correct(X, P, Y, g, R): #The log_marginal returned would be used as the particle weight in marginalised particle filtering scheme.

#     Ino = Y - g @X  # Innovation term, just the predicton error
#     S = g @ P @ g.T + R  # Innovation covariance

#     K = P @ g.T /S 

#     n = np.shape(P)[0]
#     I = np.identity(n)
#     log_cov_det = np.log(S)  # Use S for log marginal likelihood
#     cov_inv = 1/S
    
#     return X + K @ Ino, (I - K @ g) @ P, log_cov_det, np.dot((Ino).T, np.dot(cov_inv, (Ino)))


def _sym(A):
    return 0.5 * (A + A.T)

def Kalman_transit(X, P, f, Q, q_floor=1e-12):
    # Symmetrize + tiny jitter to keep PSD even when jumps are rare (Q≈0)
    Q = _sym(Q)
    Q[np.diag_indices_from(Q)] += q_floor

    X_new = f @ X
    P_new = _sym(f @ P @ f.T + Q)
    return X_new, P_new

def Kalman_correct(X, P, Y, g, R, s_floor=1e-12):
    # Scalarize innovation components to avoid silent broadcasting surprises
    Ino = float(np.asarray(Y)) - float(np.asarray(g @ X))
    S   = float(np.asarray(g @ P @ g.T)) + float(np.asarray(R))

    # Floor S to avoid log(0), 1/0; keep it finite and positive
    if not np.isfinite(S) or S <= 0.0:
        S = s_floor

    # Kalman gain
    K = (P @ g.T) * (1.0 / S)   # (n,1)

    # Joseph form keeps P PSD: P_new = (I-Kg)P(I-Kg)^T + K R K^T
    n = P.shape[0]
    I = np.eye(n)
    P_new = (I - K @ g) @ P @ (I - K @ g).T + K * float(np.asarray(R)) * K.T
    P_new = _sym(P_new)

    X_new = X + K * Ino
    log_S = np.log(max(S, s_floor))
    Ei    = (Ino * Ino) / max(S, s_floor)
    return X_new, P_new, log_S, Ei



#Computing the smoothing distributions from the filtering and predictive distributions from a given Kalman filter run.
def rts_smoother(
    filtered_means: np.ndarray,
    filtered_covariances: np.ndarray,
    predicted_means: np.ndarray,
    predicted_covariances: np.ndarray,
    transition_matrices: np.ndarray,
    return_smoothing_gains: bool = False,
):
    """
    Rauch-Tung-Striebel (RTS) Kalman smoother.

    Parameters
    ----------
    filtered_means : np.ndarray
        Filtered state means, shape (T, d, 1) or (T, d).
        filtered_means[t] = E[x_t | y_{1:t}].

    filtered_covariances : np.ndarray
        Filtered state covariances, shape (T, d, d).
        filtered_covariances[t] = Cov[x_t | y_{1:t}].

    predicted_means : np.ndarray
        One-step predictive state means, shape (T, d, 1) or (T, d).
        predicted_means[t] = E[x_t | y_{1:t-1}].
        Note: index 0 is unused by the smoother recursion.

    predicted_covariances : np.ndarray
        One-step predictive state covariances, shape (T, d, d).
        predicted_covariances[t] = Cov[x_t | y_{1:t-1}].
        Note: index 0 is unused by the smoother recursion.

    transition_matrices : np.ndarray
        State transition matrices, shape (T-1, d, d).
        transition_matrices[t] maps x_t -> x_{t+1}.

    return_smoothing_gains : bool, default=False
        If True, also return the RTS smoothing gains J_t with shape (T-1, d, d).

    Returns
    -------
    smoothed_means : np.ndarray
        Smoothed state means, same shape as filtered_means.

    smoothed_covariances : np.ndarray
        Smoothed state covariances, same shape as filtered_covariances.

    smoothing_gains : np.ndarray, optional
        Returned only if return_smoothing_gains=True.
        smoothing_gains[t] = J_t for t = 0, ..., T-2.

    Notes
    -----
    The RTS recursion is:
        J_t = P_{t|t} F_{t+1}^T (P_{t+1|t})^{-1}
        m_{t|T} = m_{t|t} + J_t (m_{t+1|T} - m_{t+1|t})
        P_{t|T} = P_{t|t} + J_t (P_{t+1|T} - P_{t+1|t}) J_t^T
    """
    filtered_means = np.asarray(filtered_means)
    filtered_covariances = np.asarray(filtered_covariances)
    predicted_means = np.asarray(predicted_means)
    predicted_covariances = np.asarray(predicted_covariances)
    transition_matrices = np.asarray(transition_matrices)


    T, d, d2 = filtered_covariances.shape
    means_are_column_vectors = (filtered_means.ndim == 3)

    smoothed_means = filtered_means.copy()
    smoothed_covariances = filtered_covariances.copy()

    if return_smoothing_gains:
        smoothing_gains = np.empty((T - 1, d, d), dtype=filtered_covariances.dtype)

    # Backward recursion:
    # initialize at T-1 (Python index), where smoothing = filtering
    # then recurse for t = T-2, ..., 0
    for t in range(T - 2, -1, -1):
        P_tt = filtered_covariances[t]
        P_t1_t = predicted_covariances[t + 1]
        F_t1 = transition_matrices[t]

        # Compute J_t = P_tt F_t1^T (P_t1_t)^{-1}
        # Avoid explicit inverse for better numerical stability.
        # Solve: P_t1_t^T X^T = (P_tt F_t1^T)^T
        # Then X = J_t
        Ptt_FtT = P_tt @ F_t1.T
        J_t = np.linalg.solve(P_t1_t.T, Ptt_FtT.T).T

        if return_smoothing_gains:
            smoothing_gains[t] = J_t

        # Mean update
        mean_residual = smoothed_means[t + 1] - predicted_means[t + 1]
        smoothed_means[t] = filtered_means[t] + J_t @ mean_residual

        # Covariance update
        cov_residual = smoothed_covariances[t + 1] - predicted_covariances[t + 1]
        smoothed_covariances[t] = P_tt + J_t @ cov_residual @ J_t.T

        # Optional symmetrization to reduce numerical asymmetry
        smoothed_covariances[t] = 0.5 * (
            smoothed_covariances[t] + smoothed_covariances[t].T
        )

    if return_smoothing_gains:
        return smoothed_means, smoothed_covariances, smoothing_gains
    return smoothed_means, smoothed_covariances