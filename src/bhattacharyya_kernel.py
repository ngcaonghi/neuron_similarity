import numpy as np
import time
'''
Utility program to get the bhattacharyya distance and affinity matrices between
HMM models from their gammas and xis.
'''
class IllegalArgumentError(ValueError):
    pass

def gamma_kernel(gamma_dt_1, gamma_dt_2):
    '''
    Helper method to perform gamma kernel on 2 gamma arrays of 2 neurons.
    '''
    return np.sqrt(np.prod([gamma_dt_1, gamma_dt_2], axis=0)).mean(axis=0).sum()

def xi_kernel(xi_dt_1, xi_dt_2):
    '''
    Helper method to perform xi kernel on 2 xi arrays of 2 neurons.
    '''
    return np.sqrt(np.prod([xi_dt_1, xi_dt_2], axis=0)).mean(axis=0)

def bhat_kernel(gamma_1, gamma_2, xi_1, xi_2, zero_distances, inf_distances):
    '''
    Helper method to perform Bhattacharyya kernel on the paramteters of 2 HMM
    models, namely gamma and xi arrays.
    '''
    phi = gamma_kernel(gamma_1[:, 0, :], gamma_2[:, 0, :]) # initial vector
    for i in range(xi_1.shape[1]):
        # iterative product of gammas, xis and phi over all time bins
        xi_t = xi_kernel(xi_1[:, i, :], xi_2[:, i, :])
        gamma_t = gamma_kernel(gamma_1[:, i, :], gamma_2[:, i, :])
        phi = np.sum(xi_t * gamma_t * phi)
    result = np.log(gamma_kernel(gamma_1[:, -1, :], gamma_2[:, -1, :]) * phi)
    # handling zero distances:
    if np.isclose(result, 0, atol=1e-3) and zero_distances==False:
        return 1e-3
    elif np.isclose(result, 0, atol=1e-3) and zero_distances==True:
        return 0
    # handling infinity distances:
    if result==np.inf and inf_distances==False:
        return 1e3
    return -result

def bhat_matrix(batch_gamma, batch_xi, matrix_type='distance',
                zero_distances=False, inf_distances=False):
    '''
    Method to compute the distance or affinity matrix of HMM models using 
    Bhattacharyya distance coefficient. 

    Arguments:
    batch_gamma: numpy.ndarray
        an array of gammas of all neurons.
        Shape: (no. of neurons, trial length, no. of states)
    batch_xi: numpy.ndarray
        an array of xis of all neurons.
        Shape: (no. of neurons, trial length - 1, no. of states, no. of states)
    matrix_type: str
        type of matrix. Default is 'distance'. Options are 'distance' and
        'affinity'. 
    zero_distances: boolean
        whether one wants the Bhattacharyya coefficient of identical models 
        to be zeros. Default at False.
        The default behavior is returning 1e-3 instead of 0 at each diagonal 
        entry of the matrix, so that certain computations with 0 would be
        hassle-free.
    inf_distances: boolean
        whether one wants the Bhattacharyya coefficient of identical models 
        to be numpy.inf. Default at False.
        The default behavior is returning 1e3 instead of numpy.inf, so that 
        certain further computations with defined values would be hassle-free.

    Return:
        a Gram matrix of Bhattacharyya distance coefficients.

    Print:
        log of remaining time in minutes, since the complexity of the algorithm 
        behind this method is polynomial.
    
    Raise:
    IllegalArgumentError: when the matrix type is not 'distance' or 'affinity'.
    '''
    N = batch_gamma.shape[0]
    gram = np.zeros([N, N])
    for r in range(gram.shape[0]):
        tic = time.clock()
        for c in range(gram.shape[1]):
            gram[r][c] = bhat_kernel(batch_gamma[r], batch_gamma[c], 
                         batch_xi[r], batch_xi[c], zero_distances)
        toc = time.clock()
        print('remaining time: {:.2f} minutes'.format((toc - tic) \
                                                    * (gram.shape[0] - r) / 60))
    if matrix_type=='distance':
        return gram
    elif matrix_type=='distance':
        return np.reiprocal(gram)
    else:
        raise IllegalArgumentError('matrix_type must be "distance" or "affinity".')

