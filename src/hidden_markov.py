import numpy as np
from scipy import stats

class EMHMMSpikes(object):
  '''
  Class for Expectation-Maximization (E-M) algorithm on neuron spikes.
  
  Attributes:
  Y: numpy.ndarray
    an array of neuron spikes. 
    Shape: (no. of neurons, no. of trials, no. of time bins, 1)
  N: int
    number of neurons.
  K: int
    number of hidden states.
  dt: float
    time bin size.
  A: numpy.ndarray
    state transition matrix.
    Shape: (K, K)
  L: numpy.ndarray
    neuron firing rates under every state inferred by the model.
    Shape: (N, K)
  gamma: numpy.ndarray
    singleton marginal distribution of states.
    Shape: (N, T, K)
  xi: numpy.ndarray
    pairwise marginal distribition of nodes in HMM.
    Shape: (N, T-1, K, K)

  Methods:
  __init__(Y, N=698, K=5, dt=0.01)
    init class.
  _forward(T, alpha, emit, log_A)
    forward pass in E-step.
  _backward(T, beta, emit, log_A)
    backward pass in E-step.
  _e_step
    perform the E-step to learn the HMM spiking model.
  _m_step
    perform the M-step to learn the HMM spiking model.
  _init_params
    initialize paramaters for the E-M algorithm.
  run
    run the E-M algorithm to learn the HMM spiking model.

  '''
  def __init__(self, Y, K=5, dt=0.01):
    '''
    Init class.

    Arguments:
      Y: numpy.ndarray
    an array of neuron spikes. 
    Shape: (no. of neurons, no. of trials, no. of time bins)
    K: int
      number of hidden states. Default at 5.
    dt: float
      time bin size. Default at 0.01 second.
    '''
    self.Y = Y[..., np.newaxis]
    self.N = len(Y)
    self.K = K
    self.dt = dt
    self.A = None
    self.L = None
    self.gamma = None
    self.xi= None

  def _forward(self, T, alpha, emit, log_A):
    '''
    Forward pass in E-step.
    '''
    for t in range(1, T):
      curr = log_A + alpha[:, t-1, :, np.newaxis]
      maxcurr = curr.max(-2)
      alpha[:, t] = (emit[:, t] \
                    + maxcurr \
                    + np.log(np.exp(curr - maxcurr[:, np.newaxis]).sum(-2)))

  def _backward(self, T, beta, emit, log_A):
    '''
    Backward pass in E-step.
    '''
    for t in range(T-2, -1, -1):
      curr = log_A + beta [:, t+1, np.newaxis] + emit[:, t + 1, np.newaxis]
      maxcurr = curr.max(-1)
      beta[:, t] = maxcurr \
                + np.log(np.exp(curr - maxcurr[..., np.newaxis]).sum(-1))

  def _e_step(self, psi, A, L):
    '''
    Perform the E-step to learn the HMM spiking model.
    '''
    # init Poisson emission, log-scale A, alpha, and beta
    emit = np.sum(stats.poisson(L * self.dt).logpmf(self.Y[..., np.newaxis]), axis=-2)
    n_trials, T = self.Y.shape
    K = len(psi)
    log_A = np.log(A)
    alpha = np.zeros((n_trials, T, K))
    alpha[:, 0] = emit[:, 0] + np.log(psi)
    beta  = np.zeros((n_trials, T, K))

    # forward-backward
    self._forward(T, alpha, emit, log_A)
    self._backward(T, beta, emit, log_A)

    # computes gamma and xi
    maxcurr = np.max(alpha[:, -1], axis=-1)
    log_like = np.log(np.exp(alpha[:, -1] 
                            - np.sum(maxcurr[:, np.newaxis], axis=-1))) \
                                          + maxcurr
    gamma = np.exp(alpha + beta  - np.expand_dims(log_like, (1, 2)))
    xi = np.exp(alpha[:, :-1, :, np.newaxis] 
                + (emit + beta)[:, 1:, np.newaxis] 
                + log_A - np.expand_dims(log_like, (1, 2, 3)))
    return gamma, xi

  def _m_step(self, gamma, xi):
    '''
    Perform the M-step to learn the HMM spiking model.
    '''
    # update psi, A, and L
    psi_ = np.mean(gamma[:, 0], axis=0)
    psi_  = psi_ / np.sum(psi_)
    A_ = np.sum(xi, axis=(0, 1)) \
            / np.sum(gamma[:, :-1], axis=(0, 1))[:, np.newaxis]
    L_ = np.sum((np.swapaxes(self.Y, -1, -2) @ gamma), axis=0) \
            / np.sum(gamma, axis=(0, 1)) \
            / self.dt
    return psi_, A_, L_

  def _init_params(self, max_transition_rate, max_firing_rate):
    '''
    Initialize paramaters for the E-M algorithm.
    '''
    psi = np.arange(1, self.K + 1) # initial emissioattan
    psi = psi / np.sum(psi)

    A = np.ones((self.K, self.K)) * max_transition_rate * self.dt / 2
    A = (1 - np.eye(self.K)) * A
    A = A + np.diag(1 - np.sum(A, axis=1))

    L = np.random.rand(self.N, self.K) * max_firing_rate
    return psi, A, L

  def run(self, epochs=10, max_transition_rate=5, max_firing_rate=60):
    '''
    Run the E-M algorithm to learn the HMM spiking model.

    Arguments:
    epochs: int
      number of epochs to train the model. Default at 10.
    max_transition_rate: int
      maximum state transition rate per second for parameter initialization. 
      Default at 5.
    max_firing_rate: int
      maximum firing rate of neurons (Hz) for parameter initialization. 
      Default at 60.

    Return:
      self.gamma, self.xi 
    '''
    psi, A, L = self._init_params(epochs, max_transition_rate, max_firing_rate)
    for e in range(epochs):
        # E-step
        gamma, xi = self._e_step(psi, A, L)
        # M-step
        psi_, A_, L_ = self._m_step(gamma, xi)
        dp, dA, dL = psi_ - psi, A_ - A, L_ - L
        psi, A, L = psi_, A_, L_
    gamma, xi = self._e_step(psi, A, L)
    self.A = A
    self.L = L
    self.gamma = gamma
    self.xi = xi
    return self.gamma, self.xi         
