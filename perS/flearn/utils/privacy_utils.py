import pickle
import numpy as np
import math
import sympy as sp
import time
from flearn.utils.utils import transform, discrete

from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib


#################################ADD NOISE#######################################
def add_laplace(updates, sensitivity, epsilon):
    '''
    inject laplacian noise to a vector
    '''
    lambda_ = sensitivity * 1.0 / epsilon
    updates += np.random.laplace(loc=0, scale=lambda_, size=updates.shape)
    return updates

def add_gaussian(updates, eps, delta, sensitivity):
    '''
    inject gaussian noise to a vector
    '''
    sigma = (sensitivity/eps) * math.sqrt(2 * math.log(1.25/delta))
    updates += np.random.normal(0, sigma) # question: same noise on each dimension
    return updates


def one_gaussian(eps, delta, sensitivity):
    '''
    sample a gaussian noise for a scalar
    '''
    sigma = (sensitivity/eps) * math.sqrt(2 * math.log(1.25/delta))
    return np.random.normal(0, sigma)

def one_laplace(eps, sensitivity):
    '''
    sample a laplacian noise for a scalar
    '''
    return np.random.laplace(loc=0, scale=sensitivity/eps)

# faster
def clip_laplace(grad, epsilon, Clip_bound):
    C = Clip_bound
    b = 2*C/epsilon
    u = grad
    exp_part = np.exp((-C-u)/b)
    S = 1 - 0.5 * np.exp((-C+grad)/b) - 0.5 * exp_part
    p = np.random.uniform(0, 1, grad.shape[0])
    sp = S*p
    
    step_point = np.sign(sp - (0.5 - 0.5*exp_part))
    X = u - step_point * b *np.log(1- 2*np.abs(sp - 0.5 + 0.5*exp_part))

    return X

def one_clip_laplace(grad, epsilon, Clip_bound):
    C = Clip_bound
    b = 2*C/epsilon
    u = grad
    exp_part = np.exp((-C-u)/b)
    S = 1 - 0.5 * np.exp((-C+grad)/b) - 0.5 * exp_part
    p = np.random.uniform(0, 1)
    sp = S*p
    
    step_point = np.sign(sp - (0.5 - 0.5*exp_part))
    X = u - step_point * b *np.log(1- 2*np.abs(sp - 0.5 + 0.5*exp_part))

    return X

def full_randomizer(vector, clip_C, eps, delta, mechanism):
    clipped = np.clip(vector, -clip_C, clip_C)
    if mechanism == 'laplace':
        perturbed = add_laplace(clipped, sensitivity=2*clip_C, epsilon=eps)
    return perturbed


####### without scale
def sampling_randomizer_clipLap(vector, choices, clip_C, eps, delta, mechanism):
    vector = np.clip(vector, -clip_C, clip_C)
    for i, v in enumerate(vector):
        if i in choices:
            if mechanism == 'laplace':
                vector[i] = clip_laplace(vector[i], eps, clip_C)
        else:
            vector[i] = 0
    return vector


def clip_randomizer_clipLap(vector, clip_C, eps, delta, mechanism):
    vector = np.clip(vector, -clip_C, clip_C)
    if mechanism == 'laplace':
        vector = clip_laplace(vector, eps, clip_C)
    return vector


def clip_randomizer(vector, clip_C, eps, delta, mechanism): #, moueacc, mouea): 
    if mechanism == 'l1_laplace':
        l1_norm = np.linalg.norm(vector, ord=1)
        interv  = np.linspace(min(vector), max(vector), 100)
        cont = [0]*100
        for v in vector:
            for i in range(100):
                if interv[i]>=v:
                    cont[i]+=1
        print(interv)
        print(cont)
        exit(0)
        # print('l2_norm:', np.linalg.norm(vector, ord=2))
        if l1_norm > clip_C*2:
            vector = vector * (clip_C*2 / l1_norm)
        perturbed = add_laplace(vector, sensitivity=2*clip_C, epsilon=eps)
        return perturbed
    vector = np.clip(vector, -clip_C, clip_C) # check clip way
    if mechanism == 'gaussian':
        perturbed = add_gaussian(vector, eps, delta, sensitivity=2*clip_C)
    elif mechanism == 'laplace':
        perturbed = add_laplace(vector, sensitivity=2*clip_C, epsilon=eps)
    # elif mechanism =='oue':
    #     perturbed = add_oue(vector, epsilon=eps, acc=moueacc, alpha=mouea)
    else:
        perturbed = vector
    return perturbed


def sampling_randomizer(vector, choices, clip_C, eps, delta, mechanism):
    vector = np.clip(vector, -clip_C, clip_C)
    re = add_laplace(np.zeros(vector.shape[0]), sensitivity=2*clip_C, epsilon=eps)
    if mechanism == 'laplace':
        for i,v in enumerate(vector):
            if i in choices:
                re[i] += vector[i]
    return vector

# expectation of clip_lap noise
def E_noisy_grad(e, u, C):
    b = 2*C/e
    exp_part_n = np.exp((-C-u)/b)
    exp_part_p = np.exp((-C+u)/b)
    S = 1 - 0.5 * exp_part_p - 0.5 * exp_part_n
    E_noise = ((b+C+u)*exp_part_n - (b+C-u)*exp_part_p) / (2*S)
    # E_noise = ((b+C)*(exp_part_n -exp_part_p) + 2u ) / (2*S)
    return E_noise+u

# for dp sgd (2016'ccs)
def set_grad_noise(norm, batch_size, epochs, num_samples, eps, delta, min_diff=1):
    start = time.time()
    left = 0.01
    right = 3
    TIME_LIMIT = 60
    searching = True
    while searching:
        noise = (right + left) / 2
        eps_tmp, _ = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(n=num_samples,batch_size=batch_size,epochs=epochs, noise_multiplier=noise, delta=delta)
        diff = eps_tmp - eps
        if abs(diff) < min_diff:
            return noise, eps_tmp, abs(diff)
        else:
            if diff > 0:
                left = noise
            else:
                right = noise

        t = time.time() - start
        if t > TIME_LIMIT:
            print("[WARNING]: cannot find noise level for eps = {}, within given the time {}".format(eps, t))
            print("[INFO]: Noisy_multiplier has been set as {} for epsilon {}".format(noise,eps_tmp))
            searching = False
            return noise, None, None