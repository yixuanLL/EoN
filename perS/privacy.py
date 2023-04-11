import numpy as np
import tensorflow as tf

num_features = 100
num_out = 1
EPSILON = 0.001
CONST_C = 0.25

def gen_noise(data_size, regularizer):
    eps_prime = EPSILON - np.log(
        1. + (2. * CONST_C / (data_size * regularizer) + (CONST_C ** 2. / (data_size ** 2. * regularizer ** 2.))))
    if eps_prime > 0.0:
        delta = 0.0
        print('eps_prime > 0, ', eps_prime)
    else:
        delta = CONST_C / (data_size * (np.exp((0.25 * eps_prime)) - 1.)) - regularizer
        eps_prime = 0.5 * EPSILON
        print('eps_prime <= 0, ', eps_prime)
    beta = 0.5 * eps_prime
    noise_mat = tf.random_normal(shape=[1, num_features], mean=0.0, stddev=beta, seed=1, dtype=tf.float32)
    return noise_mat, delta