from random import choices
import numpy as np
from tqdm import tqdm, trange
import math
import random

from .fedbase_silo import BaseFedarated
from flearn.utils.tf_utils import process_grad
from flearn.utils.utils import clip, sparsify, transform, topindex, binarySearch
from flearn.utils.privacy_utils import clip_randomizer_clipLap, sampling_randomizer_clipLap_padding, E_noisy_grad
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
from tensorflow.python.training.adam import AdamOptimizer
from flearn.optimizers.pgd import PerturbedGradientDescent


class Server(BaseFedarated):
    '''
    traditional LDP-FL
    1. sampling depends on self.rate
    2. gaussian distribution for (epsilon, delta_lk)-LDP(DP) privacy
    '''
    def __init__(self, params, learner, dataset):
        print('Using Federated prox to Train (personalized-eps-FL-PostTopk)')
        self.inner_opt = PerturbedGradientDescent(learning_rate=params['learning_rate'], mu=params['mu'])
        super(Server, self).__init__(params, learner, dataset)
        self.clip_C = self.norm
        self.m_p = int(self.clients_per_round / self.mp_rate)
        # print("Setting the padding size for each dimension with ", self.m_p)
        # self.em_s = self.clients_per_round / self.rate
        self.topk = int( (self.dim_model + self.dim_y)/self.rate)
        print("Topk selecting {} dimensions".format(self.topk))
        self.choice_list = []
        self.noisy_grads = self.E_noisy_gradients()

    def train(self):
        '''Train using Federated Proximal'''
        self.train_grouping()

    def local_process(self, flattened, eps):
        eps_lk = eps / self.topk
        vector = clip_randomizer_clipLap(flattened, self.clip_C, eps_lk, self.delta, self.mechanism)
        # post top-k
        choices = np.array(random.sample(list(range(len(flattened))), self.topk))
        re = np.array([0.0] * vector.shape[0])
        re = clip_randomizer_clipLap(re, self.clip_C, eps_lk, self.delta, self.mechanism)
        np.put(re, choices.tolist(), vector[[choices]].tolist())
        return re
        # if self.de_bias_end == 'client':
        #     vector = self.de_bias_client(vector, eps_lk, choices)
        # return vector
      
    def set_epsilon(self, eps, s_n):
        if self.pad_mod == 'on':
            eps_num = self.pad_sample
        else:
            eps_num = s_n
        # uniform distribution eps
        left = self.per_left
        right = self.per_right
        # left = eps/self.trim_dim/5
        # right = left * 9
        eps = np.random.uniform(left , right, eps_num)
        
        # normal distribution eps
        # eps = np.random.normal(5, 0.1)

        return eps.tolist()

    def server_process(self, messages):
        '''
        basic aggregate, scale with rate when Top-k is applied (when rate > 1)
        '''
        # avg = self.aggregate_p(messages)
        if self.pad_mod == 'on':
            rate = self.up_grads/self.pad_sample         
        else:
            rate = 1
            
        total_weight, base = self.aggregate_e(messages)
        avg = self.average(total_weight / self.rate * rate, base)
        self.choice_list = []
        
        if self.de_bias_end == 'server':
            avg = self.de_bias_server(avg)
        return avg
    
    def de_bias(self, noisy_grads, v):
        index = binarySearch(noisy_grads, 0, 999, v)
        u = np.linspace(-self.clip_C, self.clip_C, 1000)
        # grad = u[index-2]
        grad = u[index-1]
        return grad

    def de_bias_server(self, vector):
        for l in range(len(vector)):
            if l==0:
                for i, vs in enumerate(vector[l]):
                    for k, v in enumerate(vs):
                        vector[l][i][k] = self.de_bias(self.noisy_grads, v)
            else:
                for j, v in enumerate(vector[l]):
                    vector[l][j] = self.de_bias(self.noisy_grads, v)
        return vector
    
    def E_noisy_gradients(self):
        if self.eps_dist == 'uniform':
            e = np.linspace(self.per_left / self.topk, self.per_right / self.topk, 1000)
        u = np.linspace(-self.clip_C, self.clip_C, 1000)
        EE = []
        for ui in u:
            E_mean = np.mean(np.array(E_noisy_grad(e, ui, self.clip_C)))
            EE.append(E_mean)
        # EE.append(100)
        return EE