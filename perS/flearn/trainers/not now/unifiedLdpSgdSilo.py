import numpy as np
from tqdm import tqdm, trange
import math

from .fedbase_silo import BaseFedarated
from flearn.utils.tf_utils import process_grad
from flearn.utils.utils import clip, sparsify, topindex, transform, binarySearch
from flearn.utils.privacy_utils import clip_randomizer_clipLap, sampling_randomizer, clip_randomizer, E_noisy_grad
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
        print('Using Federated prox to Train (unified eps LDP-FL)')
        # self.inner_opt = GradientDescentOptimizer(learning_rate=params['learning_rate'])
        # self.inner_opt = AdamOptimizer(learning_rate=params['learning_rate'])
        # print('Adam')
        self.inner_opt = PerturbedGradientDescent(learning_rate=params['learning_rate'], mu=params['mu'])
        print('PGD')
        super(Server, self).__init__(params, learner, dataset)
        self.clip_C = self.norm
        # self.topk = int( (self.dim_model + self.dim_y)/self.rate)
        # self.eps_ld = self.epsilon
        # self.eps_ld = self.epsilon / self.sample
        # print("Topk selecting {} dimensions".format(self.topk))
        # self.choice_list = []

    def train(self):
        '''Train using Federated Proximal'''
        self.train_grouping()

    def local_process(self, flattened, eps):
        # choices = topindex(flattened, self.topk)
        # self.choice_list.extend(choices)
        # return sampling_randomizer(flattened, choices, self.clip_C, self.eps_ld, self.delta, self.mechanism)       
        # TODO clip and add noises
        return clip_randomizer(flattened, self.clip_C, eps, self.delta, self.mechanism)
        # return clip_randomizer_clipLap(flattened, self.clip_C, self.epsilon, self.delta, self.mechanism)

    def server_process(self, messages, epss):
        '''
        basic aggregate, scale with rate when Top-k is applied (when rate > 1)
        '''
        total_weight, base = self.aggregate_e(messages)
        return self.average(total_weight, base) # without scale
    
    def set_epsilon(self, epsilon, s_n, u):
        if self.pad_mod == 'on':
            eps_num = self.pad_sample
        else:
            eps_num = s_n
        # uniform distribution eps
        if self.eps_dist == 'single':
            eps = [epsilon] * eps_num
        elif self.eps_dist == 'uniform':
            left = self.per_left
            right = self.per_right
            eps = np.random.uniform(left , right, eps_num)
            eps = eps.tolist()
        # normal distribution eps
        elif self.eps_dist == 'normal':
            eps = np.random.normal(self.per_mean, self.per_sigma, eps_num)
            eps = np.maximum(np.array(0.01), eps)
            eps = eps.tolist()
        elif self.eps_dist == 'step':
            if int(u.split('_')[1]) % 100 < 90:
                eps = [self.per_left] * eps_num
            else:
                eps = [self.per_right] * eps_num
        elif self.eps_dist == 'uni_thre':
            left = self.per_left
            right = self.per_right
            eps = np.random.uniform(left , right, eps_num)
            for i in range(len(eps)):
                if eps[i] > self.eps_thre:
                    eps[i] = self.eps_thre
        return eps