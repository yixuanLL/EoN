import numpy as np
from tqdm import tqdm, trange
import math

from .fedbase_silo_nid import BaseFedarated
from flearn.utils.tf_utils import process_grad
from flearn.utils.utils import clip, sparsify, topindex, transform
from flearn.utils.privacy_utils import clip_randomizer_clipLap, sampling_randomizer, clip_randomizer
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
        self.inner_opt = PerturbedGradientDescent(learning_rate=params['learning_rate'], mu=params['mu'])
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
        total_weight, base = self.aggregate_w_nid(messages, epss)
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
            left = self.per_left
            eps = np.random.normal(self.per_mean, self.per_sigma, eps_num)
            eps = eps.tolist()
        elif self.eps_dist == 'step':
            if int(u.split('_')[1]) % 5 < 2:
                eps = [self.per_left] * eps_num
            else:
                eps = [self.per_left] * eps_num
        return eps