import numpy as np
from tqdm import tqdm, trange
import math

from .fedbase_silo import BaseFedarated
from flearn.utils.tf_utils import process_grad, fill_grad
from flearn.utils.utils import clip, sparsify, transform, binarySearch
from flearn.utils.privacy_utils import clip_randomizer_clipLap, E_noisy_grad, clip_randomizer, normalize_clip_randomizer_clipLap #, trim_dim_threshold
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
        print('Using Federated prox to Train (personalized-eps-FL) and sample users and gradients')
        # self.inner_opt = AdamOptimizer(learning_rate=params['learning_rate'])
        self.inner_opt = PerturbedGradientDescent(learning_rate=params['learning_rate'], mu=params['mu'])
        super(Server, self).__init__(params, learner, dataset)
        self.clip_C = self.norm
        # self.sample = int( (self.dim_model + self.dim_y)/self.rate)
        # self.eps_ld = self.epsilon / self.sample
        

    def train(self):
        '''Train using Federated Proximal'''
        self.train_grouping()

    def local_process(self, flattened, eps):
        # return clip_randomizer(flattened, self.clip_C, eps, self.delta, self.mechanism)
        vector = clip_randomizer_clipLap(flattened, self.clip_C, eps, self.delta, self.mechanism)
        ## try normal
        # vector = normalize_clip_randomizer_clipLap(flattened, self.clip_C, eps, self.delta, self.mechanism)
        return vector
    
    def set_epsilon(self, eps, s_n, u):
        if self.pad_mod == 'on':
            eps_num = self.pad_sample
        else:
            eps_num = s_n
        # uniform distribution eps
        if self.eps_dist == 'uniform':
            left = self.per_left
            right = self.per_right
            eps = np.random.uniform(left , right, eps_num)
            eps = eps.tolist()
        elif self.eps_dist == 'step':
            if int(u.split('_')[1]) % 100 < 90:
                eps = [self.per_left] * eps_num
            else:
                eps = [self.per_right] * eps_num
        # normal distribution eps
        elif self.eps_dist == 'normal':
            # left = self.per_left
            eps = np.random.normal(self.per_mean, self.per_sigma, eps_num)
            eps = np.maximum(np.array(0.01), eps)
            eps = eps.tolist()
        return eps
    
    
    def server_process(self, messages, epss):
        '''
        basic aggregate, scale with rate when Top-k is applied (when rate > 1)
        '''
        total_weight, base = self.aggregate_e(messages)
        # return self.average(total_weight/self.rate, base) # without scale
        # calibrate
        if self.pad_mod == 'on' and self.pad_sample > 1120:
            # rate = self.up_grads/self.pad_sample
            rate = 1120/self.pad_sample
        else:
            rate = 1
        avg = self.average(total_weight * rate, base)
        ## try normal
        # avg = self.average_cali(total_weight * rate, base)
        
        if self.de_bias_end == 'server':
            avg = self.de_bias_server(avg, epss)
        
        # to rescale
        # max_bound = 0.8
        # for k in range(len(avg)):
        #     avg[k] *= (max_bound/self.clip_C)
        return avg
    ## try normal
    # def average_cali(self, total_weight, base):
    #     '''
    #     return the average update after transforming back from [-1, 1] to [-C, C]
    #     '''
    #     return [transform((v.astype(np.float16) / total_weight), -1, 1, -self.clip_C, self.clip_C).astype(np.float16) for v in base]

    
    def de_bias(self, noisy_grads, v):
        # indexs = np.where(self.noisy_grads >= noisy_grad)
        index = binarySearch(noisy_grads, 0, 999, v)
        u = np.linspace(-self.clip_C, self.clip_C, 1000)
        # grad = u[index-2]
        grad = u[index-1]
        return grad

    def de_bias_server(self, vector, epss):
        noisy_grads = self.E_noisy_gradients(epss)
        
        flattened = process_grad(vector)
        for i, v in enumerate(flattened):
            flattened[i] = self.de_bias(noisy_grads, v)
        re = fill_grad(flattened, vector)
        # for l in range(len(vector)):
        #     if l==0:
        #         for i, vs in enumerate(vector[l]):
        #             for k, v in enumerate(vs):
        #                 vector[l][i][k] = self.de_bias(noisy_grads, v)
        #     else:
        #         for j, v in enumerate(vector[l]):
        #             vector[l][j] = self.de_bias(noisy_grads, v)
        return re
    
    def E_noisy_gradients(self, epss):
        e = np.array(epss)
        # if self.eps_dist == 'uniform':
            # e = np.linspace(self.per_left*(self.grad_rate*self.mp_rate), self.per_right*(self.grad_rate*self.mp_rate), 1000)
        u = np.linspace(-self.clip_C, self.clip_C, 1000)
        EE = []
        for ui in u:
            E_mean = np.mean(np.array(E_noisy_grad(e, ui, self.clip_C)))
            EE.append(E_mean)
        # EE.append(100)
        return EE