import numpy as np
from tqdm import tqdm, trange
import math

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad, fill_grad
from flearn.utils.utils import clip, sparsify, transform, binarySearch
from flearn.utils.privacy_utils import clip_randomizer_clipLap, E_noisy_grad
from flearn.optimizers.pgd import PerturbedGradientDescent


class Server(BaseFedarated):
    '''
    traditional LDP-FL
    1. sampling depends on self.rate
    2. gaussian distribution for (epsilon, delta_lk)-LDP(DP) privacy
    '''
    def __init__(self, params, learner, dataset):
        print('Using Federated prox to Train (personalized-eps-FL)')
        self.inner_opt = PerturbedGradientDescent(learning_rate=params['learning_rate'], mu=params['mu'])
        super(Server, self).__init__(params, learner, dataset)
        self.clip_C = self.norm
        

    def train(self):
        '''Train using Federated Proximal'''
        self.train_grouping()

    def local_process(self, flattened, eps):
        vector = clip_randomizer_clipLap(flattened, self.clip_C, eps, self.delta, self.mechanism)
        return vector
    
    def E_noisy_gradients(self, epss):
        e = np.array(epss)
        # if self.eps_dist == 'uniform':
        #     e = np.linspace(self.per_left, self.per_right, 1000)
        u = np.linspace(-self.clip_C, self.clip_C, 1000)
        EE = []
        for ui in u:
            E_mean = np.mean(np.array(E_noisy_grad(e, ui, self.clip_C)))
            EE.append(E_mean)
        # EE.append(100)
        return EE
        

    def server_process(self, messages, epss):
        '''
        basic aggregate, scale with rate when Top-k is applied (when rate > 1)
        '''
        total_weight, base = self.aggregate_e(messages)
        # calibrate
        rate = 1
        avg = self.average(total_weight * rate, base)
        # avg = self.average(total_weight/self.rate, base)
        if self.de_bias_end == 'server':
            avg = self.de_bias_server(avg, epss)
        return avg
    
    def de_bias(self, noisy_grads, v):
        index = binarySearch(noisy_grads, 0, 999, v)
        u = np.linspace(-self.clip_C, self.clip_C, 1000)
        grad = u[index-1]
        return grad

    def de_bias_server(self, vector, epss):
        noisy_grads = self.E_noisy_gradients(epss)
        
        flattened = process_grad(vector)
        for i, v in enumerate(flattened):
            flattened[i] = self.de_bias(noisy_grads, v)
        re = fill_grad(flattened, vector)
        return re
    
