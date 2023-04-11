from random import choices
import numpy as np
from tqdm import tqdm, trange
import math
import random

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad, fill_grad
from flearn.utils.utils import clip, sparsify, transform, topindex, binarySearch
from flearn.utils.privacy_utils import clip_randomizer_clipLap, E_noisy_grad
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
        # self.m_p = int(self.clients_per_round / self.mp_rate)
        # print("Setting the padding size for each dimension with ", self.m_p)
        self.em_s = self.clients_per_round /self.rate
        self.topk = int( (self.dim_model + self.dim_y)/self.rate)
        print("Topk selecting {} dimensions".format(self.topk))
        self.choice_list = []
        # self.noisy_grads = self.E_noisy_gradients()

    def train(self):
        '''Train using Federated Proximal'''
        self.train_grouping()

    def local_process(self, flattened, eps):
        eps_lk = eps / self.topk
        vector = clip_randomizer_clipLap(flattened, self.clip_C, eps_lk, self.delta, self.mechanism)
        # post top-k
        c = topindex(flattened, self.topk)
        if self.ps == 'topk':
            choices = topindex(vector, self.topk)
        else:
            choices = np.array(random.sample(range(len(vector)), self.topk))
        re = np.array([0.0] * vector.shape[0])
        re = clip_randomizer_clipLap(re, self.clip_C, eps_lk, self.delta, self.mechanism)
        np.put(re, choices.tolist(), vector[[choices]].tolist())
        return re


    def server_process(self, messages, epss):
        '''
        basic aggregate, scale with rate when Top-k is applied (when rate > 1)
        '''
        avg = self.aggregate_p(messages)
        if self.de_bias_end == 'server':
            avg = self.de_bias_server(avg, epss)
        return avg
    
    def E_noisy_gradients(self, epss):
        e = np.array(epss)
        u = np.linspace(-self.clip_C, self.clip_C, 1000)
        EE = []
        for ui in u:
            E_mean = np.mean(np.array(E_noisy_grad(e, ui, self.clip_C)))
            EE.append(E_mean)
        return EE
        
    
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