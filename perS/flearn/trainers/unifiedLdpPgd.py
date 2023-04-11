import numpy as np
from tqdm import tqdm, trange
import math

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad
from flearn.utils.utils import clip, sparsify, topindex, transform
from flearn.utils.privacy_utils import clip_randomizer
from flearn.optimizers.pgd import PerturbedGradientDescent

class Server(BaseFedarated):
    '''
    traditional LDP-FL
    1. sampling depends on self.rate
    2. gaussian distribution for (epsilon, delta_lk)-LDP(DP) privacy
    '''
    def __init__(self, params, learner, dataset):
        print('Using Federated prox to Train (unified eps LDP-FL)')
        self.inner_opt = PerturbedGradientDescent(learning_rate=params['learning_rate'], mu=params['mu'])
        super(Server, self).__init__(params, learner, dataset)
        self.clip_C = self.norm

    def train(self):
        '''Train using Federated Proximal'''
        self.train_grouping()

    def local_process(self, flattened, eps):
        return clip_randomizer(flattened, self.clip_C, eps, self.delta, self.mechanism)

    def server_process(self, messages, epss):
        '''
        basic aggregate, scale with rate when Top-k is applied (when rate > 1)
        '''
        total_weight, base = self.aggregate_e(messages)
        return self.average(total_weight, base) # without scale
    