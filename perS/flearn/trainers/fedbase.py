import numpy as np
import math
import tensorflow as tf
from tqdm import trange, tqdm

from flearn.models.client import Client
from flearn.utils.model_utils import Metrics

from flearn.utils.tf_utils import process_grad, fill_grad
from flearn.utils.utils import transform
import time
import random
from flearn.utils.privacy_utils import clip_laplace

class BaseFedarated(object):
    def __init__(self, params, learner, data):
        for key, val in params.items():
            setattr(self, key, val)

        # create worker nodes
        tf.reset_default_graph()
        self.client_model = learner(
            *params['model_params'], self.inner_opt, self.seed)
        self.clients = self.setup_clients(data, self.dataset, self.model,
                                          self.client_model)
        print('{} Clients in Total'.format(len(self.clients)))
        self.latest_model = self.client_model.get_params() # global model params

        self.dim_model, self.dim_x, self.dim_y = self.setup_dim(
            self.dataset, self.model)
        users, _, train_data, _ = data
        self.num_samples = math.floor(len(train_data[users[0]]['x']) // params['batch_size']) * params['batch_size']

        # initialize system metrics
        self.metrics = Metrics(self.clients, params)
        self.test_acc_list = []
        self.latest_gradient_flattened = process_grad(self.latest_model)
        

    def __del__(self):
        # self.client_model.close()
        pass

    ##################################SET UP####################################
    def setup_dim(self, dataset_name, model_name):
        if model_name == 'mclr':
            if dataset_name == 'adult':
                return 104*2, 104, 2
            elif dataset_name == 'mnist' or dataset_name == 'qmnist':
                return 784*10, 784, 10
            elif dataset_name == 'cifar10':
                return 3072*10, 3072, 10
        elif model_name == 'nn':
            if dataset_name == 'mnist':
                return 784*512+512*512+512*10, 784, 10
        elif model_name == 'cnn':
            if dataset_name == 'mnist'  or dataset_name == 'qmnist':
                return 5*5*4+4*4*4*10, 784, 10
            if dataset_name == 'cifar10':
                return 5*5*8 + 7*7*8*10, 3072, 10
        else:
            raise "Unknown dataset and model"

    def setup_clients(self, dataset, dataset_name, model_name, model=None):
        '''instantiates clients based on given train and test data directories

        Return:
            list of Clients
        '''

        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]
        # all_clients = [Client(id=u, group=g, dataset_name=dataset_name, model_name=model_name,  # noqa: E501
        #                       train_data=train_data[u], eval_data=test_data[u], model=model) for u, g in zip(users, groups)]  # noqa: E501
        all_clients = []
        # set epsilon for all the clients
        for i in range(len(users)):
            u = users[i]
            g = groups[i]
            e = self.set_epsilon(self.epsilon, 1, u)
            all_clients.append(Client(id=u, group=g, epsilon=e, dataset_name=dataset_name, model_name=model_name,  # noqa: E501
                              train_data=train_data[u], eval_data=test_data[u], model=model))
        
        return all_clients


# The .3f is to round to 3 decimal places.
    #################################TRAINING#################################
    def train_grouping(self):
        count_iter = 0
        start = time.perf_counter()
        for i_round in range(self.num_rounds):
            # loop through mini-batches of clients
            # sample users
            # if self.is_sample_clients == 'on':
            #     round_selected_clients = random.sample(self.clients, self.up_clients)
            #     iter_selected_clients_num = min(self.clients_per_round,self.up_clients)
            # else:
            round_selected_clients = self.clients
            iter_selected_clients_num = self.clients_per_round

            for iter in range(0, len(round_selected_clients), iter_selected_clients_num):
                if count_iter % self.eval_every == 0:
                    self.evaluate(count_iter)
                    elapsed = time.perf_counter() - start
                    print('%.3f seconds.' % elapsed)
                # selected_clients = self.clients[iter: iter + self.clients_per_round]
                selected_clients = round_selected_clients[iter: iter + iter_selected_clients_num]
                csolns = []
                epss = []
                ########################## local updating ##############################
                for client_id, c in enumerate(selected_clients):
                    # distribute global model
                    c.set_params(self.latest_model)
                    # local iteration on full local batch of client c
                    soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size, user_sampler=self.user_sampler) # local params
                    epss.extend(c.epsilon)
                    # local update
                    model_updates = [u - v for (u, v) in zip(soln[1], self.latest_model)]
                    # aggregate local update
                    csolns.append(model_updates)
                elapsed = time.perf_counter() - start
                print('local updating %.3f seconds.' % elapsed)

                ######################### local process #########################                                
                csolns_new=[]
                for i in range(len(csolns)):
                    csoln = csolns[i]
                    eps = epss[i]
                    flattened = process_grad(csoln) # 2-d list->1-d numpy
                    tmp = []
                    processed_update = self.local_process(flattened, eps) # clip & add noises
                    tmp = fill_grad(processed_update, self.latest_model)
                    csolns_new.append(tmp) #TODO check!!!! csolns != csolns_new
                elapsed = time.perf_counter() - start
                print('local process %.3f seconds.' % elapsed)

                self.latest_model = [u + v for (u, v) in zip(self.latest_model, self.server_process(csolns_new, epss))] # aggregate
                self.client_model.set_params(self.latest_model) # set global model
                count_iter += 1
                elapsed = time.perf_counter() - start
                print('global aggregate %.3f seconds.' % elapsed)

        # final test model
        self.evaluate(count_iter)
        elapsed = time.perf_counter() - start
        print('%.3f seconds.' % elapsed)
        print('final test accuracy:',self.test_acc_list)


    #################################EVALUATING###############################
    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []

        for c in self.clients:
            ct, cl, ns = c.train_error_and_loss()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]

        return ids, groups, num_samples, tot_correct, losses


    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        return ids, groups, num_samples, tot_correct

    def evaluate(self, i):
        stats = self.test()
        stats_train = self.train_error_and_loss()
        train_loss = np.dot(stats_train[4], stats_train[2])*1.0/np.sum(stats_train[2])
        train_acc = np.sum(stats_train[3])*1.0/np.sum(stats_train[2])
        test_acc = np.sum(stats[3])*1.0/np.sum(stats[2])
        tqdm.write('------------------')
        tqdm.write('At round {} training loss: {}'.format(i, train_loss))
        tqdm.write('At round {} training accuracy: {}'.format(i, train_acc))
        tqdm.write('At round {} testing accuracy: {}'.format(i, test_acc))
        self.metrics.accuracies.append(test_acc)
        self.metrics.train_accuracies.append(train_acc)
        self.metrics.train_losses.append(train_loss)
        # self.metrics.write()
        self.test_acc_list.append(test_acc)

    #################################LOCAL PROCESS##################################
    def local_process(self, flattened, eps=[]):
        '''
        DO NOTHING
        1. non-private
        2. no clipping
        3. no sparsification
        (for npsgd)
        '''
        return flattened

    # def set_epsilon(self, epsilon, s_n, u):
    #     eps = [epsilon]*s_n
    #     return eps
    def set_epsilon(self, epsilon, s_n, u):
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
            eps = np.maximum(np.array(self.per_left), eps)
            eps = np.minimum(np.array(self.per_right), eps)
            eps = eps.tolist()
        elif self.eps_dist == 'mixnormal':
            if np.random.random()>=0.9:
                eps = np.random.normal(self.per_right, self.per_sigma, eps_num)
            else:
                eps = np.random.normal(self.per_mean, self.per_sigma, eps_num)
            eps = np.maximum(eps, self.per_left)
            eps = np.minimum(eps, self.per_right)
            eps = eps.tolist()
        return eps
    #################################AVERAGE/AGGREGATE##############################
    def server_process(self, messages, epss):
        '''
        ONLY AGGREGATE
        weighted or evenly-weighted by num_samples
        '''
        if len(messages) == 1:
            total_weight, base = self.aggregate_e(messages)
        else:
            total_weight, base = self.aggregate_w(messages)
        return self.average(total_weight, base)
    
    def average(self, total_weight, base):
        '''
        total_weight: # of aggregated updates
        base: sum of aggregated updates
        return the average update
        '''
        return [(v.astype(np.float16) / total_weight).astype(np.float16) for v in base]

    def average_cali(self, total_weight, base, clip):
        '''
        total_weight: # of aggregated updates
        base: sum of aggregated updates
        return the average update after transforming back from [0, 1] to [-C, C]
        '''
        return [transform((v.astype(np.float16) / total_weight), 0, 1, -self.clip_C, self.clip_C).astype(np.float16) for v in base]
    
    def aggregate_w(self, wsolns):
        total_weight = 0.0  
        base = [0] * len(wsolns[0][1])
        for w, soln in wsolns:
            total_weight += w
            for i, v in enumerate(soln):
                base[i] = base[i] + w * v.astype(np.float16)
        return total_weight, base

    def aggregate_e(self, solns):
        total_weight = 0.0
        base = [0] * len(solns[0]) # \sum{weight_i}
        for soln in solns: 
            total_weight += 1.0 # num of weights
            for i, v in enumerate(soln):
                base[i] = base[i] + v.astype(np.float16)
        return total_weight, base  
          
    def aggregate_p(self, solns):
        _, base = self.aggregate_e(solns)
        self.choice_list = []  # empty the choise list after each aggregation
        return [(v/self.em_s).astype(np.float16) for v in base]
