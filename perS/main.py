import numpy as np
import argparse
import importlib
import random
import os
import tensorflow as tf
from flearn.utils.model_utils import read_data
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

# PKGS: tensorflow 1.3

# GLOBAL PARAMETERS
OPTIMIZERS = ['noDpPgd', 'unifiedLdpPgd', 'personalLdpPgd', 'personalLdpPgdTopk_p']
DATASETS = ['mnist', 'qmnist']
MODEL = ['mclr', 'nn', 'cnn']
EPS_DIST = ['normal','uniform','single']
DEBIAS_END = ['server', 'no']

MODEL_PARAMS = {
    'mnist.mclr': (10,), # num_classes
    'mnist_cpsgd.mclr': (10,), # num_classes
    'cifar10.mclr': (10,), # num_classes
    'cifar10.cnn':(10,),
    'qmnist.mclr': (10,), # num_classes
    'qmnist_cpsgd.mclr': (10,), # num_classes
    
    'mnist.cnn': (10,), # num_classes
    'mnist_cpsgd.nn': (10,), # num_classes
}


def read_options():
    ''' Parse command line arguments or load defaults '''
    parser = argparse.ArgumentParser()
    # main setting
    parser.add_argument('--optimizer',
                        help='name of optimizer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default='noDpPgd')
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        choices=MODEL,
                        default='mclr')
    parser.add_argument('--epsilon',
                        help='eps for LDP',
                        type=float,
                        default=2)
    parser.add_argument('--mu',
                        help='mu for PGD',
                        type=float,
                        default=0.01)
    parser.add_argument('--norm',
                        help='L2 norm clipping threshold',
                        type=float,
                        default=0.1)
    
    
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        choices=DATASETS,
                        default='qmnist')
    
    # for personalized scene
    parser.add_argument('--de_bias_end',
                        help='debias on server/clients;',
                        type=str,
                        choices=DEBIAS_END,
                        default='server')
    parser.add_argument('--eps_dist',
                    help='sample epsilon_i from distribution;',
                    type=str,
                    choices=EPS_DIST,
                    default='uniform')
    parser.add_argument('--per_left',
                    help='sample epsilon_i from uni[left, right];',
                    type=float,
                    default=1)
    parser.add_argument('--per_right',
                    help='sample epsilon_i from uni[left, right];',
                    type=float,
                    default=1570)
    
    parser.add_argument('--per_mean',
                    help='sample epsilon_i from normal(mean,sigma);',
                    type=float,
                    default=1)
    parser.add_argument('--per_sigma',
                    help='sample epsilon_i from normal(mean,sigma);',
                    type=float,
                    default=0.1)
    


    
    # initialization global epoch, client batchs
    parser.add_argument('--num_rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=40) #
    parser.add_argument('--eval_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=1)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=10000)
    # for local update
    parser.add_argument('--batch_size',    # LOCAL: no greater than the local data size
                        help='batch size for local iteration (for sampling-based, denotes the number of local data that will be used throughout one epoch, for grouping-based, denotes the batch size for one/multiple local iterations for one updating);',
                        type=int,
                        default=32)
                        # default=7)
    parser.add_argument('--num_epochs',    # LOCAL: local epoch
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=10)
    # for global model
    parser.add_argument('--learning_rate',
                        help='learning rate for inner solver;',
                        type=float,
                        # default=0.15)
                        default=1)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    # for privacy
    parser.add_argument('--delta',
                        help='delta for DP, delta_lk for LDP(no SS-FL)',
                        type=float,
                        default=1e-11)
                        # default=0.00001)
    parser.add_argument('--mechanism',
                        help='type of local randomizer: gaussian, laplace, moue',
                        type=str,
                        default='laplace')

    # topk with post processing
    parser.add_argument('--rate',
                        help='compression rate, 1 for no compression',
                        type=int,
                        default=5)
    # topk or randomk
    parser.add_argument('--ps',
                        help='topk or randomk',
                        type=str,
                        default='topk')
    # for padding
    # parser.add_argument('--mp_rate',
    #                     help='under factor for mp=m/mp_rate',
    #                     type=float,
    #                     default=1)
    # for sample gradients in one client
    # parser.add_argument('--grad_rate',
    #                     help='sample s grads out of all gradients from users',
    #                     type=float,
    #                     default=1)
    
    # for padding sample in one clients
    # parser.add_argument('--up_grads',
    #                     help='sample s grads out of all gradients from users',
    #                     type=int,
    #                     default=800)
    # parser.add_argument('--is_sample_clients',
    #                     help='if sample c users out of all users, on for yes, off for no',
    #                     type=str,
    #                     default='off')
    # parser.add_argument('--up_clients',
    #                     help='sample c users out of all users',
    #                     type=int,
    #                     default=4000)
    parser.add_argument('--user_sampler',
                        help='sample rate in local user',
                        type=float,
                        default=1)

    try: parsed = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))

    # Set seeds
    random.seed(1 + parsed['seed'])
    np.random.seed(12 + parsed['seed'])
    tf.set_random_seed(123 + parsed['seed'])

    # load selected model
    model_path = '%s.%s.%s.%s' % ('flearn', 'models', parsed['dataset'], parsed['model'])

    mod = importlib.import_module(model_path)
    learner = getattr(mod, 'Model')

    # load selected trainer
    opt_path = 'flearn.trainers.%s' % parsed['optimizer']
    mod = importlib.import_module(opt_path)
    optimizer = getattr(mod, 'Server')

    # add selected model parameter
    parsed['model_params'] = MODEL_PARAMS['.'.join(model_path.split('.')[2:])]

    # print and return
    maxLen = max([len(ii) for ii in parsed.keys()]);
    fmtString = '\t%' + str(maxLen) + 's : %s';
    print('Arguments:')
    for keyPair in sorted(parsed.items()): print(fmtString % keyPair)

    return parsed, learner, optimizer

def main():
    # suppress tf warnings
    tf.logging.set_verbosity(tf.logging.WARN)
    # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
    
    # parse command line arguments
    options, learner, optimizer = read_options()

    # read data
    path = "/".join(os.path.abspath(__file__).split('/')[:-1])
    # log_path = os.path.join(os.path.abspath('.'), 'out_new', options['dataset']) 
    # if not os.path.exists(log_path):
    #     os.makedirs(log_path)
    dataset = options['dataset']
    train_dir_path = 'data/'+ dataset + '/train1w'
    test_dir_path = 'data/'+ dataset + '/test1w'
    train_path = os.path.join(path, train_dir_path)
    test_path = os.path.join(path, test_dir_path)
    dataset = read_data(train_path, test_path)

    # call trainer
    t = optimizer(options, learner, dataset)
    t.train()
    
if __name__ == '__main__':
    main()
    print('=========END=========')