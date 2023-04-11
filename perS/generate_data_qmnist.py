from typing import Counter
import numpy as np
import random
import json
import os
from tqdm import trange
from os.path import join
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import pickle
import codecs
import gzip
import lzma

from collections import Counter as C

random.seed(7)

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

def open_maybe_compressed_file(path):
    if path.endswith('.gz'):
        return gzip.open(path, 'rb')
    elif path.endswith('.xz'):
        return lzma.open(path, 'rb')
    else:
        return open(path,'rb')
def read_idx2_int(path):
    with open_maybe_compressed_file(path) as f:
        data = f.read()
        assert get_int(data[:4]) == 12*256 + 2
        length = get_int(data[4:8])
        width = get_int(data[8:12])
        parsed = np.frombuffer(data, dtype=np.dtype('>i4'), offset=12)
        parsed = np.reshape(parsed.astype('i4'), (length,width))[:, 0]
        return parsed

def read_idx3_ubyte(path):
    with open_maybe_compressed_file(path) as f:
        data = f.read()
        assert get_int(data[:4]) == 8 * 256 + 3
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        parsed = np.reshape(parsed, (length, num_rows*num_cols))
        return parsed


# Setup directory for train/test data
root = "/".join(os.path.abspath(__file__).split('/')[:-1])
# root = './data/'

file_name = {
    'train' : ['qmnist-train-images-idx3-ubyte.gz', 'qmnist-train-labels-idx2-int.gz'] ,
    'test' :  ['qmnist-test-images-idx3-ubyte.gz', 'qmnist-test-labels-idx2-int.gz']
}

train_file = root + '/data/qmnist/train1w/train.json'
test_file = root + '/data/qmnist/test1w/test.json'
dir_path = os.path.dirname(train_file)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_file)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Get MNIST data
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
origi_data_path = root + '/data/qmnist/'
x_train = read_idx3_ubyte(origi_data_path + file_name['train'][0])
y_train = read_idx2_int(origi_data_path + file_name['train'][1])
x_test = read_idx3_ubyte(origi_data_path + file_name['test'][0])
y_test = read_idx2_int(origi_data_path + file_name['test'][1])

x_train = np.append(x_train, x_test, axis=0)
y_train = np.append(y_train, y_test)
#Normalize
# x_train=tf.keras.utils.normalize(x_train,axis=1)
# x_test=tf.keras.utils.normalize(x_test,axis=1)
x_train=keras.utils.normalize(x_train,axis=1)

mnist_data = [[]*10 for row in range(10)]

# for idx in range(len(x_train)):
#     flattend_xi = np.array([])
#     for i in range(0, len(x_train[idx])):
#         flattend_xi = np.append(flattend_xi, x_train[idx][i])
#     mnist_data[y_train[idx]].append(flattend_xi)  
#     # mnist_data[y_train[idx]].append(x_train[idx])
# print([len(v) for v in mnist_data])
NUM_USERS = 10000
SAMPLES_PER_USER=12
NUM_CLASSES = 10
TRAINT_RATE = 0.9

sample_num = y_train.shape[0]
y_train = np.reshape(y_train, (sample_num,1))
qmnist_data = np.concatenate((y_train, x_train), axis=1)
# 按照首行排序
qmnist_data = qmnist_data[np.lexsort(qmnist_data[:,::-1].T)]
idx_dict = C(qmnist_data[:,0].reshape(sample_num,).tolist())

start = 0
x_train = np.delete(qmnist_data, 0, axis=1)
for i in range(NUM_CLASSES):
    num_class = idx_dict[i]
    mnist_data[i] = x_train[start:start+num_class, :]
    start += num_class



###### CREATE USER DATA SPLIT #######
print("Assign samples to each user")
X = [[] for _ in range(NUM_USERS)]
y = [[] for _ in range(NUM_USERS)]
idx = np.zeros(NUM_CLASSES, dtype=np.int64)

for user in range(NUM_USERS): # for each user
    for j in range(SAMPLES_PER_USER): # for sample 1~10
        l = (user+j) % NUM_CLASSES
        try:
            X[user].append(mnist_data[l][idx[l]].tolist())
        except:
            # print('error', l, idx[l])
            # exit(0)
            continue
        y[user].append(np.array(l).tolist())
        idx[l] += 1
    pass
print("idx=", idx)


print("Create data structure")
train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

print("Setup users")
for i in trange(NUM_USERS, ncols=120):
    uname = 'f_{0:05d}'.format(i)
    
    combined = list(zip(X[i], y[i]))
    random.shuffle(combined)
    X[i][:], y[i][:] = zip(*combined)
    num_samples = len(X[i])
    train_len = int(TRAINT_RATE*num_samples)
    test_len = num_samples - train_len
    
    train_data['users'].append(uname)
    train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
    train_data['num_samples'].append(train_len)
    test_data['users'].append(uname)
    test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
    test_data['num_samples'].append(test_len)

print("writing...")

with open(train_file, 'w') as outfile:
    json.dump(train_data, outfile)
with open(test_file, 'w') as outfile:
    json.dump(test_data, outfile)
