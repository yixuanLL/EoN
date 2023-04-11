import numpy as np
import tensorflow as tf
from tqdm import trange
import random
from flearn.utils.model_utils import batch_data
from flearn.utils.tf_utils import graph_size
from flearn.utils.tf_utils import process_grad


class Model(object):
    '''
    Assumes that images are 28px by 28px
    '''

    def __init__(self, num_classes, optimizer, seed=1):

        # params
        self.num_classes = num_classes

        # create computation graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123 + seed)
            self.features, self.labels, self.train_op, self.grads, self.eval_metric_ops, self.loss = self.create_model(
                optimizer)
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph)

        # find memory footprint and compute cost of the model
        self.size = graph_size(self.graph)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            # self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops
            self.flops=0

    def create_model(self, optimizer):
        """Model function for Nerual Network."""
        image_size = 32
        channel = 3
        features = tf.placeholder(tf.float32, shape=[None,image_size*image_size*channel], name='features')
        labels = tf.placeholder(tf.int64, shape=[None, ], name='labels')
        
        input_layer=tf.reshape(features, [-1, image_size, image_size, channel])
        # input_layer=tf.transpose(img, (0,2,3,1))
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[3,3],
            strides=1,
            padding="same",
            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)
        # pool1_flat = tf.reshape(pool, [-1, 16*16*32])
        
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=16,
            kernel_size=[3,3],
            strides=1,
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)
        pool2_flat = tf.reshape(pool2, [-1, 8*8*16])

        # conv3 = tf.layers.conv2d(
        #     inputs=pool2,
        #     filters=256,
        #     kernel_size=[3,3],
        #     strides=1,
        #     padding="same",
        #     activation=tf.nn.relu)
        # # pool2 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)
        # conv4 = tf.layers.conv2d(
        #     inputs=conv3,
        #     filters=10,
        #     kernel_size=[3,3],
        #     strides=1,
        #     padding="same",
        #     activation=tf.nn.relu)
        # pool4 = tf.layers.max_pooling2d(inputs=pool2, pool_size=[2,2], strides=2)
        # pool4_flat = tf.reshape(pool4, [-1, 4*4*10])
        
        dense2 = tf.layers.dense(inputs=pool2_flat, units=32, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=dense2, units=self.num_classes,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
        # logits = tf.layers.dense(inputs=pool_flat, units=self.num_classes,activation=tf.nn.relu,
        #                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        grads_and_vars = optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
        return features, labels, train_op, grads, eval_metric_ops, loss
    
    
    # def create_model(self, optimizer):
    #     """Model function for Nerual Network."""
    #     features = tf.placeholder(tf.float32, shape=[None,784], name='features')
    #     labels = tf.placeholder(tf.int64, shape=[None, ], name='labels')
        
    #     input_layer=tf.reshape(features, [-1, 28, 28, 1])
    #     conv1 = tf.layers.conv2d(
    #     inputs=input_layer,
    #     filters=16,
    #     kernel_size=[8,8],
    #     padding="same",
    #     strides=2,
    #     activation=tf.nn.tanh)
    #     pool = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)
    #     # pool_flat = tf.reshape(pool, [-1, 7*7*16])
    #     conv2 = tf.layers.conv2d(
    #     inputs=pool,
    #     filters=32,
    #     kernel_size=[4,4],
    #     padding="same",
    #     strides=2,
    #     activation=tf.nn.tanh)
    #     pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)
    #     pool2_flat = tf.reshape(pool2, [-1, 2*2*32])
    #     dense2 = tf.layers.dense(inputs=pool2_flat, units=32, activation=tf.nn.tanh)
    #     logits = tf.layers.dense(inputs=dense2, units=self.num_classes,
    #                              kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
    #     # logits = tf.layers.dense(inputs=pool_flat, units=self.num_classes,activation=tf.nn.relu,
    #     #                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
    #     predictions = {
    #         "classes": tf.argmax(input=logits, axis=1),
    #         "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    #     }
    #     loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    #     grads_and_vars = optimizer.compute_gradients(loss)
    #     grads, _ = zip(*grads_and_vars)
    #     train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
    #     eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
    #     return features, labels, train_op, grads, eval_metric_ops, loss

    def set_params(self, model_params=None):
        if model_params is not None:
            with self.graph.as_default():
                all_vars = tf.trainable_variables()
                for variable, value in zip(all_vars, model_params):
                    variable.load(value, self.sess)

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params

    def get_gradients(self, data, model_len):

        grads = np.zeros(model_len)
        num_samples = len(data['y'])

        with self.graph.as_default():
            model_grads = self.sess.run(self.grads,
                                        feed_dict={self.features: data['x'], self.labels: data['y']})
            grads = process_grad(model_grads)

        return num_samples, grads

    def solve_inner(self, data, num_epochs=1, batch_size=32, user_sampler=1):
        '''Solves local optimization problem'''
        # for sample
        if user_sampler < 1:

            data_new = {}
            data_l = data['y'].shape[0]
            sample_num = int(user_sampler * data_l)


            sample_id = random.sample(list(range(data_l)), sample_num)
            data_new['x'] = data['x'][[sample_id]]
            data_new['y'] = data['y'][[sample_id]]
            data = data_new
        '''Solves local optimization problem'''
        for _ in range(num_epochs):  #for _ in trange(num_epochs, desc='Epoch: ', leave=False, ncols=120):
            for X, y in batch_data(data, batch_size):
                with self.graph.as_default():
                    self.sess.run(self.train_op,
                                  feed_dict={self.features: X, self.labels: y})
        soln = self.get_params()
        comp = num_epochs * (len(data['y']) // batch_size) * batch_size * self.flops
        return soln, comp
    
    def solve_inner_silo(self, data, pad_mod, pad_sample, up_grads):
        '''Solves local optimization problem and subsample gradients inside the client'''
        # original_model = self.get_params()
        solns = []
        data_new = {}
        data_l = data['y'].shape[0]

        # for padding sample
        if pad_mod == 'on':
            if(pad_sample < data_l):
                print('[ERROR]: pad sample < clinet datasets, please choose a larger pad sample! pad_sample:{}, data_l:{}',format(pad_sample, data_l))
            traindata_l_sample = random.sample(list(range(pad_sample)), up_grads)
            traindata_l_sample_in = []
            for i in traindata_l_sample:
                if i >= data_l:
                    continue
                traindata_l_sample_in.append(i)
            if len(traindata_l_sample_in)==0:
                return solns, 0
            data_new['x'] = data['x'][[traindata_l_sample_in]]
            data_new['y'] = data['y'][[traindata_l_sample_in]]
            X = data_new['x']
            y = data_new['y']
        # not padding
        else:
            data_new = data
            X = data_new['x']
            y = data_new['y']
        # train model
        for _ in range(10):  
            for X, y in batch_data(data_new, 32):          
                with self.graph.as_default():
                    self.sess.run(self.train_op,
                                    feed_dict={self.features: X, self.labels: y})
        original_model = self.get_params()
        
        # generate grad
        for X, y in batch_data(data_new, 1):
            with self.graph.as_default():
                self.sess.run(self.train_op,
                                feed_dict={self.features: X, self.labels: y})
            solns.append(self.get_params())
            self.set_params(original_model)

        comp = 0 # TODO
        return solns, comp

    def test(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''
        with self.graph.as_default():
            tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss],
                                              feed_dict={self.features: data['x'], self.labels: data['y']})
        return tot_correct, loss

    def close(self):
        self.sess.close()