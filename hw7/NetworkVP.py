# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import re
import numpy as np
import tensorflow as tf

from Config import Config

class NetworkVP:
    def __init__(self, device, model_name, action_shape, obs_shape):
        self.device = device
        self.model_name = model_name
        self.action_shape = action_shape
        self.obs_shape = obs_shape

        self.learning_rate = Config.LEARNING_RATE_START
        self.beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with tf.device(self.device):
                self._create_graph()

                self.sess = tf.Session(
                    graph=self.graph,
                    config=tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.GPUOptions(allow_growth=True)))
                self.sess.run(tf.global_variables_initializer())

                if Config.TENSORBOARD: self._create_tensor_board()
                if Config.LOAD_CHECKPOINT or Config.SAVE_MODELS:
                    vars = tf.global_variables()
                    self.saver = tf.train.Saver({var.name: var for var in vars}, max_to_keep=0)
                
    def _create_graph(self):
        self.obs = tf.placeholder(
            tf.float32, [None] + list(self.obs_shape), name='observation')
        self.advantages = tf.placeholder(tf.float32, [None], name='advantages')
        self.v_targets = tf.placeholder(tf.float32, [None], name='value_targets')
        self.actions = tf.placeholder(tf.float32, [None] + list(self.action_shape), name='actions')
        self.old_means = tf.placeholder(tf.float32, [None] + list(self.action_shape), name='old_action_means')
        self.old_log_stds = tf.placeholder(tf.float32, [None] + list(self.action_shape), name='old_action_log_stds')

        self.var_beta = tf.placeholder(tf.float32, name='beta', shape=[])
        self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])
        self.global_step = tf.Variable(0, trainable=False, name='step')
      
        # std params
        self.log_std_var = tf.get_variable('log_std_var', shape=(self.action_shape[0],), 
            dtype=tf.float32, initializer=tf.constant_initializer(np.log(1.0)))
        self.log_stds = self.param_layer(self.log_std_var, self.obs)
        self.log_stds = tf.maximum(self.log_stds, np.log(1e-6))   # min-std to prevent numerical issues
        
        # mean network "YOUR CODE HERE"
        self.means = 

        # value network "YOUR CODE HERE"
        self.values = 
        
        self.logli = self.loglikelihood(self.actions, self.means, self.log_stds)
        self.ent = self.entropy(self.log_stds)

        # policy gradient loss
        self.p_loss = - tf.reduce_mean(self.logli * self.advantages, axis=0)
        self.p_loss -= self.var_beta * tf.reduce_mean(self.ent)  # entropy regularization
        
        # value loss
        self.v_loss = tf.reduce_mean(tf.square(self.values - self.v_targets))

        self.opt = tf.train.AdamOptimizer(learning_rate=self.var_learning_rate)
        self.train_op_p = self.opt.minimize(self.p_loss, global_step=self.global_step)
        self.train_op_v = self.opt.minimize(self.v_loss, global_step=self.global_step)
   
    def loglikelihood(self, actions, means, log_stds):
      # "YOUR CODE HERE"
      pass
    
    def entropy(self, log_stds):
      # "YOUR CODE HERE"
      pass
      
    def kl(self, old_means, old_log_stds, new_means, new_log_stds):
      # "YOUR CODE HERE"
      pass
    
    def _create_tensor_board(self):
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries.append(tf.summary.scalar("Policy Loss", self.p_loss))
        #summaries.append(tf.summary.scalar("LearningRate", self.var_learning_rate))
        #summaries.append(tf.summary.scalar("Beta", self.var_beta))
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram("weights_%s" % var.name, var))

        summaries.append(tf.summary.histogram("activation_h1", self.h1))
        summaries.append(tf.summary.histogram("activation_h2", self.h2)) 
        summaries.append(tf.summary.histogram("advantages", self.advantages))
        summaries.append(tf.summary.histogram("log_likelihood", self.logli))
        summaries.append(tf.summary.histogram("entropy", self.ent))
        summaries.append(tf.summary.histogram("policy_means", self.means))
        summaries.append(tf.summary.histogram("policy_log_stds", self.log_stds))
        summaries.append(tf.summary.histogram("observations", self.obs))
        
        self.summary_op = tf.summary.merge(summaries)
        self.log_writer = tf.summary.FileWriter("logs/%s" % self.model_name, self.sess.graph)

    def param_layer(self, param, input):
        ndim = input.get_shape().ndims
        reshaped_param = tf.reshape(param, (1,) * (ndim - 1) + tuple(param.get_shape().as_list()))
        tile_arg = tf.concat([tf.shape(input)[:ndim - 1], [1]], 0)
        tiled = tf.tile(reshaped_param, tile_arg)
        return tiled

    def dense_layer(self, input, out_dim, name, func=tf.nn.tanh):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w', dtype=tf.float32, shape=[in_dim, out_dim], initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.matmul(input, w) + b
            if func is not None:
                output = func(output)

        return output

    def conv2d_layer(self, input, filter_size, out_dim, name, strides, func=tf.nn.relu):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(filter_size * filter_size * in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w',
                                shape=[filter_size, filter_size, in_dim, out_dim],
                                dtype=tf.float32,
                                initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.nn.conv2d(input, w, strides=strides, padding='SAME') + b
            if func is not None:
                output = func(output)

        return output

    def __get_base_feed_dict(self):
        return {self.var_beta: self.beta, self.var_learning_rate: self.learning_rate}

    def get_global_step(self):
        step = self.sess.run(self.global_step)
        return step

    def predict_p_and_v(self, x):
        means, log_stds, values = self.sess.run([self.means, self.log_stds, self.values], feed_dict={self.obs: x})
        return means, log_stds, values.reshape(-1)
    
    def train(self, x, y_r, adv, a, a_m, a_s, trainer_id):
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.obs: x, self.advantages:adv, self.actions: a, self.old_means: a_m, self.old_log_stds: a_s, self.v_targets: y_r})
        self.sess.run(self.train_op_p, feed_dict=feed_dict)
        self.sess.run(self.train_op_v, feed_dict=feed_dict)

    def log(self, x, y_r, adv, a, a_m, a_s):
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.obs: x, self.advantages:adv, self.actions: a})
        step, summary = self.sess.run([self.global_step, self.summary_op], feed_dict=feed_dict)
        self.log_writer.add_summary(summary, step)

    def _checkpoint_filename(self, episode):
        return 'checkpoints/%s_%08d' % (self.model_name, episode)
    
    def _get_episode_from_filename(self, filename):
        # TODO: hacky way of getting the episode. ideally episode should be stored as a TF variable
        return int(re.split('/|_|\.', filename)[2])

    def save(self, episode):
        self.saver.save(self.sess, self._checkpoint_filename(episode))

    def load(self):
        filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename(episode=0)))
        if Config.LOAD_EPISODE > 0:
            filename = self._checkpoint_filename(Config.LOAD_EPISODE)
        self.saver.restore(self.sess, filename)
        return self._get_episode_from_filename(filename)
       
    def get_variables_names(self):
        return [var.name for var in self.graph.get_collection('trainable_variables')]

    def get_variable_value(self, name):
        return self.sess.run(self.graph.get_tensor_by_name(name))
