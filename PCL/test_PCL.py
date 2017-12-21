import tensorflow as tf
import numpy as np
import random
import os
import pickle

import controller
import model
import policy
import baseline
import objective
import optimizers
import replay_buffer
import expert_paths
import gym_wrapper
import env_spec


batch_size = 1 # FLAGS.batch_size
replay_batch_size = 25 # FLAGS.replay_batch_size
num_samples = 1 # FLAGS.num_samples # number of samples from each random seed initialization
env_str = 'HalfCheetah-v1' # FLAGS.env
env = gym_wrapper.GymWrapper(env_str, distinct=1 // 1, count=1)
env_spec = env_spec.EnvSpec(env.get_one())
max_step = 100 # FLAGS.max_step
cutoff_agent = 1000 # FLAGS.cutoff_agent
num_steps = 100000 # FLAGS.num_steps
validation_frequency = 50 # FLAGS.validation_frequency
target_network_lag = 0.99 # FLAGS.target_network_lag
sample_from = 'target' # FLAGS.sample_from
critic_weight = 0.0 # FLAGS.critic_weight
objective = 'pcl' # FLAGS.objective
trust_region_p = False # FLAGS.trust_region_p
value_opt = 'grad' # FLAGS.value_opt
max_divergence = 0.001 # FLAGS.max_divergence
learning_rate = 0.002 # FLAGS.learning_rate
clip_norm = 40 # FLAGS.clip_norm
clip_adv = 1.0 # FLAGS.clip_adv
tau = 0.0 # FLAGS.tau
tau_decay = None # FLAGS.tau_decay # decay tau by this much every 100 steps
tau_start = 0.1 # FLAGS.tau_start
eps_lambda = 0.0 # FLAGS.eps_lambda # relative entropy regularizer
update_eps_lambda = True # FLAGS.update_eps_lambda
gamma = 0.995 # FLAGS.gamma
rollout = 10 # FLAGS.rollout
fixed_std = True # FLAGS.fixed_std # fix the std in Gaussian distributions
input_prev_actions = True # FLAGS.input_prev_actions # input previous actions to policy network
recurrent = False # FLAGS.recurrent
input_time_step = False # FLAGS.input_time_step # input time step into value calucations
use_online_batch = False # FLAGS.use_online_batch
batch_by_steps = True # FLAGS.batch_by_steps
unify_episodes = True # FLAGS.unify_episodes
replay_buffer_size = 20000 # FLAGS.replay_buffer_size
replay_buffer_alpha = 0.1 # FLAGS.replay_buffer_alpha
replay_buffer_freq = 1 # FLAGS.replay_buffer_freq
eviction = 'fifo' # FLAGS.eviction
prioritize_by = 'step' # FLAGS.prioritize_by
num_expert_paths = 0 # FLAGS.num_expert_paths
internal_dim = 64 # FLAGS.internal_dim
value_hidden_layers = 2 # FLAGS.value_hidden_layers
tf_seed = 42 # FLAGS.tf_seed # random seed for tensorflow
save_trajectories_dir = None # FLAGS.save_trajectories_dir # directory to save trajectories to, if desired
load_trajectories_dir = None # FLAGS.load_trajectories_dir # file to load expert trajectories from
# hparams # All hyperparameters


hparams = dict((attr, getattr(self, attr))
                        for attr in dir(self)
                        if not attr.startswith('__') and
not callable(getattr(self, attr)))
