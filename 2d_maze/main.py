import os

# set some env vars
os.environ.setdefault('JAX_PLATFORM_NAME', 'gpu')     # tell JAX to use GPU
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.2'  # don't use all gpu mem

import gym
import jax
import haiku as hk
from optax import adam
import jax.numpy as jnp
import numpy as np
import timeit
import time
import jax.experimental.optimizers as joptimizers
import matplotlib.pyplot as plt
import importlib
import sys
import os
import pickle

import environment
import utils
import graph
import RL
import supervised
from config import Configuration

env_n = Configuration.batch_size #256 * 8 
env_size = 32
epoch_len = Configuration.epoch_len #5000 // 2#10000 * 2
discount = 0.99
positive_sample_repetitions = 2 #20
data_dir = sys.argv[1]
os.system('mkdir data')
os.system(f'mkdir data/{data_dir}')
os.system(f'cp config.py data/{data_dir}/')

# INITIALIZE PARAMS AND ENV

key = jax.random.PRNGKey(np.random.randint(100000))
env = environment.Mazes2d(key, env_n, env_size - 2, env_size - 2)
keys = jax.random.split(key, 2)
s = env.init_state(keys[1])

keys = jax.random.split(keys[0], 4)
pi_init, pi_f = graph.get_policy(s, keys[1])
V_init,  V_f  = graph.get_value( s, keys[2])
R_init,  R_f  = graph.get_reward(s, keys[3])

V_params  = V_init(keys[1], s)
pi_params = pi_init(keys[2], s)
R_params  = R_init(keys[3], s)

# OPTIMIZERS

V_opt_init, V_opt_update, V_opt_get_params = joptimizers.adam(0.0001)
V_opt_state = V_opt_init(V_params)
V_opt_update = jax.jit(V_opt_update)

pi_opt_init, pi_opt_update, pi_opt_get_params = joptimizers.adam(Configuration.policy_lr)
pi_opt_state = pi_opt_init(pi_params)
pi_opt_update = jax.jit(pi_opt_update)

R_opt_init, R_opt_update, R_opt_get_params = joptimizers.adam(0.001)
R_opt_state = R_opt_init(R_params)
R_opt_update = jax.jit(R_opt_update)

jit_TD_learning  = jax.jit(RL.TD_learning, static_argnums=0)
jit_A2C_learning = jax.jit(RL.A2C_learning, static_argnums=0)


layers = list(pi_params.keys())
print(layers)


# MAIN TRAINING

neg_samples = []
neg_samples_epoch = []
guide_mask = jax.random.randint(keys[0], [env_n, 1], 0, 0) - 1

for epoch in range(70):
    print("Epoch: ", epoch)

    if Configuration.load_reward:
        with open(f'data/final_baseline_longer_terminalsamples_longer_150/{epoch}.params', 'rb') as f:
            R_params = pickle.load(f)

    pi_guide = hk.data_structures.to_mutable_dict(pi_params)
    pi_guide = hk.data_structures.to_immutable_dict(pi_guide)
    heatmap =  np.zeros((32, 32))#env.mazes[0] * 0
    keys = jax.random.split(keys[0], 4)
    s = env.init_state(keys[1]) * 0 + 1
    s_next = s
    
    #V_params  = V_init(keys[1], s)
    #pi_params = pi_init(keys[3], s)
    if Configuration.reset_pi_head:
        pi_params = hk.data_structures.to_mutable_dict(pi_params)
        pi_params[layers[-1]]['w'] = 0. * pi_params['linear_1']['w']
        pi_params[layers[-1]]['b'] = 0. * pi_params['linear_1']['b']
        pi_params = hk.data_structures.to_immutable_dict(pi_params)

    if Configuration.reset_policy:
        pi_params = pi_init(keys[2], s)
        pi_opt_init, pi_opt_update, pi_opt_get_params = joptimizers.adam(0.0001)
        pi_opt_state = pi_opt_init(pi_params)
        pi_opt_update = jax.jit(pi_opt_update)

    if Configuration.reset_value:
        V_params = V_init(keys[2], s)
        V_opt_init, V_opt_update, V_opt_get_params = joptimizers.adam(0.0001)
        V_opt_state = V_opt_init(V_params)
        V_opt_update = jax.jit(V_opt_update)

    #V_params = hk.data_structures.to_mutable_dict(V_params)
    #V_params['linear_2']['w']  = 1 * V_params['linear_2']['w']
    #V_params['linear_2']['b']  = 1 * V_params['linear_2']['b']
    #V_params = hk.data_structures.to_immutable_dict(V_params)
    
    pi_opt_state = pi_opt_init(pi_params)
    V_opt_state = V_opt_init(V_params)
    
    
    R_mat = utils.visualize_R(R_f, R_params)
    R_mat = jnp.clip(R_mat, -10., 1000)
    R_mean = np.mean(R_mat)
    R_max = np.max(R_mat - R_mean)
    init_t = time.time()
    key = keys[0]
    
 
    for t in range(epoch_len):
        if t % 250 == 0:
            s = s * 0 + 1
            if t < epoch_len * 0.66:
                random_mask = jax.random.randint(key, [guide_mask.shape[0] -1, 1], 
                                                 0, 200)
                guide_mask = jnp.concatenate([guide_mask[:1],
                                              random_mask])
        _ = RL.training_step(s, 
                          env.step,
                          R_f,
                          pi_f,
                          V_f,
                          jit_TD_learning,
                          jit_A2C_learning,
                          V_opt_update,
                          V_opt_get_params,
                          pi_opt_update,
                          pi_opt_get_params,
                          R_params,
                          pi_params,
                          pi_guide,
                          V_params,
                          V_opt_state,
                          pi_opt_state,
                          guide_mask,
                          discount,
                          key)
        s, V_opt_state, V_params, pi_opt_state, pi_params, guide_mask, key = _
    
    print('time', time.time() - init_t)

    pos_samples = []
    neg_samples_short = []
    
    pos_hm =  np.zeros((32, 32))
    neg_hm =  np.zeros((32, 32))
    
    for k in range(positive_sample_repetitions):
        s = 0 * s + 1
        alive = None
        for j in range(250):
            if j <= 200:
                explore = 1.0
            else:
                explore = 0.0
            s_next, done, r, a, keys = RL.inference(s, env.step, pi_f,
                                              pi_params, keys[1],
                                               explore=explore)
            done = done[:, 0]
            s = s_next
            if j > 200:
                if alive is None:
                    alive = 1 - done
                alive = alive * (1 - done)
                if np.sum(alive) > 0:
                    pos_samples.append(s_next[alive == 1] + 0)
                pos_hm = utils.update_heatmap(pos_hm, s)
            elif k == 0:
                neg_samples.append(s_next + 0)
                neg_samples_epoch.append(r * 0 + epoch)
                neg_samples_short.append(s_next + 0)
                neg_hm = utils.update_heatmap(neg_hm, s)
                
    rand_samples = []
    s = 0 * s + 1
    for j in range(30):
        s_next, _, r, a, keys = RL.inference(s, env.step, pi_f,
                                          pi_params, keys[1],
                                           explore=0.0)
        s = s_next
        rand_samples.append(s_next + 0)
    print('time', time.time() - init_t)
    pos_samples_jnp = jnp.concatenate(pos_samples, axis=0)
    neg_samples_jnp = jnp.concatenate(neg_samples, axis=0)
    rand_samples_jnp = jnp.concatenate(rand_samples, axis=0)
    neg_samples_epoch_jnp = jnp.concatenate(neg_samples_epoch, axis=0)[:, None]
    neg_samples_short_jnp = jnp.concatenate(neg_samples_short, axis=0)
    
    #TODO: FIX HACK
    #rand_samples_jnp = neg_samples_short_jnp
    #if epoch > 2:
    #    neg_samples = neg_samples + fut_neg_samples
    #    neg_samples_epoch = neg_samples_epoch + fut_neg_samples_e
    
    
    #fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))

    #print(R_max)
    np.save(f'data/{data_dir}/{epoch}.neg.npy', neg_hm)
    np.save(f'data/{data_dir}/maze.npy', env.mazes[0])
    
    with open(f'data/{data_dir}/{epoch}.params', 'wb') as f:
        pickle.dump(R_params, f)
    #axes[0].matshow(neg_hm)
    #axes[1].matshow(pos_hm)
    #axes[2].matshow(R_mat)
    #axes[3].matshow(jnp.clip(R_mat, 0.0, 0.05))

    #fig.tight_layout()
    #fig.show()   
    #plt.show()
    
    #R_params  = R_init(keys[1], s)
    #R_opt_state = R_opt_init(R_params)
    key = keys[0]
    for j in range(500):
        R_params, R_opt_state, key, loss = supervised.train(R_opt_update, R_opt_get_params,
                                                 R_f, 256, R_params, R_opt_state,
                                                 pos_samples_jnp, neg_samples_jnp, neg_samples_epoch_jnp,
                                                 neg_samples_short_jnp, rand_samples_jnp,
                                                 epoch, key)
    print('time', time.time() - init_t)
    #print('time', time.time() - init_t)

   
            
