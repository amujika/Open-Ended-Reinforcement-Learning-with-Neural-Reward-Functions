import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from functools import partial

@jax.jit
def update_heatmap(old_heatmap, s):
    assert old_heatmap.shape[0] == old_heatmap.shape[1]
    
    size = old_heatmap.shape[0]
    s_1d = s[:, 1] + size * s[:, 0]
    s_1d = jax.nn.one_hot(s_1d, size * size, dtype=jnp.float32)
    
    cur_heatmap = jnp.sum(s_1d, axis=0)
    cur_heatmap = jnp.reshape(cur_heatmap, old_heatmap.shape)
    
    return old_heatmap + cur_heatmap

@jax.jit
def random_s_reset(s, p, key):
    reset = jax.random.bernoulli(key, p, [s.shape[0], 1])
    s_new = reset * s +  (1 - reset) * (s * 0 + 1)
    
    return s_new


def visualize_R(R_f, R_params):
    inp = np.zeros([32 * 32, 2])
    for i in range(32):
        for j in range(32):
            inp[i + 32*j, 0] = i  
            inp[i + 32*j, 1] = j
    cur_r_ = R_f(R_params, inp)[:, 0]
    cur_r = np.zeros([32, 32])
    for i in range(32):
        for j in range(32):
            cur_r[i, j] = cur_r_[i + 32*j]  
    
    return cur_r