import gym
import numpy as np
from functools import partial

from jax import jit
import jax.image as jimage
import jax.numpy as jnp
import jax.random as jrandom
import jax.nn as jnn

# Auxiliary function to create subprocess environments
def make_env(env_id, seed):
    def _f():
        env = gym.make(env_id)
        env.seed(seed)
        return env
    return _f

# Reshape/recolor the current observation and stack it with the previous ones
@jit
def pre_process(cur_obs, prev_obs, frame_stack=4):
    batch_size = cur_obs.shape[0]
    
    obs = jimage.resize(cur_obs[:, 30:], #TODO: remove this hack to ignore lives 
                        (batch_size, 84,84,3),
                        method="nearest")
    obs = jnp.mean(obs, axis=-1, keepdims=True)# / 255.#, dtype=jnp.uint8) / 255.
    
    ans= jnp.concatenate((obs, 
                          prev_obs[:, :, :, :frame_stack-1])
                         ,axis=-1)
    
    return jnp.uint8(ans)

# Given some logits, output a categorical sample and the log probability of it
@jit
def sample_prob(key, logits):
    a = jrandom.categorical(key, logits)
    logp = jnn.log_softmax(logits)    
    logp = [logp[_, a[_]]for _ in range(a.shape[0])]
    return a, logp

# Converts the transition from the tracer into a batch of transitions.
# This allows using the tracer in batch mode
def fix_transition(transition, batch_s, mask=None, non_terminal=None):
    if mask is None:
        mask = np.arange(batch_s) < batch_s + 1
    
    transition.S         = transition.S[0][mask]
    transition.A         = transition.A[0][mask]
    transition.logP      = transition.logP[0][mask]
    transition.Rn        = transition.Rn[0][mask]
    transition.S_next    = transition.S_next[0][mask]
    transition.A_next    = transition.A_next[0][mask]
    transition.logP_next = transition.logP_next[0][mask]

    transition.In  = np.repeat(transition.In,  batch_s)
    if not non_terminal is None:
        transition.In *= non_terminal
    transition.In = transition.In[mask]

    transition.W   = np.repeat(transition.W,   batch_s)[mask]
    transition.idx = np.repeat(transition.idx, batch_s)[mask]


#TODO: jit this
#@partial(jit, static_argnums=(0,))
def get_batch(b_size, pos_samples, neg_samples, long_neg_samples):
    assert b_size % 3 == 0
    pos_idx = np.random.randint(0, pos_samples.shape[0], b_size // 3)
    pos_batch = pos_samples[pos_idx]
    neg_idx = np.random.randint(0, neg_samples.shape[0], b_size // 3)
    neg_batch = neg_samples[neg_idx]
    long_neg_idx = np.random.randint(0, long_neg_samples.shape[0], b_size // 3)
    long_neg_batch = long_neg_samples[long_neg_idx]
    
    return np.concatenate((pos_batch, neg_batch, long_neg_batch), axis=0)

def compute_loss(reward):
    b_size = reward.shape[0]
    assert b_size % 3 == 0
    loss_pos = jnp.square(0.05 - reward[:b_size // 3])
    loss_pos = jnp.mean(loss_pos)
    loss_neg = jnp.square(0.05 + reward[b_size // 3:])
    loss_neg = jnp.mean(loss_neg)
    
    return loss_pos + loss_neg
