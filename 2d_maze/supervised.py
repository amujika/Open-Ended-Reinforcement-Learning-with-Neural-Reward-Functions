import jax
import jax.random as jrandom
import jax.numpy as jnp
from functools import partial
import numpy as np

@partial(jax.jit, static_argnums=[0, 1, 2, 3])
def train(R_opt_update, R_opt_get_params, R_f, batch_s, 
          R_params, R_opt_state, pos_samples, neg_samples, neg_samples_epoch,
          neg_samples_short, neg_samples_rand, cur_epoch, key):
    
    keys = jrandom.split(key, 4)
    pos_batch = sample_batch(pos_samples, batch_s//4, keys[0])
    neg_batch, neg_batch_epoch = sample_batch_epoch(neg_samples, batch_s//4, keys[1], neg_samples_epoch)
    neg_batch_rand = sample_batch(neg_samples_rand, batch_s//4, keys[3])
    neg_batch_short = sample_batch(neg_samples_short, batch_s//4, keys[2])
    #neg_batch_epoch = neg_samples_epoch[idx]
    
    loss, grads = R_learning(R_f, R_params, pos_batch, neg_batch, neg_batch_epoch, neg_batch_short, neg_batch_rand, cur_epoch)
    
    R_opt_state = R_opt_update(0, grads, R_opt_state)
    R_params = R_opt_get_params(R_opt_state)
    
    return R_params, R_opt_state, keys[-1], loss

@partial(jax.value_and_grad, argnums=1)
def R_learning(R_f, R_params, pos_batch, neg_batch, neg_batch_epoch, neg_batch_short, neg_batch_rand, cur_epoch):
    pos_y = R_f(R_params, pos_batch)
    neg_y = R_f(R_params, neg_batch)
    neg_y_rand = R_f(R_params, neg_batch_rand)
    neg_y_short = R_f(R_params, neg_batch_short)
    
    pos_loss = jnp.mean(jnp.square(pos_y - 0.05))
    epoch_decay = jnp.clip(cur_epoch - neg_batch_epoch, 0, 10)
    #neg_loss = jnp.mean(jnp.square(neg_y - 0.025 + (0.1/10) * epoch_decay))
    neg_loss = jnp.mean(jnp.square(neg_y + 0.05))
    neg_loss_short = jnp.mean(jnp.square(neg_y_short + 0.05))
    neg_loss_rand  = jnp.mean(jnp.square(neg_y_rand + 0.05))
    
    #pos_loss = -jnp.mean(jnp.log(jax.nn.sigmoid(pos_y)))
    #neg_loss = -jnp.mean(jnp.log(1 - jax.nn.sigmoid(neg_y)))
    #neg_loss_short = -jnp.mean(jnp.log(1 - jax.nn.sigmoid(neg_y_short)))
    #neg_loss_epoch = -jnp.mean(jnp.log(1 - jax.nn.sigmoid(neg_y_epoch)))
    
    return 1.0 * pos_loss + neg_loss + neg_loss_short + 0 * neg_loss_rand

@partial(jax.jit, static_argnums=1)
def sample_batch(samples, batch_s, key):
    idx = jrandom.randint(key, [batch_s], 0, samples.shape[0])
    return samples[idx]

@partial(jax.jit, static_argnums=1)
def sample_batch_epoch(samples, batch_s, key, samples_epoch):
    idx = jrandom.randint(key, [batch_s], 0, samples.shape[0])
    #print(samples.shape, samples_epoch.shape)
    return samples[idx], samples_epoch[idx]
