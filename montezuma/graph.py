import haiku as hk
import coax
import jax
import jax.numpy as jnp

class Temp(hk.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, x):
        h_size = x.shape[-1]
        temp = hk.get_parameter("temp", shape=[1, h_size], dtype=x.dtype, init=jnp.ones)
        return x * temp
    
class MixHead(hk.Module):
    def __init__(self, output_size, name=None):
        super().__init__(name=name)
        self.output_size = output_size

    def __call__(self, x):
        j, k = x.shape[-1], self.output_size
        mix = hk.get_parameter("mix", shape=[1], dtype=x.dtype, init=jnp.ones)
        
        w1 = hk.get_parameter("w1", shape=[j, k], dtype=x.dtype, init=jnp.zeros)
        b1 = hk.get_parameter("b1", shape=[k], dtype=x.dtype, init=jnp.zeros)

        w2 = hk.get_parameter("w2", shape=[j, k], dtype=x.dtype, init=jnp.zeros)
        b2 = hk.get_parameter("b2", shape=[k], dtype=x.dtype, init=jnp.zeros)
        
        w = mix * w1 + (1 - mix) * jax.lax.stop_gradient(w2)
        b = mix * b1 + (1 - mix) * jax.lax.stop_gradient(b2)
        return jnp.dot(x, w) + b

def shared(S, is_training):
    seq = hk.Sequential([
        #coax.utils.diff_transform,
        hk.Conv2D(16, kernel_shape=8, stride=4), jax.nn.relu,
        hk.Conv2D(32, kernel_shape=4, stride=2), jax.nn.relu,
        hk.Flatten(),
    ])
    return seq(S/255.)

def policy(S, is_training, action_n, only_head):
    stop_grads = jax.lax.stop_gradient if only_head else lambda x: x
    logits = hk.Sequential((
        hk.Linear(256), jax.nn.relu,
        stop_grads,
        #MixHead(action_n),
        hk.Linear(action_n, w_init=jnp.zeros),
        #Temp(),
    ))
    X = shared(S, is_training)
    return {'logits': logits(X)}

def value(S, is_training, only_head):
    stop_grads = jax.lax.stop_gradient if only_head else lambda x: x
    value = hk.Sequential((
        hk.Linear(256), jax.nn.relu,
        stop_grads,
        hk.Linear(1, w_init=jnp.zeros), jnp.ravel
    ))
    X = shared(S, is_training)
    return value(X)

def reward_hk(S):
    reward = hk.Sequential((
        hk.Linear(256), jax.nn.relu,
        hk.Linear(1)#, jax.nn.softplus
    ))
    X = shared(S, False)
    return reward(X) #jax.nn.softplus(reward(X) - 4)

def reward(sample_data, rng):
    reward_f = hk.transform(reward_hk)
    reward_f = hk.without_apply_rng(reward_f)
    
    params = reward_f.init(rng, sample_data)

    return params, jax.jit(reward_f.apply)
