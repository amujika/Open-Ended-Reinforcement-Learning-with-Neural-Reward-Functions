import haiku as hk
import jax
import jax.numpy as jnp

#def to_1hot(x):
#    y = x[:, :1] + 32 * x[:, 1:]
#    y = jax.nn.one_hot(y, 32 * 32, dtype=jnp.float32)
#    y = jnp.reshape(y, [-1, 32 * 32])
#    return y

def to_1hot(x):
    y = x[:, :1] + 32 * x[:, 1:]
    y = jax.nn.one_hot(y, 32 * 32, dtype=jnp.float32)
    y = jnp.reshape(y, [-1, 32, 32, 1])#

    f = hk.Sequential((
        hk.Conv2D(4, kernel_shape=5, stride=3), jax.nn.relu,
        hk.Conv2D(8, kernel_shape=4, stride=3), jax.nn.relu,
        hk.Flatten()
    ))
    return f(y)

#def to_1hot(x):
#    y = x / 32. #x[:, :1] + 32 * x[:, 1:]
    #y = jax.nn.one_hot(y, 32 * 32, dtype=jnp.float32)
    #y = jnp.reshape(y, [-1, 32 * 32])
#    return y

#def to_1hot(x):
#    y = jax.nn.one_hot(x, 32, dtype=jnp.float32)
#    y = jnp.reshape(y, [-1, 64])
#    return y

def policy_hk(S):
    f = hk.Sequential((
        to_1hot,
        hk.Linear(100), jax.nn.relu,
        #hk.Linear(256), jax.nn.relu,
        hk.Linear(5, w_init=jnp.zeros, with_bias=True),
    ))
    pi = f(S) 
    return pi

def value_hk(S):
    f = hk.Sequential((
        to_1hot,
        hk.Linear(100), jax.nn.relu,
        #hk.Linear(256), jax.nn.relu,
        hk.Linear(1),#, w_init=jnp.zeros),
    ))
    v = f(S)
    return v

def reward_hk(S):
    return value_hk(S)

def get_policy(sample_data, rng):
    policy = hk.transform(policy_hk)
    policy = hk.without_apply_rng(policy)
    
    params = policy.init(rng, sample_data)

    return policy.init, jax.jit(policy.apply)

def get_value(sample_data, rng):
    value = hk.transform(value_hk)
    value = hk.without_apply_rng(value)
    
    params = value.init(rng, sample_data)

    return value.init, jax.jit(value.apply)

def get_reward(sample_data, rng):
    reward = hk.transform(reward_hk)
    reward = hk.without_apply_rng(reward)
    
    params = reward.init(rng, sample_data)

    return reward.init, jax.jit(reward.apply)
