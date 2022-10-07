import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import jit
from functools import partial
import utils
from config import Configuration

@partial(jit, static_argnums=(1, 2))
def unrolled_inference(s, 
                       env_func,
                       pi_func,
                       pi_params,
                       key):
    for i in range(10):
        s_next, r, a, keys = inference(s, 
                                       env_func,
                                       pi_func,
                                       pi_params,
                                       key)
        s = s_next
        key = keys[1]
    return s_next, r, a, keys
        

@partial(jit, static_argnums=(1, 2), inline=True)
def inference(s, 
              env_func,
              pi_func,
              pi_params,
              key,
              explore=1.0):
    
    logits = pi_func(pi_params, s)
    logits = logits * explore
    #logits = jnp.where(explore, logits *0, logits)
    a = jrandom.categorical(key, logits)
        
    
    s_next, terminal = env_func(a, s)
    
    #Hacky, to be removed
    r  = (s_next[:, 0] == 21) + 0.
    r *= (s_next[:, 1] == 15) + 0.
    
    return s_next, terminal, r, a, jax.random.split(key, 2)

def TD_advantage(V_func, V_params, s, s_next, cur_r, discount):
    cur_V  = V_func(V_params, s)[:, 0]
    next_V = V_func(V_params, s_next)[:, 0]
    next_V = jax.lax.stop_gradient(next_V)

    #print(next_V.shape)
    #print(discount.shape)
    #print(aaa)
    advantage = cur_V - (cur_r + discount * next_V)
    #print(
    #print(advantage)
    #print(aaa)
    return advantage

@partial(jax.value_and_grad, has_aux=True, argnums=1)
def TD_learning(V_func, V_params, s, s_next, cur_r, discount, mask):
    advantage = TD_advantage(V_func, V_params, s, s_next, cur_r, discount)
    advantage *= mask
    
    return jnp.mean(jnp.square(advantage)), advantage

@partial(jax.value_and_grad, argnums=1)
def A2C_learning(pi_func, pi_params, s, a, r, adv):
    logits = pi_func(pi_params, s)
    log_p = jax.nn.log_softmax(logits)
    p = jax.nn.softmax(logits)
    log_p_act = log_p[jnp.arange(log_p.shape[0]), a]
    
    entropy = -jnp.sum(log_p * p, axis=-1) 
    adv = jax.lax.stop_gradient(adv)

    extra_entropy = (r > 0) * Configuration.extra_entropy 
    print(entropy.shape, extra_entropy.shape)

    return jnp.mean(adv * log_p_act) \
        - jnp.mean((Configuration.entropy + extra_entropy) * entropy)


@partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
def training_step_(s, 
                   env_func,
                   R_func,
                   pi_func,
                   V_func,
                   TD_func,
                   A2C_func,
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
                   key):
    for i in range(10):
        _ = training_step(s, 
                             env_func,
                              R_func,
                              pi_func,
                              V_func,
                              TD_func,
                              A2C_func,
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
    return s, V_opt_state, V_params, pi_opt_state, pi_params, guide_mask, key

def get_training_step(env_func,
                    R_func,
                    pi_func,
                    V_func,
                    TD_func,
                    A2C_func,
                    V_opt_update,
                    V_opt_get_params,
                    pi_opt_update,
                    pi_opt_get_params):
    def f(carry, x):
        (s,
        R_params,
                  pi_params,
                  pi_guide,
                  V_params,
                  V_opt_state,
                  pi_opt_state,
                  guide_mask,
                  discount,
                  key) = carry
        _ = training_step(s, 
                  env_func,
                  R_func,
                  pi_func,
                  V_func,
                  TD_func,
                  A2C_func,
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
        next_carry = (s,
                        R_params,
                        pi_params,
                        pi_guide,
                        V_params,
                        V_opt_state,
                        pi_opt_state,
                        guide_mask,
                        discount,
                        key)  
        return next_carry, x
    return f

@partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
def training_step(s, 
                  env_func,
                  R_func,
                  pi_func,
                  V_func,
                  TD_func,
                  A2C_func,
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
                  key):
    keys = jax.random.split(key, 3)
    
    #s = utils.random_s_reset(s, 1 - 1/500., keys[0])
    s_next, terminal, r, a, keys = inference(s, env_func, pi_func,
                                      pi_params, keys[1])
    
    s_next_guide, _, _, _, keys = inference(s, env_func, pi_func,
                                         pi_guide, keys[1])
    #TODO: Fix learning for guided
    if not Configuration.use_guide:
        guide_mask = guide_mask * 0 - 1
    if Configuration.use_guide:
        s_next = jnp.where(guide_mask < 0, s_next, s_next_guide)
    
    r = R_func(R_params, s_next)[:, 0]
    #r = jnp.clip(r, 0.0, 0.0001) * 500
    if Configuration.clip_reward:
        r = jnp.clip(r, 0, 0.05)
    if Configuration.clip_reward_below_only:
        r = jnp.clip(r, 0, 500)
    if Configuration.binarize_reward:
        r = jnp.sign(r) * 0.05
    #r = jnp.clip(r, -.05, 0.05) * 0.01 + jnp.clip(r - 0.04, 0, 0.00001) * 100
    #r *= 100
    discount = discount * (1 - terminal[..., 0])
    
    td_loss, grads = TD_func(V_func, V_params, 
                             s, s_next, r,
                             discount, guide_mask < 0)
    V_opt_state = V_opt_update(0, grads, V_opt_state)
    V_params = V_opt_get_params(V_opt_state)


    pi_loss, grads = A2C_func(pi_func, pi_params, s, a, r, td_loss[1])
    pi_opt_state = pi_opt_update(0, grads, pi_opt_state)
    pi_params = pi_opt_get_params(pi_opt_state)
    
    return s_next, V_opt_state, V_params, pi_opt_state, pi_params, guide_mask - 1, keys[2]
    
    