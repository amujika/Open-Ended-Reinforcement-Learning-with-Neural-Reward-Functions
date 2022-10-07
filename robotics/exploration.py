import jax
import jax.numpy as jnp

from configs import Configuration

# All functions need to be of the following form: 
# f: dict, action_size, batch_mode -> explore_fn, where 
# explore_fn: observation, batch_size, key -> action
# All function specific params need to be passed in the dict.
# batch_mode has to be specified to have the right form of action.
# Add new functions also in create_function, so they can be used 
# by only changing the config file.

# Note that additional params with the same name overrule the config params.
def create_function(additional_params, action_size):
  exploration_params = {**Configuration.exploration_params, 
                        **additional_params}
  if exploration_params['type'] == 'fixed_std':
    observation2action = fixed_std(exploration_params, action_size)
  if exploration_params['type'] == 'policy_fixed_std':
    observation2action = policy_fixed_std(exploration_params, action_size)
  if exploration_params['type'] == 'policy_scale_std':
    observation2action = policy_scale_std(exploration_params, action_size)
  if exploration_params['type'] == 'policy_mult_std':
    observation2action = policy_mult_std(exploration_params, action_size)
  wrapped_observation2action = wrap_for_batch_size(observation2action)
  return wrapped_observation2action

# As the activation of the units corresponding to std is transformed 
# with log(1+exp(x)) + .001, we calculate the desired activation 
# to have the given std after transformation. 
def invert_target_std_dev(std_dev):
  target = jnp.log(jnp.exp(std_dev - 0.001) - 1)
  return target


# wrap the exploration function to make observation shapes always be in 
# batch form. Note that if the batch_size is changed, it will be recompiled.
def wrap_for_batch_size(exploration_function):
  @jax.jit
  def wrapped_exploration_function(observation, key):
    if len(observation.shape) >= 2:
      batch_size = observation.shape[0]
    else:
      batch_size = 1
      observation = jnp.expand_dims(observation, 0)
    action = exploration_function(observation, batch_size, key)
    if observation.shape[0] == 1:
      action = jnp.squeeze(action)
    return action
  return wrapped_exploration_function

# Set all elements which do not encode means to 0
def set_std_to_zero(activation, batch_size, action_size):
  mask = jnp.append(jnp.ones((batch_size, action_size)),
                    jnp.zeros((batch_size, action_size)), 
                    axis = 1)
  means = activation * mask
  return means

# Set all elements which do not encode std_devs to 0
def set_mean_to_zero(acivation, batch_size, action_size):
  std_devs = acivation - set_std_to_zero(acivation, batch_size, action_size)
  return std_devs


# take the policy given in by the params, keep the mean of the Gaussian, 
# substitute the standard deviation by a given parameter.
def policy_fixed_std(exploration_params, action_size):
  activation_fn = exploration_params['activation_fn']
  activation_to_act = exploration_params['activation_to_act']
  inference_params = exploration_params['agent_params']
  target_activity = invert_target_std_dev(exploration_params['std_dev'])

  def observation2action(observation, batch_size, key):
    activation = activation_fn(inference_params, observation)
    means = set_std_to_zero(activation, batch_size, action_size)

    constant_matrix = target_activity * jnp.ones((batch_size, 2 * action_size))
    std_devs = set_mean_to_zero(constant_matrix, batch_size, action_size)

    mod_activation = means + std_devs
    action = activation_to_act(mod_activation, key)
    return action
  return observation2action

# Generates a function that samples actions from tanh(Gaussian(0, std_dev))  
def fixed_std(params_dict, action_size):
  std_dev  = params_dict['std_dev']
  use_tanh = params_dict['use_tanh']
  
  def observation2action(observation, batch_size, key):
    action = jax.random.normal(key, (batch_size, action_size))
    action *= std_dev
    if use_tanh:
      action = jnp.tanh(action)
    return action    
  return observation2action

# take the policy given in by the params, keep the mean of the Gaussian, 
# substitute the standard deviation by adding a constant to all std output 
# units in such a way that the maximum is equal to std_dev (after transform).
def policy_scale_std(exploration_params, action_size):
  activation_fn = exploration_params['activation_fn']
  activation_to_act = exploration_params['activation_to_act']
  inference_params = exploration_params['agent_params']
  target_activity = invert_target_std_dev(exploration_params['std_dev'])

  def observation2action(observation, batch_size, key):
    activation = activation_fn(inference_params, observation)

    means = set_std_to_zero(activation, batch_size, action_size)

    std_devs = set_mean_to_zero(activation, batch_size, action_size)
    _, only_std_devs = jnp.split(std_devs, 2, axis=1)
    max_std_devs = jnp.max(only_std_devs, axis = 1)
    add = jnp.expand_dims(target_activity - max_std_devs, 1)
    shifted_std_devs = std_devs + add
    std_devs = set_mean_to_zero(shifted_std_devs, batch_size, action_size)

    mod_activation = means + std_devs
    action = activation_to_act(mod_activation, key)
    return action
  return observation2action


# take the policy given in by the params, keep the mean of the Gaussian, 
# substitute the standard deviation by adding std_dev to all std output units. 
# This translates to taking log(1 + e^c e^x) + 0.001 as std of the Gaussian 
# instead of log(1+e^x) + 0.001.
# Not so sure how sensible this one is.
def policy_mult_std(exploration_params, action_size):
  activation_fn = exploration_params['activation_fn']
  activation_to_act = exploration_params['activation_to_act']
  inference_params = exploration_params['agent_params']
  # TODO: Different name in dict?
  additive_constant = exploration_params['std_dev']

  def observation2action(observation, batch_size, key):
    activation = activation_fn(inference_params, observation)

    means = set_std_to_zero(activation, batch_size, action_size)

    shifted_std_devs = activation + additive_constant
    std_devs = set_mean_to_zero(shifted_std_devs, batch_size, action_size)

    mod_activation = means + std_devs
    action = activation_to_act(mod_activation, key)
    return action
  return observation2action
