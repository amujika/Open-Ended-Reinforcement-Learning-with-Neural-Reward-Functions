import jax.numpy as jnp
import jax
from jax import jit
import jax.example_libraries.optimizers as joptimizers
import haiku as hk
from functools import partial
import pickle

from configs import Configuration


class Network:
    def __init__(self, obs_size, rng, init_params = None):
        self.rng, init_rng = jax.random.split(rng)
        # Generate network and corresponding forward functions
        hk_activation_fn = hk.transform(activation_function)
        hk_activation_fn = hk.without_apply_rng(hk_activation_fn)

        self.params = hk_activation_fn.init(init_rng, jnp.zeros((obs_size)))
        if not init_params is None:
            self.params = init_params
        self.get_activation = jit(hk_activation_fn.apply)
        self.get_reward = partial(get_reward, 
                                  activation_fn=self.get_activation)

        # Generate optimizer for training the network
        (self.opt_init,
         self.opt_update,
         self.opt_get_params) = joptimizers.adam(0.001)
        self.opt_state = self.opt_init(self.params)
        # Not sure JITing here does anything, but it doesn't hurt....
        self.opt_update = jit(self.opt_update)
        self.opt_get_params = jit(self.opt_get_params)
        
    # Train once with the given data and update the internal variables
    def train_step(self, pos_samples, neg_samples, old_samples, batch_size):
        (loss,
         self.rng,
         self.opt_state,
         self.params) = _train_step(self.opt_state, 
                                    Configuration.reward_target_value,
                                    pos_samples, neg_samples, old_samples,
                                    self.rng, self.opt_update, 
                                    self.opt_get_params, batch_size,
                                    self.get_activation)
        return loss
    
    # Fix reward function to current parameters
    # WARNING: Not sure what happens if you change self.params afterwards. It 
    # will interact in weird ways with jax.jit probably. Don't do that!
    def get_reward_fn(self):
        def fixed_reward_fn(obs):
            return self.get_reward(self.params, obs)[..., 0]
        return fixed_reward_fn

# Changes the environment such that the reward is computed using reward_fn
def modify_env(env, reward_fn, test):
    env.unwrapped.step = replace_step_fn(env.unwrapped.step, reward_fn)
    if not isinstance(reward_fn, list):
        # No need to change the reset function in a single task setup
        return
    task_n = len(reward_fn) + 1
    # append 1-hot to observation, env.observation_size is changed automagically
    env.unwrapped.reset = replace_reset_fn(env.unwrapped.reset, task_n, test)


# Changes the step function such that the reward is computed using reward_fn
def replace_step_fn(step_fn, reward_fn):
    if not isinstance(reward_fn, list):
        # Just replace the reward in a single task setup
        def new_step_fn(state, action):
            next_state = step_fn(state, action)
            next_state = next_state.replace(reward=reward_fn(state.obs))
            return next_state
        return new_step_fn
    else:
        task_n = len(reward_fn) + 1
        def new_step_fn(state, action):
            task_id = state.obs[..., -task_n:]
            state = state.replace(obs=state.obs[..., :-task_n])
            next_state = step_fn(state, action)

            # Compute all possible reward for the next state
            next_rewards = [next_state.reward] \
                         + [jnp.squeeze(rew_fn(next_state.obs))
                               for rew_fn in reward_fn]
            next_reward = 0
            for i in range(task_n):
                # Select only the reward indicated by task_id
                next_reward += next_rewards[i] *  task_id[..., i]
            
            #Task ID stays constant
            next_obs = jnp.concatenate([next_state.obs, task_id], -1)
            next_state = next_state.replace(obs=next_obs,
                                            reward=next_reward)
            return next_state
        return new_step_fn

# Changes the reset function such that the observation has a 1hot encoded task
def replace_reset_fn(reset_fn, task_n, test):
    def new_reset_fn(rng):
        reset_rng, task_rng = jax.random.split(rng)
        state = reset_fn(reset_rng)
        batch_shapes = state.obs.shape[:-1]
        # Sample tasks u.a.r
        task = jnp.int32(jax.random.uniform(task_rng, 
                                            minval=0, 
                                            maxval=task_n,
                                            shape=batch_shapes))
        if test:
            task = task * 0 # Run only the original task in eval

        # Append the task id to the observation
        new_obs = jnp.concatenate([state.obs, 
                                   jax.nn.one_hot(task, task_n)], 
                                  -1)
        state = state.replace(obs=new_obs)
        return state
    return new_reset_fn

# Generate Haiku init/apply for the unclipped reward function.
def activation_function(obs):
    architecture = Configuration.reward_network_architecture
    non_linearity = architecture['nonlinearity']
    layer_sizes   = architecture['layer_sizes']
    num_layers = len(layer_sizes)

    sequential_layers = []
    for i in range(num_layers):
        sequential_layers.append(hk.Linear(layer_sizes[i]))
        if i != num_layers - 1: #No non-linearity on the output
            sequential_layers.append(non_linearity)

    reward_net = hk.Sequential(sequential_layers)
    return reward_net(obs)

# Modify activation for RL. Allows using binary, clipped, etc. rewards.
@partial(jit, static_argnames=['activation_fn'])
def get_reward(params, obs, activation_fn):
    activation = activation_fn(params, obs)
    reward = Configuration.reward_transformation_function(activation)
    return reward

# MSE where the positive/negative samples' label is +/-target, respectively.    
@partial(jax.value_and_grad, argnums = 0)
def compute_loss(params, pos_batch, neg_batch, target, activation_fn):
    pos_activation = activation_fn(params, pos_batch)
    neg_activation = activation_fn(params, neg_batch)

    pos_loss = jnp.mean(jnp.square(target - pos_activation))
    neg_loss = jnp.mean(jnp.square(target + neg_activation))

    return pos_loss + neg_loss

@partial(jit, static_argnames=['b_size'])
def get_batch(b_size, samples, key):
    idx = jax.random.randint(key, (b_size,), 0, jnp.shape(samples)[0])
    return samples[idx]

# Sample a batch of positive/negative/old samples and train on that
@partial(jit, static_argnames= ['opt_update',
                                'opt_get_params',
                                'batch_size', 
                                'activation_fn'])
def _train_step(opt_state, target,
                pos_samples, neg_samples, old_samples, key,
                opt_update, opt_get_params, batch_size, activation_fn):
    next_key, pos_key, neg_key, old_key = jax.random.split(key, 4)

    #TODO: Make proportion of different samples a Configuration parameter
    pos_batch = get_batch(batch_size // 3, pos_samples, pos_key)
    neg_batch = get_batch(batch_size // 3, neg_samples, neg_key)
    old_batch = get_batch(batch_size // 3, old_samples, old_key)

    # Old samples are treated as negative ones
    neg_batch = jnp.append(neg_batch, old_batch, 0)

    loss, grads = compute_loss(opt_get_params(opt_state), pos_batch,
                                neg_batch,target, activation_fn)

    opt_state = opt_update(0, grads, opt_state)
    return loss, next_key, opt_state, opt_get_params(opt_state)


# Save params of a reward network
def save_reward_params(file_path, reward_params):
    file = open(file_path, 'wb')
    pickle.dump(reward_params, file)
    file.close()

# Load reward network params and turn it into a reward_network object.
def load_reward_net(file_path):
    file = open(file_path, 'rb')
    rnet_params = pickle.load(file)
    file.close()
    # Get observation size by checking input dimension of the network.
    obs_size = rnet_params['linear']['w'].shape[0]
    # As we give parameters to the object, the random key is ignored.
    rnet = Network(obs_size, jax.random.PRNGKey(0),rnet_params)
    return rnet


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    net_key, pos_key, neg_key = jax.random.split(key, 3)

    rnet = Network(5, net_key)
    pos_samples = jax.random.normal(pos_key,(1000,5))
    neg_samples = jax.random.normal(neg_key,(1000,5)) + 10*jnp.ones((1000,5))
    all_neg_samples = neg_samples

    for j in jnp.arange(10000):
        loss  = rnet.train_step(pos_samples, neg_samples, all_neg_samples, 3)
        if j %499 ==0:
            print(loss)

