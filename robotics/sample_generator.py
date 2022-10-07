from os import stat
import jax 
import jax.numpy as jnp

from configs import Configuration

class SampleGenerator:
    def __init__(self, env_fn, rng):
        self.rng = rng
        self.obs_size = env_fn().observation_size

        self.pos_samples = jnp.zeros((0, self.obs_size))
        self.neg_samples = jnp.zeros((0, self.obs_size))
        self.old_samples = jnp.zeros((0, self.obs_size))

        episode_length = Configuration.samples_num_neg  \
                       + Configuration.samples_num_pos

        self.env = env_fn(batch_size = Configuration.samples_batch_size,
                          episode_length = episode_length)

        self.reset = jax.jit(self.env.reset, backend='cpu')
        self.step = jax.jit(self.env.step)


    # Update the Sample store with new observations.
    def update(self, pos_inference_fn, neg_inference_fn):
        env_key, neg_key, pos_key, self.rng = jax.random.split(self.rng, 4)

        # We assume there is a single device
        assert len(jax.devices()) == 1
        state = self.reset(env_key)
        state = jax.device_put(state, jax.devices()[0]) # Reset runs on CPU


        (self.neg_samples, 
         state) = self.collect_samples(state, neg_inference_fn,
                                       Configuration.samples_num_neg,
                                       neg_key, False, 
                                       Configuration.samples_neg_fraction)
        (self.pos_samples,
         state) = self.collect_samples(state, pos_inference_fn,
                                       Configuration.samples_num_pos,
                                       pos_key, True)

        self.old_samples = jnp.append(self.old_samples, self.neg_samples, 0)

    # Collect samples according to inference_fn from the first 'fraction' 
    # fraction of environments
    # Cut positive trajectories at the first terminal state.
    def collect_samples(self, state, inference_fn, 
                        num_samples, key, are_positive, env_fraction=1.):
        samples = []
        env_number = int(Configuration.samples_batch_size * env_fraction)

        done = state.done

        inference_fn = jax.jit(inference_fn)
        act_keys = jax.random.split(key, num_samples)

        for i in jnp.arange(num_samples):
            action = inference_fn(state.obs, act_keys[i])
            state = self.step(state, action)
            if are_positive:
                obs =  jnp.compress( 1-done, state.obs, axis = 0)
                samples.append(obs)
                done =  jnp.maximum(done, state.done)
            else:
                samples.append(state.obs[:env_number])
        samples = jnp.concatenate(samples, axis=0)
        return samples, state
    
    def get_samples(self):
        return self.pos_samples, self.neg_samples, self.old_samples

        
        
if __name__ == "__main__":
    from brax import envs
    import time
    import os
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    key = jax.random.PRNGKey(0)
    env_fn = envs.create_fn(env_name='ant')
    sample_generator = SampleGenerator(env_fn, key)
    action_size =  env_fn().action_size

    @jax.jit
    def neg_inference_function(obs,key):
        action = jnp.zeros((Configuration.samples_batch_size,action_size))
        return action
    
    @jax.jit
    def pos_inference_function(obs,key):
        action = .5* jnp.ones((Configuration.samples_batch_size,action_size))
        return action

    timer = [time.time()]    
    for i in range(3):
        sample_generator.update(neg_inference_function, pos_inference_function)
        timer.append(time.time())
        pos_samples, neg_samples, old_samples = sample_generator.get_samples()
        print(jnp.shape(pos_samples))
        print(jnp.shape(neg_samples))
        print(jnp.shape(old_samples))

    print(timer[1]- timer[0])
    print(timer[2]- timer[1])

    






