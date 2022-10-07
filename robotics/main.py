import os
import sys
import functools
from datetime import datetime
from brax import envs
import jax
from flax.core import frozen_dict

from configs import Configuration
import our_ppo
import exploration
from sample_generator import SampleGenerator
from reward_network import Network
# TODO: Now we need the whole module to save the params. How to proceed?
import reward_network as rnet
import plotting

# Generate the function for training taking the parameters from Configuration.py
def get_train_function():
    hyperparam = Configuration.ppo_hyperparameters[Configuration.train_env_name]
    return functools.partial(
        our_ppo.train, 
        num_timesteps = hyperparam['training_steps'],
        log_frequency = 20,
        reward_scaling = hyperparam['reward_scaling'], 
        episode_length = 1000,
        normalize_observations = True,
        action_repeat = 1, 
        unroll_length = hyperparam['unroll'], 
        num_minibatches = 32,
        num_update_epochs = hyperparam['num_update_epochs'],
        discounting = hyperparam['discount'], 
        learning_rate = hyperparam['lr'],
        entropy_cost = hyperparam['entropy_cost'],
        num_envs = 2048, batch_size = 1024,
        body_lr_factor = hyperparam['body_lr_factor'],
        l2_decay = hyperparam['l2_decay']
    )



# Reinitialize the policy head to zeros. 
def prepare_params(params):
    norm_params, pol_params, val_params = params

    if Configuration.reset_policy_head:
        # The policy has one extra layer
        last_layer_idx = len(Configuration.ppo_policy_layers)
        pol_params = frozen_dict.unfreeze(pol_params)
        pol_params["params"][f"hidden_{last_layer_idx}"]["kernel"] *= 0
        pol_params["params"][f"hidden_{last_layer_idx}"]["bias"]   *= 0

    if Configuration.reset_value_head:
        last_layer_idx = len(Configuration.ppo_value_layers) - 1
        val_params = frozen_dict.unfreeze(val_params)
        val_params["params"][f"hidden_{last_layer_idx}"]["kernel"] *= 0
        val_params["params"][f"hidden_{last_layer_idx}"]["bias"]   *= 0

    # norm_params[0] is number of steps. We want to decay it exponentially to
    # allow the mean/std to change over time, but we don't want to set it to 0, 
    # otherwise previous mean/std are lost inmediatly
    norm_params = (norm_params[0] * Configuration.normalization_step_decay,
                   norm_params[1],
                   norm_params[2])

    return (norm_params, pol_params, val_params)

# Trains the model to optimize the current reward function
def train(env_fn, rew_fn, init_params, iteration, verbose=True):
    train_fn = get_train_function()
    
    # Prepare the progress function
    times = [datetime.now()]
    progress_fn = functools.partial(plotting.plot_progress, 
                                    x_data=[], y_data=[],
                                    time_data=times,
                                    iteration=iteration)

    # Train
    inference_fns, params, _ = train_fn(environment_fn=env_fn, 
                                       init_params=init_params,
                                       progress_fn=progress_fn,
                                       reward_fn=rew_fn)

    params_file_path = f'data/agent_params/{iteration}.params'
    # The directory is found inside the function, only need to give the name
    plotting.save_params(params_file_path, params)
    
    if not Configuration.multitask_learning:
        # TODO: plotting is not compatible with multitask currently
        # Start a python program which creates a gif of a trajectory of this policy.
        # As this is quite slow, we do not want learning to wait on it. Hence the 
        # separate program.
        os.system(f'python3 plotting.py {iteration} &')

    if verbose:
        print(f'Time to JIT: {times[1] - times[0]}')
        print(f'Time to train agent: {times[-1] - times[1]}')
    
    return inference_fns, params


# Use the run_experiment.py file to start experiments
def main():
    # Without this things break for some reason.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


    env_fn = envs.create_fn(env_name=Configuration.train_env_name)


    key = jax.random.PRNGKey(Configuration.seed)
    key, reward_key, samples_key = jax.random.split(key, 3)
    obs_size = env_fn().observation_size
    act_size = env_fn().action_size
    # TODO: do we want to reset it at each generation?
    reward_network = Network(obs_size, reward_key)
    sample_generator = SampleGenerator(env_fn, samples_key)

    init_params = (None, None, None)

    for generation in range(Configuration.num_generations):  
        print('Current Epoch:', generation)
        # TODO: Give randomnessto the train function?
        # TODO: add normalization to the reward?
        rew_fn = reward_network.get_reward_fn()
        if generation in Configuration.original_reward_generations:
            rew_fn = None
        else:
            # Save the reward used by the current agent.
            rnet_file_path = f'data/reward_params/{generation}.rparams'
            rnet.save_reward_params(rnet_file_path, reward_network.params)

        # TODO: find solution for multitask learning to make code cleaner.
        if Configuration.multitask_learning:
            rew_fn = []
            for task in Configuration.multitask_ids:
                # TODO: this still assumes that we are outside the experiments 
                # folder, change it? Think about it when 
                # cleaning up the multitasking.
                file_path = f'{Configuration.multitask_folder}data/' + \
                            f'reward_params/{task}.rparams'
                cur_rew = rnet.load_reward_net(file_path)
                rew_fn.append(cur_rew.get_reward_fn())
        

        # TODO: name created files differently when we train on the 
        # real reward.
        inference_fns, inference_params = train(env_fn,
                                                rew_fn,
                                                init_params,
                                                generation)
        inference_fn, activation_fn, activation_to_act = inference_fns

        if generation in Configuration.original_reward_generations:
            # Don't generate samples or keep the parameters after training on
            # the original reward
            continue

        if Configuration.multitask_learning:
            # We run a single epoch when multitask_learning
            break

        # Fix inference function with current parameters
        inference_fn_fixed = lambda obs, key: inference_fn(inference_params,
                                                           obs, key)

        timer = datetime.now() 

        # Will be combined params from configs. Note that name clashes will 
        # take the version in updating_exploration_params.
        additional_exploration_params = {'activation_fn': activation_fn,
                                         'activation_to_act': activation_to_act,
                                         'agent_params': inference_params}
        explore_fn = exploration.create_function(additional_exploration_params,
                                                 act_size)

        # TODO: add exploration around origin as in Montezuma?
        sample_generator.update(explore_fn, inference_fn_fixed)
        (pos_samples, 
         neg_samples, 
         old_samples) = sample_generator.get_samples()
        print('Time to get samples:', datetime.now() - timer)
        
        timer = datetime.now()
        for i in range(Configuration.reward_grad_steps):
            reward_network.train_step(pos_samples, neg_samples, old_samples,
                                      Configuration.reward_batch_size)
        print('Time to train reward:', datetime.now() - timer)
        
        # TODO: check what to do with which part of the params
        init_params = prepare_params(inference_params)


if __name__ == '__main__':
    # Only use the specified GPU.
    cuda_device = str(sys.argv[1])
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    main()
