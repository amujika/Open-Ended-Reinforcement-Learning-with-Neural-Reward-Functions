
import matplotlib.pyplot as plt
from datetime import datetime
from brax.io import model
from brax.io import image
import jax
import os
import pickle
from brax import envs
import sys

import exploration
from configs import Configuration
import our_ppo
import reward_network
from particle_based_MI import plot_MI_and_scatter

# Writes progress to the *_data variables and stores a plot of it.
# Must be used with functools to specify a directory to save the plot.
def plot_progress(num_steps,
                  metrics,
                  x_data,
                  y_data,
                  time_data,
                  iteration):
    time_data.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(float(metrics['eval/episode_reward']))
    # x_data doesn't always start at 0, so we shift it
    real_x_data = [x_data[i]- x_data[0] for i in range(len(x_data))]
    learning_progress = {'num_steps': real_x_data, 'reward': y_data}
    file = open(f'data/training_progress/progress_{iteration}.data', 'wb')
    pickle.dump(learning_progress, file)
    file.close() 
    plt.style.use('ggplot')
    plt.xlim(Configuration.plot_x_limits)
    # To avoid getting out of the plot.
    y_limits = list(Configuration.plot_y_limits)
    y_limits[0] = min(y_limits + y_data)
    y_limits[1] = max(y_limits + y_data)
    plt.ylim(y_limits)
    plt.xlabel('# environment steps')
    plt.ylabel('reward per episode')
    plt.plot(real_x_data, y_data)
    plt.savefig(f'plot/training_progress/progress_{iteration}.png')
    plt.clf()


# Save params of a model
def save_params(file_path, params):
    model.save_params(file_path, params)

# Get the parameters back. IMPORTANT: The architecture of the agent specified in
# configs.py needs to be identical to the loaded one.
def load_params_and_inference_fn(iteration, obs_size, act_size):
    file_path = f'data/agent_params/{iteration}.params'
    # Need to give sample params to be able to match the structure
    inf_fns, sample_params = our_ppo.make_params_and_inference_fn(obs_size,
                                                                 act_size, True)
    loaded_params = model.load_params(file_path, sample_params)
    return inf_fns, loaded_params

# Get trajectory of the given inference functions in the given order for the 
# corresponding number of timesteps. This and the additional trajectories of 
# the other batched environments are saved.
def get_and_save_trajectory(state, step_fn, list_inference_fn, 
                            list_num_timesteps, key, file_path = None):
    list_inference_fn = [jax.jit(inference_fn) 
                            for inference_fn in list_inference_fn]
    trajectory = []
    multiple_trajectories = []

    for (inference_fn, timesteps) in zip(list_inference_fn, list_num_timesteps):
        for i in range(timesteps):
            key, act_key = jax.random.split(key)
            act = inference_fn(state.obs, act_key)
            state = step_fn(state, act)
            multiple_trajectories.append(state)
            # Get the state from the first environment by only taking the first 
            # row of each array.
            single_state = jax.tree_util.tree_map(lambda x: x[0], state)
            trajectory.append(single_state)
    
    if file_path is not None:
        file = open(file_path, 'wb')
        pickle.dump(multiple_trajectories, file)
        file.close() 

    return trajectory, state



# Create a gif of the given trajectory of states. 
def create_gif(file_name, env, trajectory):
    qps = [state.qp for state in trajectory]
    resolution = Configuration.gif_resolution
    gif = image.render(env.sys, qps, resolution, resolution, fmt='gif')
    path = f'gifs/{file_name}.gif'
    file = open(path, 'wb')
    file.write(gif)
    file.close()



# Plot the reward per timestep of the trained policy for the given trajectory of
# states. If a reward function is supplied, use it instead of the real reward.
def plot_episode_reward(file_name, trajectory, rew_fn = None):
    if rew_fn is None:
        rewards = [state.reward for state in trajectory]
    else:
        rewards = [rew_fn(state.obs) for state in trajectory]
    x_data = [i+1 for i in range(len(rewards))]
    x_limits = [min(x_data),max(x_data)]
    total_rew = int(sum(rewards))
    # To avoid getting out of the plot.
    y_limits = [min(rewards + [0]), max(rewards)+1]
    plt.style.use('ggplot')
    plt.xlim(x_limits)
    plt.ylim(y_limits)
    plt.xlabel(f'total reward: {total_rew}')
    plt.ylabel('reward')
    plt.plot(x_data, rewards)
    path = f'plot/episode_rewards/{file_name}'
    plt.savefig(path)
    plt.clf() 


# This main function plots a trajectory of the trained policy when called with 
# a generation index as sys.agrv[1]
# Also it is assumed that the file is run inside the experiment folder.
# Note that the randomness is fixed here, i.e. calling it twice gives 
# the same keys to the plotting functions.
if __name__ == "__main__":
    # To make sure that this runs on CPU and does not interfere with training.
    os.environ["CUDA_VISIBLE_DEVICES"] = ''

    iteration = str(sys.argv[1])

    # Prepare environment to get the trajectories.
    env_name = Configuration.train_env_name
    env_fn = envs.create_fn(env_name=env_name)
    # Only the first environment will be used for the plotting in this file, 
    # the others are used for the MI metric.
    env = env_fn(batch_size = Configuration.mi_trajectories)
    # TODO: is having a deterministic key here fine?
    key = jax.random.PRNGKey(0)
    key, gif_key, reset_key, episode_key, explore_key = jax.random.split(key, 5)
    first_state = env.reset(reset_key)
    step_fn = jax.jit(env.step)

    is_real_reward = int(iteration) in Configuration.original_reward_generations

    inference_fns, params = load_params_and_inference_fn(iteration,
                                                        env.observation_size,
                                                        env.action_size)
    inference_fn, activation_fn, activation_to_act = inference_fns
    # Fix the params into the function.
    prepared_inference_fn = lambda obs, key: inference_fn(params,
                                                        obs, 
                                                        key)

    trajectory_path = f'data/trajectories/{iteration}.states'
    trajectory, _ = get_and_save_trajectory(first_state, step_fn, 
                                        [prepared_inference_fn], 
                                        [1000], episode_key, 
                                        trajectory_path)

    # Plot the reward in one episode. If trained on the artificial reward, 
    # plot it as well.
    plot_name = f'real_rewards_{iteration}'
    plot_episode_reward(plot_name, trajectory)
    if not is_real_reward:
        file_path = f'data/reward_params/{iteration}.rparams'
        rew_net = reward_network.load_reward_net(file_path)
        rew_fn = rew_net.get_reward_fn()
        plot_name = f'artificial_rewards_{iteration}'
        plot_episode_reward(plot_name, trajectory, rew_fn)

    # Create gifs with the normal trajectory.
    create_gif(f'agents/agent_{iteration}', env, 
               trajectory[:Configuration.gif_num_steps])

    #Plotting exploration.
    _, explore_state = get_and_save_trajectory(first_state, step_fn, 
                                    [prepared_inference_fn], 
                                    [Configuration.explore_gif_num_steps[0]],
                                    episode_key, )

    explore_params = [2**(x-3) for x in range(7)]    
    for mult in explore_params:
        std_dev = mult * Configuration.exploration_params['std_dev']
        updating_exploration_params = {'activation_fn': activation_fn,
                                       'activation_to_act': activation_to_act,
                                       'agent_params': params,
                                       'std_dev': std_dev}

        explore_fn = exploration.create_function(updating_exploration_params,
                                                    env.action_size)

        exp_trajectory, _ = get_and_save_trajectory(
                                    explore_state, step_fn, [explore_fn],
                                    [Configuration.explore_gif_num_steps[1]],
                                    explore_key)

        create_gif(f'exploration/exploration_{iteration}_{mult}',
                   env, exp_trajectory)

    # Plot the particle based mutual information metric up to this generation. 
    # The plus one is because of the indexing
    plot_MI_and_scatter(int(iteration)+1)
    
    