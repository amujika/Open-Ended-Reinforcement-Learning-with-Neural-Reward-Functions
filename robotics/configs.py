import jax.numpy as jnp
from functools import partial

class Configuration(object):
    # TODO: subsample scatter plot and MI metric
    # TODO: change random seed.
    # TODO: save episode rewards to compute mean +- std.
    train_env_name = 'humanoid'
    seed = 0
    # For easy experiment change
    explore = 'scale'
    explore_hp = 1

    num_generations = 50

    # TODO: what to do with the hp's which are the same everywhere?
    ppo_hyperparameters = {
    'ant':         {'lr' : 0.00029, 'reward_scaling' : 5.58242, 'unroll' : 5, 
                    'training_steps' : 20000000, 'num_update_epochs' : 14, 
                    'discount' :  0.92318, 'entropy_cost' : 0.00200, 
                    'l2_decay' : 0, 'body_lr_factor' : .5},
    'halfcheetah': {'lr' : 0.00010, 'reward_scaling' : 0.24532, 'unroll' : 3, 
                    'training_steps' : 10000000, 'num_update_epochs' : 15, 
                    'discount' :  0.99109, 'entropy_cost' : 0.00062, 
                    'l2_decay' : 0, 'body_lr_factor' : .5},
    'humanoid':    {'lr' : 0.00017, 'reward_scaling' : 0.15326, 'unroll' : 6, 
                    'training_steps' : 20000000, 'num_update_epochs' : 10, 
                    'discount' :  0.99114, 'entropy_cost' : 0.02087, 
                    'l2_decay' : 0, 'body_lr_factor' : .5},
    #zero shot hyperparameters below:
    #'ant':         {'lr' : 0.00012, 'reward_scaling' : 0.06651, 'unroll' : 1, 
    #                'training_steps' : 10000000, 'num_update_epochs' : 13, 
    #                'discount' :  0.97172, 'entropy_cost' : 0.00391, 
    #                'l2_decay' : 0, 'body_lr_factor' : 1},
    #'halfcheetah': {'lr' : 0.00025, 'reward_scaling' : 0.67, 'unroll' : 1, 
    #                'training_steps' : 2000000, 'num_update_epochs' : 4, 
    #                'discount' :  0.955, 'entropy_cost' : 0.0191, 
    #                'l2_decay' : 0, 'body_lr_factor' : 1},
    #'humanoid':   {'lr' : 0.00013, 'reward_scaling' : 0.21804, 'unroll' : 6, 
    #                'training_steps' : 100000000, 'num_update_epochs' : 11, 
    #                'discount' :  0.98146, 'entropy_cost' : 0.01162, 
    #                'l2_decay' : 0, 'body_lr_factor' : 1}
    }

    # An output head with the appropriate action space is added to the policy
    ppo_policy_layers = [512, 512]
    ppo_value_layers  = [512, 512, 1]

    original_reward_frequency = 5
    original_reward_generations = [] #[z for z in range(0, 
                                                    #num_generations,
                                                    #original_reward_frequency)]

    reward_network_architecture = {'layer_sizes': [87,1],
                                    'nonlinearity': jnp.tanh}
    reward_transformation_function = partial(jnp.clip, a_min = 0, a_max = 5)
    reward_target_value = 5.
    reward_grad_steps = 300
    # reward_batch_size should be divisible by 3, because of the sample split
    reward_batch_size = 171
    assert reward_batch_size % 3 == 0


    # If the folder structure is changed or new folders are added, the
    # run_experiment file has to be adapted to accomodate 
    # the change.
    experiment_name = f'{train_env_name}_{explore}_{explore_hp}_{seed}'
    #experiment_name = 'test'
    experiment_folder = f'experiments/{experiment_name}/'
    # Note if you change the next two values during an experiment, the gifs 
    # will change.
    gif_resolution = 200
    gif_num_steps = 400
    # first number is #steps with the trained policy, second #exploration steps.
    explore_gif_num_steps = [100,200]
    plot_x_limits = (0, ppo_hyperparameters[train_env_name]['training_steps'])
    y_limits = {'ant': 8000, 'humanoid': 15000, 
                'humanoidstandup': 50000, 'halfcheetah': 12000}
    plot_y_limits = (0, y_limits[train_env_name])

    samples_num_neg = 300
    samples_num_pos = 40
    samples_batch_size = 8192
    # We collect less negative samples from less environments to save memory
    samples_neg_fraction = 0.01

    # Choose the kind of exploration for the positive smaples with the 
    # corresponding function name from exploration.py and specify its parameters
    exploration_params = {'type': f'policy_{explore}_std',
                          'std_dev' : explore_hp,
                          'use_tanh': True}
    
    normalization_step_decay = 0.1

    reset_policy_head = True
    reset_value_head  = False

    multitask_learning = False
    multitask_folder = f'experiments/ant_skills/'
    multitask_ids = ['3', '2']
    if multitask_learning:
        # We don't use the real reward when multitask learning
        if len(original_reward_generations) != 0:
            print('original_reward_generations variable ignored for MULTITASK LEARNING')
            original_reward_generations = []

    mi_dimensions = {'ant':         {'x_vel': 13, 'y_vel': 14, 'z': 0},
                     'humanoid':    {'x_vel': 22, 'y_vel': 23, 'z': 0},
                     'halfcheetah': {'x_vel': 11, 'z': 0}}
    mi_1d_num_buckets = 1000
    # TODO: check that this is what the braxlines guy is actually doing.
    mi_cut_index = 500
    mi_trajectories = 10
    mi_limits = [-10,10]