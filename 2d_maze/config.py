import sys

class Configuration(object):    
    entropy = 0.005
    extra_entropy = 0.05 #0.035
    clip_reward = True
    clip_reward_below_only = False
    binarize_reward = False
    use_guide = True
    reset_policy = False
    reset_value = False
    reset_pi_head= True
    load_reward = False
    

    epoch_len = 5000 * 4 * 3
    policy_lr = 0.0001

    batch_size = 256 * 8