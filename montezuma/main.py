import os
import time
import coax
import jax
import haiku as hk
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
import jax.experimental.optimizers as joptimizers
from optax import adam
import gym
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from gym import spaces
from moviepy.editor import ImageSequenceClip
import matplotlib.pyplot as plt

import graph
import utils
import our_entropy
import our_ppo_clip

if __name__ == '__main__':
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    batch_s = 32
    trace_n = 1
    use_terminal = True
    game_id = 'MontezumaRevengeDeterministic-v4'
    exp_name = 'base_experiment'
    key = jrandom.PRNGKey(1)
    cur_s = jnp.zeros((batch_s, 84, 84, 4), jnp.uint8)

    os.system('mkdir gifs_' + exp_name )
    os.system('mkdir plots_' + exp_name )


    # env with preprocessing
    envs= [utils.make_env(game_id, seed) for seed in range(batch_s)]
    envs = SubprocVecEnv(envs)
    envs.reset()


    dummy_env = gym.make(game_id)
    _ = dummy_env.reset()
    #Freaking hack
    dummy_env.observation_space = spaces.Box(0, 255, (84, 84, 4), np.uint8)

    # function approximators
    policy_f_full = lambda S, is_training: graph.policy(S, is_training, 
                                                        dummy_env.action_space.n,
                                                        False) 
    policy_f_head = lambda S, is_training: graph.policy(S, is_training, 
                                                        dummy_env.action_space.n,
                                                        True) 
    value_f_full = lambda S, is_training: graph.value(S, is_training, 
                                                    False) 
    value_f_head = lambda S, is_training: graph.value(S, is_training, 
                                                    True) 
    pi_full = coax.Policy(policy_f_full, dummy_env)
    pi_head = coax.Policy(policy_f_head, dummy_env)
    v_full = coax.V(value_f_full, dummy_env)
    v_head = coax.V(value_f_head, dummy_env)

    reward_params, reward_f = graph.reward(cur_s, key)
    key, subkey = jrandom.split(key)

    @jax.jit
    def reward_training(params, x):
        reward = reward_f(params, x)
        loss = utils.compute_loss(reward)
        l2_loss = 1e-5 * joptimizers.l2_norm(params)
        loss += l2_loss
        loss -= jax.lax.stop_gradient(l2_loss) # Returns the true loss
        
        return loss 

    rew_grads = jax.value_and_grad(reward_training, argnums=0)
    opt_init, opt_update, opt_get_params = joptimizers.adam(0.001)

    if use_terminal:
        terminal_mask = np.zeros(batch_s) - 1.

    long_neg_samples_ = []
    terminal = np.zeros((batch_s)) + trace_n
    for epoch in range(1000):
        pi_head.params = pi_full.params
        v_head.params  = v_head.params
        pi = pi_head 
        v = v_head
        
        if epoch == 0:
            pi_guide = pi.copy()
        else:
            pi_guide = pi.copy()
            pi_params = hk.data_structures.to_mutable_dict(pi.params)
            
            #mix = pi_params['mix_head']['mix']
            #print(mix)
            #pi_params['mix_head']['w2'] = mix       * pi_params['mix_head']['w1'] \
            #                            + (1 - mix) * pi_params['mix_head']['w2']
            #pi_params['mix_head']['b2'] = mix       * pi_params['mix_head']['b1'] \
            #                            + (1 - mix) * pi_params['mix_head']['b2']
            
            #pi_params['mix_head']['w1'] = 0 * pi_params['mix_head']['w1']
            #pi_params['mix_head']['b1'] = 0 * pi_params['mix_head']['b1']
            #pi_params['mix_head']['mix'] = 1 + (0 * pi_params['mix_head']['mix'])
            
            
            #pi_params['temp']['temp'] = 0 * pi_params['temp']['temp']
            
            pi_params['linear_1']['w'] = 0 * pi.params['linear_1']['w']
            pi_params['linear_1']['b'] = 0 * pi.params['linear_1']['b']
            pi.params = hk.data_structures.to_immutable_dict(pi_params)
            #v_params = hk.data_structures.to_mutable_dict(v.params)
            #v_params[ 'linear_1']['w'] = 0 * v.params[ 'linear_1']['w']
            #v_params[ 'linear_1']['b'] = 0 * v.params[ 'linear_1']['b']
            #v.params = hk.data_structures.to_immutable_dict(v_params)

        # target networks
        v_targ = v.copy()
        
        # policy regularizer (avoid premature exploitation)
        #entropy = coax.regularizers.EntropyRegularizer(pi_head, beta=0.001)

        # updaters
        #simpletd = coax.td_learning.SimpleTD(v_head, optimizer=adam(3e-4))
        #ppo_clip = coax.policy_objectives.PPOClip(pi_head, regularizer=entropy,
        #                                          optimizer=adam(3e-4))
        # reward tracer and replay buffer
        tracer = coax.reward_tracing.NStep(n=trace_n, gamma=0.99)
        buffer = coax.experience_replay.SimpleReplayBuffer(capacity=128*batch_s)

        # run episodes


        tot_rew = 0
        avg_rew = 0
        init_t = time.time()
        
        cur_range = 400000
        cutoff = 30000
        steps_after_cutoff = 20000
        prev_s = None
        allowed_to_end = False
        extra_entropy = 9
        prev = 0.0
        for i in range(cur_range):
            if i >= cutoff and epoch != 0 and prev < 10.0:
                cutoff = i + steps_after_cutoff
            if i >= cutoff + steps_after_cutoff and allowed_to_end >= 4:
                if epoch != 0 and prev < 10.0:
                    cutoff = i + steps_after_cutoff
                else:
                    break
            if i == 0 or (i == 0 and epoch ==0):
                pi_full.params = pi_head.params
                pi = pi_full
                v  = v_full
                v_targ = v.copy()
                #entropy = coax.regularizers.EntropyRegularizer(pi_full, beta=0.003)
                entropy = our_entropy.EntropyRegularizer(pi_full, beta=0.003)
                simpletd = coax.td_learning.SimpleTD(v_full, v_targ, optimizer=adam(3e-4))
                ppo_clip = our_ppo_clip.PPOClip(pi_full, regularizer=entropy,
                                                optimizer=adam(3e-4), epsilon=0.2)
                
                
            if i % 500 == 0: 
                if i == 0:
                    tot_deaths = 1
                if tot_deaths <= 3:
                    allowed_to_end += 1
                else:
                    allowed_to_end = 0
                tot_deaths = 0
                envs.reset()
                prev_s = None
                tracer = coax.reward_tracing.NStep(n=trace_n, gamma=0.99)
                if i < cutoff and epoch != 0:
                    guide_mask = np.random.randint(450, size=batch_s) #- 1000
                    #guide_mask[batch_s // 2:] = guide_mask[batch_s // 2:] * 0 + 450 * (1 - (i / 20000.0))
                    guide_mask[0] = -10
                else:
                    guide_mask = np.zeros(batch_s)
            
            key, subkey = jrandom.split(key)

            logits = pi._function(pi.params, pi.function_state, pi.rng,
                                cur_s, is_training=False)[0]['logits']
            guide_logits = pi_guide._function(pi_guide.params,
                                            pi_guide.function_state,
                                            pi_guide.rng,
                                            cur_s, is_training=False)[0]['logits']
            
            a, logp = utils.sample_prob(key, logits)
            key, subkey = jrandom.split(key)
            guide_a, guide_logp = utils.sample_prob(key, guide_logits)
            
            a = np.where(guide_mask - 10 < 0, a, guide_a)
            logp = np.where(guide_mask - 10 < 0, logp, guide_logp)
            
            s_next, r, done, info = envs.step(a)
            tot_deaths += np.sum(done)
            if prev_s is None:
                prev_s = s_next[:, 17, 55:95, 0]
            #done = np.sum(prev_s != s_next[:, 17, 55:95, 0], axis=-1) > 0
            prev_s = s_next[:, 17, 55:95, 0]

            r = reward_f(reward_params, cur_s)[:, 0]
            r = jnp.clip(r, 0.0, 0.05)# * 500
            
            if use_terminal:
                terminal_mask = (1. - done) * terminal_mask \
                            + done * (trace_n - 1)
            
            s_next = utils.pre_process(s_next, cur_s)
            tot_rew += np.mean(r)
            # trace rewards and add transition to replay buffer
            
            tracer.add(cur_s, a, r, False, np.array(logp))
            cur_s = s_next
            while tracer:
                transition = tracer.pop()
                if use_terminal:
                    utils.fix_transition(transition, batch_s, guide_mask < 0,
                                        terminal_mask < 0)
                else:
                    utils.fix_transition(transition, batch_s, guide_mask < 0)
                buffer.add(transition)

            # learn
            if len(buffer) >= buffer.capacity:
                # 4 epochs per round
                num_batches = int(4 * buffer.capacity / (32 * batch_s)) 
                for _ in range(num_batches):
                    transition_batch = buffer.sample(32 * batch_s)
                    #print(transition_batch)
                    metrics_v, td_error = simpletd.update(transition_batch,
                                                        return_td_error=True)
                    metrics_pi = ppo_clip.update(transition_batch, td_error, extra_entropy)
                    v_targ.soft_update(v, tau=1.0)
                buffer.clear()
            if i % 5000 == 4999:
                if i >= cutoff + steps_after_cutoff:
                    extra_entropy -= 0
                if extra_entropy < 0: extra_entropy = 0
            if i % 1000 == 999:
                if epoch != 0 and i == cutoff + 999 and prev > np.mean(tot_rew):
                    cutoff += steps_after_cutoff
                print(i, np.mean(tot_rew), time.time() - init_t)
                init_t = time.time()
                prev = np.mean(tot_rew)
                tot_rew = 0
            guide_mask -= 1
            if use_terminal:
                terminal_mask -= 1

        pos_samples = []
        neg_samples = []
        
        
        for pos_epoch in range(10):
            movie = []
            rew_list = []
            envs.reset()
            prev_s = None
            pos_alive = None
            cum_rew = 0
            for i in range(500):
                key, subkey = jrandom.split(key)
                #TODO: Increase softmax temperature in final phase
                logits = pi._function(pi.params, pi.function_state, pi.rng,
                                    cur_s, is_training=False)[0]['logits']
                if i > 450:
                    logits *= 0.
                a, logp = utils.sample_prob(key, logits)

                s_next, r, done, info = envs.step(a)
                if pos_alive is None:
                    pos_alive = 1 - done
                pos_alive *= 1 - done
                if prev_s is None:
                    prev_s = s_next[:, 17, 55:95, 0] 
                #done = np.sum(prev_s != s_next[:, 17, 55:95, 0], axis=-1) > 0
                prev_s = s_next[:, 17, 55:95, 0] 
                r = reward_f(reward_params, cur_s)[:, 0]
                #r = jax.nn.relu(r)
                r = jnp.clip(r, 0.0, 0.05)# * 500
                cum_rew += r
                cur_s = utils.pre_process(s_next, cur_s)

                if i <= 450 and pos_epoch==0:
                    neg_samples.append(cur_s + 0)
                    if epoch <= 15:
                        long_neg_samples_.append(np.array(cur_s) + 0)
                    else:
                        idx = np.random.randint(len(long_neg_samples_))
                        long_neg_samples_[idx] = np.array(cur_s) + 0
                elif i > 450:
                    pos_alive *= cum_rew > 1.0
                    #if pos_alive is None:
                    #    pos_alive = 1 - done
                    #pos_alive *= 1 - done
                    if np.sum(pos_alive) == 0:
                        break
                    pos_samples.append(cur_s[pos_alive == 1] + 0)
                movie.append(s_next)
                rew_list.append(r + 0)

        #TODO: remove Random Agent neg samples
        envs.reset()
        for i in range(0):
            key, subkey = jrandom.split(key)

            if i ==0:
                logits = pi._function(pi.params, pi.function_state, pi.rng,
                                    cur_s, is_training=False)[0]['logits']
                logits *= 0.
            a, logp = utils.sample_prob(key, logits)

            s_next, r, done, info = envs.step(a)
            cur_s = utils.pre_process(s_next, cur_s)
            neg_samples.append(cur_s + 0)
            
        
        rew_list = np.stack(rew_list, axis=1)
        pos_samples = np.concatenate(pos_samples, axis=0)
        #print(pos_samples.shape)
        neg_samples_ = np.array(neg_samples)
        long_neg_samples = np.array(long_neg_samples_)
        pos_samples = np.reshape(pos_samples, (-1, 84, 84, 4))
        neg_samples = np.reshape(neg_samples, (-1, 84, 84, 4))
        long_neg_samples = np.reshape(long_neg_samples, (-1, 84, 84, 4))
        movie = np.stack(movie, axis=1)
        clip = ImageSequenceClip(list(movie[0]), fps=10)
        clip.write_gif('gifs_' + exp_name + '/' + str(epoch) + '.gif', fps=10)
        compress = 'gifsicle -i gifs_' + exp_name + '/' + str(epoch) \
                + '.gif -O3 --colors 32 -o gifs_' + exp_name + '/' + str(epoch) + ".gif"
        
        plt.plot(rew_list[0])
        plt.savefig('plots_' + exp_name + '/' + str(epoch) + '.png')
        plt.clf()
        #os.system(compress)


        opt_state = opt_init(reward_params)

        for _ in range(500*3):
            cur_batch = utils.get_batch(81 * 3*3, pos_samples, 
                                        neg_samples, long_neg_samples)
            loss, grads = rew_grads(reward_params, cur_batch)
            opt_state = opt_update(0, grads, opt_state)
            reward_params = opt_get_params(opt_state)
            #print(loss)
        print(loss)
        
