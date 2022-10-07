
from datetime import datetime
import functools
import os

from brax import envs
from brax import jumpy as jp
from brax.training import ppo
import our_ppo

import optuna

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def run_trial(trial):
  env_name = "halfcheetah"
  env_fn = envs.create_fn(env_name=env_name)

  train_fn = functools.partial(
        our_ppo.train, num_timesteps = 100000, log_frequency = 20,
        reward_scaling = trial.suggest_float("r_scaling", 5e-2, 2e2, log=True),
        episode_length = 1000, normalize_observations = True,
        action_repeat = 1, 
        unroll_length = trial.suggest_int("unroll", 1, 15), 
        num_minibatches = 32,
        num_update_epochs = trial.suggest_int("update_epoch", 1, 15), #4, 
        discounting = trial.suggest_float("discount", 0.9, 0.995, log=True),#0.97, 
        learning_rate = trial.suggest_float("lr", 1e-4, 2e-3, log=True), #3e-4,
        entropy_cost = trial.suggest_float("entropy", 1e-4, 1e-1, log=True),
        num_envs = 2048, batch_size = 128
  )

  xdata = []
  ydata = []
  times = [datetime.now()]

  def progress(num_steps, metrics):
    times.append(datetime.now())
    xdata.append(num_steps)
    ydata.append(metrics['eval/episode_reward'])

  _ = train_fn(environment_fn=env_fn, progress_fn=progress)

  return -max(ydata)
  #TODO run multiple runs and use average

study = optuna.create_study()
study.optimize(run_trial, n_trials=200)

print(study.best_params) 