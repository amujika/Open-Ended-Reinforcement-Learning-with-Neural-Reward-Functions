import os
import sys
import pickle
from configs import Configuration

# Creating the right folder structure to save plots/data/gifs. 
# Throws an error if the experiment name was already used.
def create_directories(experiment_folder):
    if os.path.exists(experiment_folder):
        raise Exception('The experiment name was already used.')
    os.system(f'mkdir -p {experiment_folder}gifs/agents/ &')
    os.system(f'mkdir -p {experiment_folder}gifs/exploration/ &')
    os.system(f'mkdir -p {experiment_folder}plot/episode_rewards/ &')
    os.system(f'mkdir -p {experiment_folder}plot/training_progress/ &')
    for key in Configuration.mi_dimensions[Configuration.train_env_name]:
        os.system(f'mkdir -p {experiment_folder}plot/' +
                  f'mutual_information/{key}/ &')
    os.system(f'mkdir -p {experiment_folder}plot/' + 
              f'mutual_information/scatter_plots/ &')    
    os.system(f'mkdir -p {experiment_folder}data/agent_params/ &')
    os.system(f'mkdir -p {experiment_folder}data/reward_params/ &')
    os.system(f'mkdir -p {experiment_folder}data/trajectories/ &')
    os.system(f'mkdir -p {experiment_folder}data/training_progress/ &')

    episode_rewards = []
    file = open(f'{experiment_folder}data/real_rewards.data', 'wb')
    pickle.dump(episode_rewards, file)
    file.close

# Creates the needed folder structure for this experiment in the experiment 
# folder, copies the current code into it and runs the experiment with 
# working directory inside it. 
# Tell it the index of the GPU it should use for the experiment,
# if none is specified, only create the folder structure and copy the code.
if __name__ == '__main__':
    create_directories(Configuration.experiment_folder)
    # Copy all code into the folder and change the working directory into it.
    os.system(f'cp *.py {Configuration.experiment_folder} &')

    if len(sys.argv) > 1:
        cuda_device = sys.argv[1]
        os.chdir(Configuration.experiment_folder)
        os.system(f'python3 main.py {cuda_device}')
    else:
        print('You only copied the code and the experiment is not running!')
