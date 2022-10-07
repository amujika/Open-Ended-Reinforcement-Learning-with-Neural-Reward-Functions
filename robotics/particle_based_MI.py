import matplotlib.pyplot as plt
import numpy as np
import pickle

from configs import Configuration

def compute_entropy(distribution):
    normalized_distribution = distribution / np.sum(distribution)
    entropy = 0
    for prob in normalized_distribution:
        # + 1e-12 to avoid instabilities, shouldn't matter here.
        entropy += - prob * np.log(prob + 1e-12)
    return entropy

def x2idx(x, limits, num_buckets):
    # Set minimum value to 0
    shifted_x = x - limits[0] 
    normalizer = limits[1] - limits[0]
    #Set maximum value to 1
    normalized_x = shifted_x / normalizer 

    return int(normalized_x * num_buckets)

# Compute and plot the one dimensional MI metric given a set of observations in 
# this dimension from a number of skills. The MI metric is equal to the 
# "entropy" of all observations minus the average "entropy" 
# of the observations of a single skill. "entropy" is the entropy of a 
# normalized histrogram in this dimension with a fixed number of buckets.
def plot_1d_MI(skills, dim_name, subsampling):
    num_buckets = Configuration.mi_1d_num_buckets
    limits = Configuration.mi_limits
    
    number_of_skills = len(skills)
    if subsampling:
        skills =  skills[((number_of_skills-1) % 3)::3]

    # Make an iterable color palette for different skills
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(skills))))


    mi = 0
    # buckets for all skills together
    total_p = np.zeros(num_buckets)

    for skill in skills:
        # buckets for only the current skill
        cur_p = np.zeros(num_buckets)
        for obs in skill:
            cur_p[  x2idx(obs, limits, num_buckets)] += 1
            total_p[x2idx(obs, limits, num_buckets)] += 1
        
        c = next(color)
        plt.plot(np.linspace(limits[0], limits[1], num_buckets), cur_p, c = c)
        mi -= (1/len(skills)) * compute_entropy(cur_p)
    mi += compute_entropy(total_p)
    plt.xlabel(f'{dim_name}: MI {mi} #buckets {num_buckets}')
    plt.ylabel('#obs in bucket')
    # the minus one is there to match gif name of the last skill added.
    plt.savefig(f'plot/mutual_information/' +
                f'{dim_name}/{dim_name}_{number_of_skills-1}_{subsampling}')
    plt.clf()
    return mi

def scatter_plot_2d(scatter_skills, mi_dimensions, env_name):
    plt.rcParams['text.usetex'] = True
    for dim_0 in mi_dimensions:
        for dim_1 in mi_dimensions:
            if dim_0 < dim_1:
                # Make an iterable color palette for different skills
                number_of_skills = len(scatter_skills[dim_0])

                fig, ax  = plt.subplots()
                x = []
                y = []
                c = []
                for i in range(number_of_skills):
                    x.append(scatter_skills[dim_0][i])
                    y.append(scatter_skills[dim_1][i])
                    c.append([i] * len(scatter_skills[dim_0][i]))
                pc = ax.scatter(x, y, c = c, s =0.25, cmap = 'rainbow')
                ax.set_xlabel(f'{dim_0}', fontsize= 21)
                ax.set_ylabel(f'{dim_1}', fontsize= 21)
                ax.set_title(f'{env_name}',fontsize= 25)
                freq = np.maximum(number_of_skills // 5,1)
                ticks = np.arange(0, number_of_skills, freq)
                ticks = np.append(ticks, number_of_skills-1)
                cbar = fig.colorbar(pc, ax = ax, ticks = ticks)
                cbar.ax.tick_params(labelsize=14) 
                cbar.set_label('skill', fontsize = 17)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                
                fig.savefig(f'plot/mutual_information/scatter_plots/' + 
                            f'{dim_0}_{dim_1}_{number_of_skills}_skills',
                            bbox_inches='tight')
                plt.clf()    


# plot the MI plot for skills 0 to i for all i <= iteration. Additionally 
# plot how the MI changes with number of dimensions and create a scatter plot 
# of the dimensions given in the config file.
def plot_MI_and_scatter(iteration):
    trajectories = []
    for i in range(iteration):
        # Ignore the trajectories with original reward for the MI computation.
        if i in Configuration.original_reward_generations:
            continue
        file_path = f'data/trajectories/{i}.states'
        file = open(file_path, 'rb')
        trajectory = pickle.load(file)
        file.close()
        trajectories.append(trajectory)

    env_name = Configuration.train_env_name
    mi_dimensions = Configuration.mi_dimensions[env_name]
    
    # To save the trajectory in the two dimensions for the scatter plot.
    scatter_skills = {}

    for dim_name, dim in mi_dimensions.items():
        skills = []
        # To plot how the MI changes over time.
        mi_list = []
        mi_subsampling_list = []
        for i in range(len(trajectories)):
            # Get the second part of the observations of each skill 
            # corresponding to the cut specified in the config file.
            observations = [trajectories[i][j].obs[k][dim]
                            for k in range(Configuration.mi_trajectories)
                            for j in range(Configuration.mi_cut_index,
                                           len(trajectories[i]))]

            skills.append(observations)
            mi = plot_1d_MI(skills, dim_name, subsampling = False)
            mi_list.append(mi)
            mi_subsampling = plot_1d_MI(skills, dim_name, subsampling = True)
            mi_subsampling_list.append(mi_subsampling)

        plt.plot(mi_list, c = 'b')
        plt.plot(mi_subsampling_list, c  = 'r')
        plt.xlabel(f'#skills with {Configuration.mi_1d_num_buckets} buckets')
        plt.ylabel(f'MI metric for {dim_name}')
        #Add zero such that it is at the top of the folder
        plt.savefig(f'plot/mutual_information/{dim_name}/0_{dim_name}')
        plt.clf()

        file = open(f'data/MI_{dim_name}.data', 'wb')
        pickle.dump(mi_list, file)
        file.close()        
        
        # Add the trajectories for the scatter plot
        scatter_skills[dim_name] = skills

    # TODO: subsample for this
    scatter_plot_2d(scatter_skills, mi_dimensions, env_name)
    

# Needs an iteration as sys.argv[1] and experiment name as sys.argv[2]
# otherwise it doesn't work. env name in configs file also needs to match.
if __name__ == "__main__":
    import os
    import sys
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    plt.style.use('ggplot')
    iteration = int(sys.argv[1])
    experiment_name = sys.argv[2]
    os.chdir(f'experiments/{experiment_name}')
    plot_MI_and_scatter(iteration)
    

    
