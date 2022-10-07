# Open-Ended-Reinforcement-Learning-with-Neural-Reward-Functions
**Robert Meier*** and **Asier Mujika***

<sub><sup>*Equal contribution</sup></sub>


NeurIPS 2022, 
[[arXiv]](https://arxiv.org/abs/2202.08266)

![cheetah](https://user-images.githubusercontent.com/1566072/194556860-f598a6ef-0c41-4d53-ab0c-a345a84d754c.gif)
![ant](https://user-images.githubusercontent.com/1566072/194556872-5a328065-9cb5-4968-8d5d-e68b5fcb80a9.gif)
![human_back](https://user-images.githubusercontent.com/1566072/194556952-2e6f17ad-c7d7-4d7b-99f6-f1ee25d490e7.gif)
![human_one_leg](https://user-images.githubusercontent.com/1566072/194556916-fbc91531-418e-4772-938a-930bd66fb025.gif)
![montzeuma](https://user-images.githubusercontent.com/1566072/194556052-82802114-e6b7-4033-a16d-2ecf0923ad9a.gif)


Each folder contains the code for a set of experiments from Section 4 of the paper.
You can run the following commands from the corresponding folder to reproduce the results.

#### 2d Navigation

`python3 main.py {experiment_name}` where {experiment_name} is the folder that will be used to store the experiment results.

#### Robotic Environments
`python3 run_experiment.py {gpu_id}` where {gpu_id} is a single gpu id to tell CUDA which device to use. 

#### Montezumas Revenge
`python3 main.py`
