TD3 Implementation on RL Environment   
Final Goal

To make a car learn to drive on the roads of a map (RL environment) using TD3 theory.
Video - https://youtu.be/pigvys7b_Xw
What was the Approach-
These are the steps

    Understood the TD3 architecture and the way to implement it for the environments - AntBulletEnv-v0, Walker2DBulletEnv-v0 and HalfCheetahBulletEnv-v0.

    Printed the different environment variables like Action Space, Observation Space - their low, high and sample values.

    print(env.observation_space.high) - [inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf]

    print(env.observation_space.low) - [-inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf]

    print(env.action_space.high) - [1. 1. 1. 1. 1. 1.]

    print(env.action_space.low) - [-1. -1. -1. -1. -1. -1.]

    env.env.action_space.sample() - array([-0.58517057, -0.5024034 , -0.6100568 , -0.88058037, -0.14023775, -0.7811404 ], dtype=float32)

    env.observation_space.sample() - array([ 0.2560038 , 1.2759786 , 0.5100617 , 0.6512928 , 1.6200854 , -0.28926647, 0.06890053, -1.6094683 , -0.9811666 , 0.12742993, 1.3960222 , -0.04613981, -0.8242046 , -1.4867542 , 0.11702857, -0.76355684, 0.7695135 , 0.3610245 , 1.0636773 , -1.0533894 , 0.5983344 , -0.1122881 ], dtype=float32)

    Explored the available Github of pybullet to understand these variables - https://github.com/bulletphysics/bullet3/blob/ec2b6dd920135a5df804d521727cc06446a6a3bd/examples/pybullet/gym/pybullet_envs/robot_locomotors.py#L103

    Went through Github of CarRacing-v0 and Atari environments to understand how to implement TD3 when image screenshots of the environment are required.

    Understood the "step" function of the environment.

Steps done for EndGame assignment

    Keeping the sensors as-is, changed the DQN architecture to TD3 architecture (without CNN. Similar to "walker" environment) to make it work.
    After Step 1 was completed, implemented TD3 with a cropped image as input to Actor network. This cropped image did not have car image embedded into it. I made this work first. Parameters in this approach were as below:
        State - Cropped image of size 80x80 reduced to 32x32 for the CNN.
        Replay Buffer
            State (cropped image without car)
            Next State (cropped image from sand image from the car's new position based on the action taken)
            Reward for the action taken
            Action taken
            Done (defined it to True when the car hits the walls or when it reaches both the targets or when the episode is completed)
    After Step 2 was completed, improvised the image by embedding the car into it taking its angle into consideration, but the image of the car was surrounded by a rectangle around it! Continued with the same Replay Buffer and same parameters. Modified parameters were as below:
        State - Cropped image of size 160x160 along with car placed considering its angle reduced to 32x32 for the CNN.
        Replay Buffer
            State (cropped image with car)
            Next State (cropped image from sand image from the car's new position based on the action taken)
            Reward for the action taken
            Action taken
            Done (defined it to True when the car hits the walls or when it reaches both the targets or when the episode is completed)
    Added "Orientation (-Orientation, Orientation)" parameters as additional states and removed car image from the cropped image instead rotating the cropped image based on car's angle. Modified parameters were as below:
        States - Cropped Image rotated in car's angle, -Orientation, Orientation
        Replay Buffer
            States (cropped image rotated in car's angle, -orientation, orientation)
            Next States (cropped image rotated in car's angle from the car's new position and angle, -orientation, orientation based on the action taken)
            Reward for the action taken based on the "Distance"
            Action taken
            Done (defined it to True when the car hits the walls or when it reaches both the targets or when the episode is completed)
    Added "Distance" parameter as an additional state but this resulted in rotation issue. I tried to fix the rewards, crop size, max_action but nothing worked for me. Modified parameters are as below:
        States - Cropped Image rotated in car's angle, -Orientation, Orientation, Distance
        Replay Buffer
            States (cropped image rotated in car's angle, -Orientation, Orientation, Distance)
            Next States (cropped image rotated in car's angle, -Orientation, Orientation, Distance based on the action taken)
            Reward for the action taken based on the "Distance"
            Action taken
            Done (defined it to True when the car hit the walls or when it reached target or when the episode was completed)
    Continued to consider only 3 states for my implementation - Cropped Image rotated in car's angle, -Orientation, Orientation.
    Improvised the Rewards strategy.
        As I observed that when car gets stuck at the walls, it was not able to come out of that state. To overcome this scenario, I am checking for the episode timesteps and if the episode goes to "Done" within 5 timesteps, I am giving high penalty.
        As car mostly goes on sand and less on roads, I am maintaining a counter on how many timesteps the car has gone on sand and penalizing if the car is continuously on sand or is on sand within the initial timesteps of an episode. Also, ending the episode when the car stays on sand for certain number of timesteps continuously.
    Tuned hyperparameters like Learning Rate, max_action, crop size, CNN embedding dimension.
    Every time Done becomes true, I am resetting the car's position randomly.
    Saving the models when it is Done (=True) and when Episode reward is positive.
    Implemented the inferencing using best positive episode reward models.
