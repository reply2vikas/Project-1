# TD3 Implementation on Custom Environment (RL)

## 

## The Final Goal

To make a car learn to drive on the roads of a map (custom environment) using **TD3** architecture.

### 

### Video - https://youtu.be/pigvys7b_Xw

## 

## Approach taken for game-

### 

### Steps taken before ENDGAME assignment-

1. Understood the TD3 architecture and the way to implement it for the  environments - AntBulletEnv-v0, Walker2DBulletEnv-v0 and  HalfCheetahBulletEnv-v0.

2. Printed the different environment variables like Action Space, Observation Space - their low, high and sample values.

   print(env.observation_space.high) - [inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf]

   print(env.observation_space.low) - [-inf -inf -inf -inf -inf -inf  -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf  -inf -inf]

   print(env.action_space.high) - [1. 1. 1. 1. 1. 1.]

   print(env.action_space.low) - [-1. -1. -1. -1. -1. -1.]

   env.env.action_space.sample() - array([-0.58517057, -0.5024034 ,  -0.6100568 , -0.88058037, -0.14023775,       -0.7811404 ],  dtype=float32)

   env.observation_space.sample() - array([ 0.2560038 ,  1.2759786 ,   0.5100617 ,  0.6512928 ,  1.6200854 ,       -0.28926647,  0.06890053,  -1.6094683 , -0.9811666 ,  0.12742993,        1.3960222 , -0.04613981,  -0.8242046 , -1.4867542 ,  0.11702857,       -0.76355684,  0.7695135 ,   0.3610245 ,  1.0636773 , -1.0533894 ,        0.5983344 , -0.1122881 ],  dtype=float32)

3. Explored the available Github of pybullet to understand these variables - 

   https://github.com/bulletphysics/bullet3/blob/ec2b6dd920135a5df804d521727cc06446a6a3bd/examples/pybullet/gym/pybullet_envs/robot_locomotors.py#L103

4. Went through Github of CarRacing-v0 and Atari environments to  understand how to implement TD3 when image screenshots of the  environment are required.

5. Understood almost every "step" function of the environment.

### 

### Final Steps done for  Assignment (EndGame)

1. ***Keeping the sensors as-is***, changed the DQN architecture to TD3 architecture (without CNN. Similar to "walker" environment) to make it work.

2. After Step 1 was completed, implemented TD3 with a 

   **cropped image**

    as input to Actor network. This cropped image 

   did not have car image

    embedded into it. I made this work first. Parameters in this approach were as below:

   1. State - Cropped image of size 80x80 reduced to 32x32 for the CNN.
   2. Replay Buffer
      1. State (cropped image without car)
      2. Next State (cropped image from sand image from the car's new position based on the action taken)
      3. Reward for the action taken
      4. Action taken
      5. Done (defined it to True when the car hits the walls or when it reaches both the targets or when the episode is completed)

3. After Step 2 was completed, improvised the image by 

   **embedding the car**

    into it taking its angle into consideration, but the image of the car  was surrounded by a rectangle around it! Continued with the same Replay  Buffer and same parameters. Modified parameters were as below:

   1. State - Cropped image of size 160x160 along with car placed considering its angle reduced to 32x32 for the CNN.
   2. Replay Buffer
      1. State (cropped image with car)
      2. Next State (cropped image from sand image from the car's new position based on the action taken)
      3. Reward for the action taken
      4. Action taken
      5. Done (defined it to True when the car hits the walls or when it reaches both the targets or when the episode is completed)

4. *Added* "*Orientation (-Orientation, Orientation)*"

    parameters as additional states and removed car image from the cropped  image instead rotating the cropped image based on car's angle. Modified  parameters were as below:

   1. States - Cropped Image rotated in car's angle, -Orientation, Orientation
   2. Replay Buffer
      1. States (cropped image rotated in car's angle, -orientation, orientation)
      2. Next States (cropped image rotated in car's angle from the car's new position and angle, -orientation, orientation based on the action  taken)
      3. Reward for the action taken based on the "Distance"
      4. Action taken
      5. Done (defined it to True when the car hits the walls or when it reaches both the targets or when the episode is completed)

5. **Added "Distance"**

    parameter as an  additional state but this resulted in rotation issue. I tried to fix the rewards, crop size, max_action but nothing worked for me. Modified  parameters are as below:

   1. States - Cropped Image rotated in car's angle, -Orientation, Orientation, Distance
   2. Replay Buffer
      1. States (cropped image rotated in car's angle, -Orientation, Orientation, Distance)
      2. Next States (cropped image rotated in car's angle, -Orientation, Orientation, Distance based on the action taken)
      3. Reward for the action taken based on the "Distance"
      4. Action taken
      5. Done (defined it to True when the car hit the walls or when it reached target or when the episode was completed)

6. Continued to consider only 3 states for my implementation - Cropped Image rotated in car's angle, -Orientation, Orientation.

7. Improvised the 

   **Rewards strategy**

   .

   1. As I observed that when car gets stuck at the walls, it was not able to come out of that state. To overcome this scenario, I am checking for the episode timesteps and if the episode goes to "Done" within 5  timesteps, I am giving high penalty.
   2. As car mostly goes on sand and less on roads, I am maintaining a  counter on how many timesteps the car has gone on sand and penalizing if the car is continuously on sand or is on sand within the initial  timesteps of an episode. Also, ending the episode when the car stays on  sand for certain number of timesteps continuously.

8. Tuned hyperparameters like Learning Rate, max_action, crop size, CNN embedding dimension.

9. Every time Done becomes true, I am ***resetting the car's position randomly***.

10. Saving the models when it is Done (=True) and when Episode reward is positive.

11. Implemented the inferencing using best positive episode reward models.

12. That`s ALL.
