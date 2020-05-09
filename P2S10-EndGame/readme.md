Problem statement:

On a map environment of city we have to run the car(agent) from one location(goal) to the other.
Using TD3 algorithm to train the continuous policy estimation model.
Extract patch which contains car and its surroundings, can be used as state and use CNN based neural network model for actor-critic networks.
"img_map_car.png" image above is the sample image of the map with car image pasted on it.

"patch_4040.png" is the patch of image resized to 40x40 which also contains the car corrected with orientation, will be sent to the CNN models as part of state information.

Changes made from the Session-7 Assginment:

1. Replaced dqn with TD3: TD3 is policy estimation algorith, which contains 6 NN models in total. Two actor models -- actor model, actor target Four critic models -- two critic models, two critic targets.

*three model networks take input from the curret state(s) and its action(a). *three target networks take input from the nest state(s') and next action(a'). For detailed theoritical understanding with code please refer to this link https://github.com/reply2vikas/2018-cycle-2/tree/master/P2S9.

2. State space: For state input we are using an 60x60 patch of sand image around the current car location. Resized it to 40x40 using cv2. These patches are stored into the Replay Buffer and subsiquently sampled in batches to be used for training the TD3 models. Along with Image, car orientation (two values -ve and +ve) values are sent as state information. In total state space, representing each state is one image of dimentions 60x60 resized to 40x40, and a vector a two values representing car orientation.

3. Action space: Since the TD3 allowes us the flexibility of continuous action spaces. We are using one output dimention to represent the angle of car. This single output value estimates values in the range of -5 to 5, which is also the range of car rotation from left extream to right extream. This is seen as a regresion problem on the side of actor model.

4. Episode termination states: On reaching goal(with +ver reward) or the boundery of the map(with -ve reward) the episode termination step is reached. Episode also ends if the car does not reach the map wall or goal in 2000 steps, here -ve reward is given. In total I have used 7 gols which will be targetted one after the other:

    # A = (1420,622)  b = (9,85)  C = (580,530) D = (780,360) 
    # E = (1100,310) F = (115,450) G = (1050,600)
After the end of episode is reached variable Done = 1, and the the car is reset to a random location in the map. Also after teh end of episode the training starts for the number of steps equal to the number of step in the previous episode.

Since I am running first 2000 steps with uniform random sampling from the range of -5 to 5 as action to be sent to the car. First few episodes will perform random exloration. Subsiquently the actions to move the car will come from policy model(model actor). Even here I am introducing 20% random action being seleted for some exploration scope.

5.Neural network model for actor-critic: Since our input is state is an image I have used CNN based model for both actor and critic models. To feed both state(s) and action(a) together to the critic models is tricky because concatinating two tensors of different dimentions is not possible. And they are also of different feature type. For this to work, passed the image through the cnn and after the flatten layer the output is 1-dimentional feature vector, this is concatinated to the action value and followed up with two more fully connected layers.

CURRENT STATUS

I have included td3 into the car-map environment with out any issue. Everything is properly integrated.

But the policy netork does not estimate good actions, in a way it is giving similar values every time. Resulting in car taking circular turns in the same location.

I am using a CPU based pc with a ram size of 4GB and dual cpu (celeron Laptop) cores.

I tried my level best by integrted everything into code linked here for the TD3.
