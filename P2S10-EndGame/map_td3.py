# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import time
import os
import math

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from PIL import ImageDraw
from kivy.graphics.texture import Texture

import scipy
from scipy import ndimage

# Importing the Dqn object from our AI in ai.py
from ai_td3 import TD3

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1350')
Config.set('graphics', 'height', '550')

# Implementing Experience Replay
class ReplayBuffer(object):

  def __init__(self, max_size=150000):#1e6
    self.storage = []
    self.max_size = max_size
    self.ptr = 0


  def add(self, state, new_state, orientation, new_orientation, action, reward, done_bool):
    data = (state, new_state, orientation, new_orientation, action, reward, done_bool)
    if len(self.storage) == self.max_size:
        self.storage[int(self.ptr)] = data
        self.ptr = (self.ptr + 1) % self.max_size
    else:
        self.storage.append(data)

  
  def sample(self, batch_size):
    ind = np.random.randint(0, len(self.storage), size=batch_size)
    batch_states, batch_next_states, batch_orientations, batch_next_orientations, batch_actions, batch_rewards, batch_dones = [], [], [], [], [], [], []
    for i in ind:
      state, next_state, orientation, next_orientation, action, reward, done = self.storage[i]
      batch_states.append(np.array(state, copy=False))
      batch_next_states.append(np.array(next_state, copy=False))
      batch_orientations.append(np.array(orientation, copy=False))
      batch_next_orientations.append(np.array(next_orientation, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))

    return np.array(batch_states), np.array(batch_next_states), np.array(batch_orientations), np.array(batch_next_orientations), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

seed = 0
save_models = True # Boolean checker whether or not to save the pre-trained model
torch.manual_seed(seed)
np.random.seed(seed)
action_dim = 1
state_dim = 4
max_action = 7

sand_image = np.asarray(PILImage.open("./images/MASK1.png").convert('L'))/255
crop_size = 75

croppedimage_size = 32

_max_episode_steps = 1000
episode_num = 0

# Initializing the last distance
distance = 0
orientation = 0
sand_penalty = 0
distance_reward = 0
boundary_penalty = 0
goal_reward = 0
goal_penalty = 0
nonsand_reward = 0
sand_episode_penalty = 0
sand_pointer = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = TD3(state_dim, action_dim, orientation, max_action)
replay_buffer = ReplayBuffer()
scores = []
im = CoreImage("./images/MASK1.png")

file_name = "%s_%s" % ("TD3", str(seed))
print ("---------------------------------------")
print ("Settings: %s" % (file_name))
print ("---------------------------------------")

if not os.path.exists("./results"):
    os.makedirs("./results")
if save_models and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")


# Initializing the map
first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    global orientation
    sand = np.zeros((longueur,largeur))
    img = PILImage.open("./images/mask.png").convert('L')
    sand = np.asarray(img)/255
    goal_x = 1000
    goal_y = 400
    first_update = False
    global swap
    swap = 0
    orientation = [orientation, -orientation]
    new_orientation = np.array(orientation)

# Function to return cropped image based on sand image and car position as input
def getCroppedImage(sand_image, car_x, car_y, angle, crop_size = 40, croppedimage_size = 32): 
    pad = crop_size*2
    #pad for safety
    crop1 = np.pad(sand_image, pad_width=pad, mode='constant', constant_values = 1)
    #imageio.imwrite('./images/crop1padded.png', crop1)
    centerx = car_x + pad
    centery = largeur - car_y + pad

    #first small crop
    startx, starty = int(centerx-(crop_size)), int(centery-(crop_size))
    crop1 = crop1[starty:starty+crop_size*2, startx:startx+crop_size*2]
    #imageio.imwrite('./images/firstcrop.png', crop1)

    #rotate
    crop1 = scipy.ndimage.rotate(crop1, -angle, mode='constant', cval=1.0, reshape=False, prefilter=False)
    #imageio.imwrite('./images/rotatedcrop.png', crop1)

    #again final crop
    startx, starty = int(crop1.shape[0]//2-crop_size//2), int(crop1.shape[0]//2-crop_size//2)

    im = crop1[starty:starty+crop_size, startx:startx+crop_size] #.reshape(crop_size, crop_size, 1)
    #imageio.imwrite('./images/finalcrop.png', im)

    im = torch.from_numpy(np.array(im)).float()#.div(255)

    im = im.unsqueeze(0).unsqueeze(0)

    im = F.interpolate(im,size=(croppedimage_size,croppedimage_size))

    return im


# Creating the car class

class Car(Widget):
    angle = NumericProperty(0)
    rotation = float(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)


    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation


class Game(Widget):

    car = ObjectProperty(None)

    reward_window = []

    start_timesteps = 36000 	# Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
    max_timesteps = 108000 		# Total number of iterations/timesteps
    max_episodes = 180  		# max num of episodes
    eval_episode = 1
    
    expl_noise = 0.1 			# Exploration noise - STD value of exploration Gaussian noise
    batch_size = 2 				# Size of the batch
    discount = 0.99 			# Discount factor gamma, used in the calculation of the total discounted reward
    tau = 0.005 				# Target network update rate
    policy_noise = 0.2 			# STD of Gaussian noise added to the actions for the exploration purposes
    noise_clip = 0.5 			# Maximum value of the Gaussian noise added to the actions (policy)
    policy_freq = 2 			# Number of iterations to wait before the policy network (Actor model) is updated

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_timesteps = 0
    done = True
    running_score = 0
    done_counter = 0

    """ parameters for epsilon decay """
    epsilon_start = 1
    epsilon_final = 0.01
    decay_rate = max_episodes / 50


    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)


    # Function to get orientation of the car with respect to target
    def get_orientation(self):
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        return orientation


    def reset(self):
        self.car.x = np.random.randint(50, 1400)
        self.car.y = np.random.randint(50, 600)
        self.car.angle = 0

    
    def evaluate(self):
        brain.load(file_name, './pytorch_models/')


    # Function for inferencing
    def eval_update(self, dt):
        global brain
        global reward
        global scores
        global distance
        global orientation
        global goal_x
        global goal_y
        global longueur
        global largeur
        global swap

        global _max_episode_steps
        global episode_num
        global episode_reward
        global state

        global sand_penalty
        global distance_reward
        global boundary_penalty
        global goal_reward
        global goal_penalty
        global nonsand_reward
        global sand_episode_penalty
        global sand_pointer

        longueur = self.width
        largeur = self.height
        if first_update:
            init()


        if self.total_timesteps < self.max_timesteps:
            if self.done:
                if self.total_timesteps != 0:
                    with open("./logs/Result_Evel.txt", 'a') as f:
                        sys.stdout = f
                        print("Total Steps: ", self.total_timesteps, "Episode: ",episode_num, "Reward: ", episode_reward,"Episode TimeSteps: ", self.episode_timesteps, "Sand Penalty: ", sand_penalty, "NonSand Reward: ", nonsand_reward, "Distance Reward: ", distance_reward, "Boundary Penalty: ", boundary_penalty, "Goal Reward: ", goal_reward, "Goal Penalty: ", goal_penalty)

                # Get cropped image by passing sand and car's postion as input
                state = getCroppedImage(sand_image, self.car.x, self.car.y, self.car.angle, crop_size = crop_size)

                # Set the Done to False
                self.done = False

                sand_pointer = 0

                # Set rewards and episode timesteps to zero
                episode_reward = 0
                self.episode_timesteps = 0
                episode_num += 1
                self.eval_episode += 1


            orientation = self.get_orientation()
            orientation = [orientation, -orientation]
            orientation = np.array(orientation)
            action = brain.select_action(state, orientation)

            # If the explore_noise parameter is not 0, we add noise to the action and we clip it
            if self.expl_noise != 0:
                expl_noise = (self.epsilon_start - self.epsilon_final) * math.exp(-1. * self.total_timesteps / self.decay_rate)
                action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(-max_action, max_action)

           	# Based on the action given by actor, move the car to next state
            self.car.move(action.tolist()[0])

            # Get cropped image by passing new position of the car
            next_state = getCroppedImage(sand_image, self.car.x, self.car.y, self.car.angle, crop_size = crop_size)

            new_distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)

            new_orientation = self.get_orientation()
            new_orientation = [new_orientation, -new_orientation]
            new_orientation = np.array(new_orientation)

            if sand[int(self.car.x),int(self.car.y)] > 0:
                sand_pointer += 1
                self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
                #print(1, goal_x, goal_y, new_distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
                reward = -0.5
                sand_penalty += 0.5
                self.done = False
            else: # otherwise
                self.car.velocity = Vector(2, 0).rotate(self.car.angle)
                sand_pointer = 0
                reward = 10
                nonsand_reward += 10
                self.done = False
                #print(0, goal_x, goal_y, new_distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
                if new_distance < distance:
                    reward = 20
                    distance_reward += 20

            if sand_pointer >= _max_episode_steps/2:
                reward -= 1
                sand_episode_penalty += 1
                self.done = True

            if self.episode_timesteps < 25 and sand_pointer == 1:
                reward -= 0.2

            if self.car.x < 20:
                self.car.x = 20
                reward -= 10
                boundary_penalty += 10
                self.done = True
            if self.car.x > self.width - 20:
                self.car.x = self.width - 20
                reward -= 10
                boundary_penalty += 10
                self.done = True
            if self.car.y < 10:
                self.car.y = 10
                reward -= 10
                boundary_penalty += 10
                self.done = True
            if self.car.y > self.height - 10:
                self.car.y = self.height - 10
                reward -= 10
                boundary_penalty += 10
                self.done = True

            if new_distance < 25:
                reward += 35
                goal_reward += 35
                if swap == 1:
                    with open("./logs/Result_Evel.txt", 'a') as f:
                        sys.stdout = f
                        print("Reached the target A: x: %s, y: %s"%(self.car.x, self.car.y))
                    goal_x =  1000
                    goal_y =  400
                    swap = 0
                    self.done = True
                else:
                    with open("./logs/Result_Evel.txt", 'a') as f:
                        sys.stdout = f
                        print("Reached the target B: x: %s, y: %s"%(self.car.x, self.car.y))
                    goal_x = 200
                    goal_y = 100
                    swap = 1
                    self.done = False
            else:
                reward -= 0.1
                goal_penalty += 0.1

            scores.append(brain.score())

            # We check if the episode is done
            done_bool = 1 if self.episode_timesteps + 1 == _max_episode_steps else float(self.done)
            if self.episode_timesteps + 1 == _max_episode_steps:
                self.done = True
  
            # We increase the total reward
            episode_reward += reward
  
            # We store the new transition into the Experience Replay memory (ReplayBuffer)
            replay_buffer.add(state, next_state, orientation, new_orientation, action, reward, done_bool) #distance, new_distance, 

            # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
            state = next_state
            orientation = new_orientation
            distance = new_distance
            self.episode_timesteps += 1
            self.total_timesteps += 1
            self.timesteps_since_eval += 1


    def update(self, dt):

        global brain
        global reward
        global scores
        global distance
        global orientation
        global goal_x
        global goal_y
        global longueur
        global largeur
        global swap

        global _max_episode_steps
        global episode_num
        global episode_reward
        global state

        global sand_penalty
        global distance_reward
        global boundary_penalty
        global goal_reward
        global goal_penalty
        global nonsand_reward
        global sand_episode_penalty
        global sand_pointer

        longueur = self.width
        largeur = self.height
        if first_update:
            init()

        if self.total_timesteps < self.max_timesteps:

            if self.done:
                self.done_counter += 1
                if self.total_timesteps != 0:
                    with open("./logs/Result.txt", 'a') as f:
                        sys.stdout = f
                        print("Total Steps: ", self.total_timesteps, "Episode: ",episode_num, "Reward: ", episode_reward,"Episode TimeSteps: ", self.episode_timesteps, "Sand Episode Penalty: ", sand_episode_penalty, "Sand Penalty: ", sand_penalty, "NonSand Reward: ", nonsand_reward, "Distance Reward: ", distance_reward, "Boundary Penalty: ", boundary_penalty, "Goal Reward: ", goal_reward, "Goal Penalty: ", goal_penalty)

                if self.total_timesteps > self.start_timesteps/3:
                    brain.train(replay_buffer, self.episode_timesteps)

                self.car.x = np.random.randint(50, 1400)
                self.car.y = np.random.randint(50, 600)
                self.car.angle = 0
                self.car.velocity = Vector(12, 0)

                # Get cropped image by passing sand and car's postion as input
                state = getCroppedImage(sand_image, self.car.x, self.car.y, self.car.angle, crop_size = crop_size)

                # Set the Done to False
                self.done = False

                sand_pointer = 0

                # Set rewards and episode timesteps to zero
                episode_reward = 0
                self.episode_timesteps = 0
                episode_num += 1

            orientation = self.get_orientation()
            orientation = [orientation, -orientation]
            orientation = np.array(orientation)

            # Before 10000 timesteps, we play random actions
            if self.total_timesteps < self.start_timesteps:
                action = np.random.uniform(-max_action, max_action, (1,))
            else:
                action = brain.select_action(state, orientation)
                # If the explore_noise parameter is not 0, we add noise to the action and we clip it
                if self.expl_noise != 0:
                    expl_noise = (self.epsilon_start - self.epsilon_final) * math.exp(-1. * self.total_timesteps / self.decay_rate)
                    action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(-max_action, max_action)

            # Based on the action given by actor, move the car to next state
            self.car.move(action.tolist()[0])

            # Get cropped image by passing new position of the car
            next_state = getCroppedImage(sand_image, self.car.x, self.car.y, self.car.angle, crop_size = crop_size)

            new_distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)

            new_orientation = self.get_orientation()
            new_orientation = [new_orientation, -new_orientation]
            new_orientation = np.array(new_orientation)

            if sand[int(self.car.x),int(self.car.y)] > 0:
                sand_pointer += 1
                self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
                #print(1, goal_x, goal_y, new_distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
                reward = -0.5
                sand_penalty += 0.5
                self.done = False
            else: # otherwise
                self.car.velocity = Vector(2, 0).rotate(self.car.angle)
                sand_pointer = 0
                reward = 10
                nonsand_reward += 10
                self.done = False
                #print(0, goal_x, goal_y, new_distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
                if new_distance < distance:
                    reward = 20
                    distance_reward += 20

            if sand_pointer >= _max_episode_steps/2:
                reward -= 1
                sand_episode_penalty += 1
                self.done = True

            if self.episode_timesteps < 25 and sand_pointer == 1:
                reward -= 0.2

            if self.car.x < 20:
                self.car.x = 20
                reward -= 10
                boundary_penalty += 10
                self.done = True
            if self.car.x > self.width - 20:
                self.car.x = self.width - 20
                reward -= 10
                boundary_penalty += 10
                self.done = True
            if self.car.y < 10:
                self.car.y = 10
                reward -= 10
                boundary_penalty += 10
                self.done = True
            if self.car.y > self.height - 10:
                self.car.y = self.height - 10
                reward -= 10
                boundary_penalty += 10
                self.done = True

            if new_distance < 25:
                reward += 35
                goal_reward += 35
                if swap == 1:
                    with open("./logs/Result.txt", 'a') as f:
                        sys.stdout = f
                        print("Reached the target A: x: %s, y: %s"%(self.car.x, self.car.y))
                    goal_x =  1000
                    goal_y =  400
                    swap = 0
                    self.done = True
                else:
                    with open("./logs/Result.txt", 'a') as f:
                        sys.stdout = f
                        print("Reached the target B: x: %s, y: %s"%(self.car.x, self.car.y))
                    goal_x = 200
                    goal_y = 100
                    swap = 1
                    self.done = False
            else:
                reward -= 0.1
                goal_penalty += 0.1

            scores.append(brain.score())

            # We check if the episode is done
            done_bool = 1 if self.episode_timesteps + 1 == _max_episode_steps else float(self.done)
            if self.episode_timesteps + 1 == _max_episode_steps:
                self.done = True

            if self.done and self.total_timesteps > self.start_timesteps:
                if episode_reward > 0: #and self.done_counter % 3 == 0:
                    if save_models: brain.save("%s_%s_%s" % (file_name, int(episode_reward), self.done_counter), directory="./pytorch_models")
  
            # We increase the total reward
            episode_reward += reward
            
            # We store the new transition into the Experience Replay memory (ReplayBuffer)
            replay_buffer.add(state, next_state, orientation, new_orientation, action, reward, done_bool)

            # We update the state, orientation, distance, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
            state = next_state
            orientation = new_orientation
            distance = new_distance
            self.episode_timesteps += 1
            self.total_timesteps += 1

# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1
            img = PILImage.fromarray(sand.astype("uint8")*255)
            img.save("./images/sand.jpg")

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1

            
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()

        # For Training
        #Clock.schedule_interval(parent.update, 1.0/60.0)

        # For Inferencing
        parent.evaluate()
        Clock.schedule_interval(parent.eval_update, 1.0/60.0)

        self.painter = MyPaintWidget()
        # clearbtn = Button(text = 'clear')
        # savebtn = Button(text = 'save', pos = (parent.width, 0))
        # loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        # clearbtn.bind(on_release = self.clear_canvas)
        # savebtn.bind(on_release = self.save)
        # loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        # parent.add_widget(clearbtn)
        # parent.add_widget(savebtn)
        # parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()