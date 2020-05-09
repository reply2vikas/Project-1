# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

import cv2
from matplotlib import cm
import scipy.ndimage

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty,BoundedNumericProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture

# Importing the Dqn object from our AI in ai.py
from ai import TD3

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = TD3((1,40,40),1,10)
# action2rotation = [0,5,-5]
last_reward = 0
reward = 0
# scores = []
im = CoreImage("./images/MASK1.png")

# textureMask = CoreImage(source="./kivytest/simplemask1.png")


# Initializing the map
first_update = True
def init():
    global sand
    global img
    global car_img
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros((longueur,largeur));print("---",sand.shape)
    img = PILImage.open("./images/mask.png").convert('L')
    # img = img.rotate(90, PILImage.NEAREST, expand = 1)
    car_img = PILImage.open("./images/car.png")
    car_img = car_img.resize((20,10),PILImage.ANTIALIAS)
    sand = np.asarray(img)/255
    print("sand",sand.shape)
    goal_x = 1420
    goal_y = 622
    first_update = False
    global swap
    swap = 0
    global Done
    Done = 0
    global count 
    count = 0
    global episode_timesteps
    episode_timesteps = 0


# Initializing the last distance
last_distance = 0

# Creating the car class

class Car(Widget):
    
    angle = BoundedNumericProperty(0)
    rotation = BoundedNumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation

# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):

        global brain
        global reward
        global last_reward
        # global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global swap
        global Done
        global count
        global episode_timesteps
        

        longueur = self.width
        largeur = self.height
        if first_update:
            init()

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y

        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.


        img = PILImage.open("./images/MASK1.png").convert('L')

        car_in_img = img
        
        # Rotate the car located in the map to alighn it to the orientation of goal
        car_img1 =  car_img.rotate(self.car.angle, PILImage.NEAREST, expand = 1)

        # Paste car image in the black & weight map it its location
        car_in_img.paste(car_img1, (int(self.car.x), 660-int(self.car.y)), car_img1)
        
        # img_citymap.paste(car_img, (int(self.car.x), int(self.car.y)), car_img)
        img_car_sand = np.asarray(car_in_img)/255

        # Take a patch of 60x60 fromt he black and wight map image with car pasted in its place
        image = img_car_sand[660-int(self.car.y)-30:660-int(self.car.y)+30, int(self.car.x)-30:int(self.car.x)+30]
        
        
        # Resize image from 60x60 to 40x40
        image = cv2.resize(image, dsize=(40, 40), interpolation=cv2.INTER_CUBIC)
        # print("image patch",image.shape)

        # image_patch = PILImage.fromarray(np.uint8(cm.gist_earth(image)*255))
        
        # Save images of the entire map and the resized patches that will go into the CNN
        # this is just for visualization purpose and hense commented
        # car_in_img.save("./img/patch-{}.png".format(count),"PNG")
        # image_patch.save("./img_patch/patch-{}.png".format(count),"PNG")

        

        count += 1
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        last_signal = image
        last_signal1 = [distance, orientation, -orientation]
        action, episode_timesteps = brain.update(reward, last_signal,last_signal1, Done, count, episode_timesteps)
        Done = 0
        # scores.append(brain.score())
        rotation = action
        self.car.move(rotation)
        
        if sand[int(self.car.x), int(self.car.y)] > 0:
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
            print(1, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            if distance < last_distance:
                reward = -0.8
            else:
                reward = -1
        else: # otherwise
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            reward = last_reward + 0.2
            print(0, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            if distance < last_distance:
                reward = last_reward +(0.5)
            else:
                reward = last_reward +(-0.5)

        if self.car.x < 40:
            # print("RAN INTO A MAP EDGE")
            # self.car.x = 30
            self.car.x = randint(100, self.width)
            self.car.y = randint(100, self.height)
            reward = -10
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            # Done = 1
        if self.car.x > self.width - 40:
            # print("RAN INTO A MAP EDGE")
            # self.car.x = self.width - 30
            self.car.x = randint(100, self.width)
            self.car.y = randint(100, self.height)
            reward = -10
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            # Done = 1
        if self.car.y < 40:
            # print("RAN INTO A MAP EDGE")
            # self.car.y = 30
            self.car.x = randint(100, self.width)
            self.car.y = randint(100, self.height)
            reward = -10
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            # Done = 1
        if self.car.y > self.height - 40:
            # print("RAN INTO A MAP EDGE")
            # self.car.y = self.height - 30
            self.car.x = randint(100, self.width)
            self.car.y = randint(100, self.height)
            reward = -10
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            # Done = 1

        # If the car does not reach the distination or crash into the map walls till 3000 steps
        if episode_timesteps > 2500:
            Done = 1
            # episode_timesteps = 1000
            reward = -10
            print("RAN TOO LONG WITH OUT REACHING GOAL")
            self.car.x = randint(100, self.width)
            self.car.y = randint(100, self.height)

        # A = (1420,622)  b = (9,85)  C = (580,530) D = (780,360) 
        # E = (1100,310) F = (115,450) G = (1050,600)
        if distance < 50:
            Done = 1
            reward = 10
            print("GOAL x-{}, y-{} REACH".format(self.car.x,self.car.y))
            self.car.x = randint(100, self.width)
            self.car.y = randint(100, self.height)
            if swap == 6:#A
                print("#############")
                print("TARGET --- A")
                print("#############")
                goal_x = 1420
                goal_y = 622
                swap = 5
            elif swap == 5:#C
                print("#############")
                print("TARGET --- C")
                print("#############")
                goal_x = 580
                goal_y = 530
                swap = 4
            elif swap == 4:#F
                print("#############")
                print("TARGET --- F")
                print("#############")
                goal_x = 115
                goal_y = 450
                swap = 0
            else:#D
                print("#############")
                print("TARGET --- D")
                print("#############")
                goal_x = 780
                goal_y = 360
                swap = 6
        last_distance = distance
        last_reward = reward

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
        Clock.schedule_interval(parent.update, 1.0/60.0)
        return parent

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
