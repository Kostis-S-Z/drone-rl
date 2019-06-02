from rendering_env import CameraEnvironment

import time

import json
import numpy as np
from keras.models import model_from_json


def get_input():
    """ Get camera image, scale to [0,1] and reshape array."""
    input_t = env.get_camera_pixels()
    input_t = input_t/255.0
    input_t = input_t.reshape((1, 64, 64, 3))
    return input_t


with open("model.json", "r") as jfile:
    model = model_from_json(json.load(jfile))
model.load_weights("model.h5")
model.compile("sgd", "mse")


# Initialize environment
env = CameraEnvironment()

num_of_steps = 100  # number of steps to take in each epoch

num_actions = 5  # move +-vertical / +-horizontal, stay still
deg_mv = 1 # movement by degrees
epochs = 100

min_steps = 50  # the least amount of steps needed to center the face
# in order to truly evaluate how fast the model goes to the best solution
# we can compare it to a pathfinding algorithm
# the best solution would be X number of steps
# where X = starting_horizontal_degree/deg_mv + starting_vertical_degree/deg_mv

start = time.clock()
for i in range(1, epochs):
    env.init_environment()
    # get initial input
    input_t = get_input()

    best_solution = np.abs(env.get_hor_degrees()) + np.abs(env.get_ver_degrees())  # if deg_mv is 1

    centered_in = 0
    for steps in range(0, num_of_steps):
        input_tm1 = input_t
        # get next action

        q = model.predict(input_tm1)  # find the q-values for the actions
        action = np.argmax(q[0])  # choose the action with the highest q value

        # apply action, get rewards and new state

        if action == 0:  # 0-1 = vertical, 2-3 = horizontal, 4 = stay still
            env.move_vertical(deg_mv)
        elif action == 1:
            env.move_vertical(-deg_mv)
        elif action == 2:
            env. move_horizontal(deg_mv)
        elif action == 3:
            env.move_horizontal(-deg_mv)
        else:
            # stay still
            pass

        reward = env.get_reward()

        if reward >= 1:
            centered_in = steps
            break

        input_t = get_input()

    print("Epoch {}/{}".format(i, epochs))
    print(" Face centered in {} steps.".format(centered_in))

stop = time.clock()
print("Comlpeted in: {:.2f} seconds".format(stop-start))
