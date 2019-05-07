# Keras imports
from keras.models import Sequential
from keras.layers.core import Flatten
from keras.layers.core import Dense

# Modules import
from rendering_env import CameraEnvironment as MayaviEnv
from dataset_env import CameraEnvironment as DataEnv
from experience_replay import ExperienceReplay

# Other Imports
import json
import time
import math
import numpy as np

# Initialize Environment
env_type = 'dataset'  # 'mayavi' , 'dataset'

trained_model = None  # Change None to path if you want to load trained weights

# parameters
learning_rate = 0.001

# slack = 0.05  # leave a margin for error in order to speed up the epochs
# slack_decay_rate = 0.01  # the higher the more hard for the model to succeed

exp_decay_rate = 0.00005  # the higher the faster the model will take actions on its own

num_of_steps = 50  # number of steps to take in each epoch

num_actions = 5  # move +-vertical / +-horizontal, stay still
deg_mv = 5  # movement by degrees
epochs = 100  # episodes
max_memory = 1000  # number of observations that can be stored in agent's memory
hidden_size = 120  # number of neurons
batch_size = 100  # number of images to process in each epoch
grid_size = 64  # number of pixels of image (64x64)


def run():
    if env_type == 'dataset':
        env = DataEnv()  # with the dataset
    else:
        env = MayaviEnv()  # with mayavi

    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(grid_size, grid_size, 3), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Flatten())
    model.add(Dense(num_actions))
    model.compile(loss="mean_squared_error", optimizer='adam')

    # if you want to continue training another model
    if trained_model is not None:
        model.load_weights(trained_model)

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory)

    # Train
    start = time.clock()
    max_reward = 0
    explore = 1  # exploration start as 1 and decay after time

    for i in range(0, epochs):
        loss = 0.
        done = False  # face is centered or out of bounds

        env.init_environment()

        # if (explore > 0.2 and i%25==0):
        #    explore -= 0.1
        explore = explore * math.exp(-(exp_decay_rate * i))

        # get initial input
        input_t = env.get_camera_pixels()
        steps = 0
        epoch_reward = 0
        while not done and steps < num_of_steps:
            steps += 1
            input_tm1 = input_t
            # get next action
            # with a small possibility (eg. 0.1) do a random action in order to explore
            if np.random.rand() <= explore:
                action = np.random.randint(0, num_actions, size=1)
            else:
                q = model.predict(input_tm1)  # find the q-values for the actions
                action = np.argmax(q[0])  # choose the action with the highest q value

            # apply action, get rewards and new state

            if action == 0:  # 0-1 = vertical, 2-3 = horizontal, 4 = stay still
                env.move_vertical(deg_mv)
            elif action == 1:
                env.move_vertical(-deg_mv)
            elif action == 2:
                env.move_horizontal(deg_mv)
            elif action == 3:
                env.move_horizontal(-deg_mv)
            else:
                # stay still
                pass

            reward = env.get_reward()
            epoch_reward += reward

            # print("action: {}   |  reward: {}   |  explore: {}".format(action, reward, explore))

            input_t = env.get_camera_pixels()

            # store experience
            exp_replay.remember([input_tm1, action, reward, input_t], done)

            # calculate targets
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)
            # adapt model
            loss += model.train_on_batch(inputs, targets)

        if epoch_reward > max_reward:
            max_reward = epoch_reward

        # print("Epoch {} | Explore: {} | Loss {:.4f} | Reward {:.2f}".format(i, explore, loss, epoch_reward))
        if i % 10 == 0:
            stop_ep = time.clock()
            print("Epoch {}/{}| Explore: {} | Loss {:.4f} | Max Reward: {:.2f}/70.40".format(i, epochs, explore, loss,
                                                                                             max_reward))
            t_el = stop_ep - start
            print("     Time elapsed: {:.2f}".format(t_el))

            if i % 200 == 0:
                # Save trained model weights and architecture, this will be used by the visualization code
                model.save_weights("model_of_{}_epochs.h5".format(i), overwrite=True)
                with open("model_of_{}_epochs.json".format(i), "w") as outfile:
                    json.dump(model.to_json(), outfile)

    stop = time.clock()
    print("Comlpeted in: {:.2f} seconds".format(stop - start))
    print("Saving model...")

    # Save trained model weights and architecture, this will be used by the visualization code
    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)


if __name__ == '__main__':
    print("Initializing environment...")
    run()
