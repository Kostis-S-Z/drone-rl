import numpy as np
import cv2
import pickle


class CameraEnvironment:

    def __init__(self):
        """ input data must be in a form of dictionary """
        self.data = pickle.load(open("data.pkl", "rb"), encoding='latin1')

        for key, value in self.data.items():
            self.data[key] = value/255.0

        self.horizontal_degree = 0
        self.vertical_degree = 0
        self.reward = 0

    def init_environment(self, horizontal_start=None, vertical_start=None):
        """
        Initializes the environment and sets the camera to a random position
        :return:
        """
        self.reward = 0

        if horizontal_start is None:
            self.horizontal_degree = np.random.randint(-90, 90)
            self.vertical_degree = np.random.randint(-60, 60)
        else:
            self.horizontal_degree = horizontal_start
            self.vertical_degree = vertical_start

    def get_camera_pixels(self, view=False):
        """
        Returns the view of the camera in an array shape
        and if view=True then renders it with OpenCV
        """
        image = self.data[(self.horizontal_degree, self.vertical_degree)]

        if view:
            cv2.imshow('Image', image)

        return image.reshape((1, 64, 64, 3))

    def get_hor_degrees(self):
        """ Return the horizontal position. """
        return self.horizontal_degree

    def get_ver_degrees(self):
        """ Return the vertical position. """
        return self.vertical_degree

    def step_environment(self, horizontal_degree, vertical_degree):
        self.move_horizontal(horizontal_degree)
        self.move_vertical(vertical_degree)

    def move_horizontal(self, degrees):
        """
        Moves the camera horizontally by degrees
        :param degrees:
        :return:
        """
        if degrees > 0:
            degrees = min(degrees, 90 - self.horizontal_degree)
        else:
            degrees = max(degrees, -90 - self.horizontal_degree)
        self.horizontal_degree += degrees

    def move_vertical(self, degrees):
        """
        Moves the camera vertically by degrees
        :param degrees:
        :return:
        """
        if degrees > 0:
            degrees = min(degrees, 60 - self.vertical_degree)
        else:
            degrees = max(degrees, -60 - self.vertical_degree)
        self.vertical_degree += degrees

    def get_reward(self):
        """
        Evaluates the environment status (1.0 is the max reward when the face is correctly centered)
        :return:
        reward = 0.5 / (self.horizontal_degree**2 + 1.0)
        reward += 0.5 / (self.vertical_degree**2 + 1.0)
        return reward
        """
        # assess = "Wrong"  # if move was to the right or wrong direction
        # prev_reward = self.reward  # the reward of the previous state

        horizontal_reward = (np.abs(self.horizontal_degree / 90.0) - 1.0) ** 2
        vertical_reward = (np.abs(self.vertical_degree / 60.0) - 1.0) ** 2
        reward = (horizontal_reward + vertical_reward) / 2.0

        # if (reward > prev_reward) : assess = "Right"

        # boost = ((reward - prev_reward)) * (1.0 - reward)
        # self.reward = reward + boost
        self.reward = reward

        if reward >= 1:
            self.reward += 0.1  # give a motive to stay centered

        # print("previous: {}  |  reward: {}   | assess: {}  | boost: {}".format(prev_reward,
        # self.reward, assess, boost))
        return self.reward
