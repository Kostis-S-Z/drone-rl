import cv2
import menpo3d
import numpy as np


class CameraEnvironment:

    def __init__(self):
        mesh = menpo3d.io.import_builtin_asset('james.obj')

        view = mesh.view()

        self.scene = view.get_figure().scene
        self.camera = self.scene.camera

        # Set initial view
        self.camera.zoom(1.5)
        self.camera.azimuth(-35)
        self.camera.elevation(-35)

        self.horizontal_degree = 0
        self.vertical_degree = 0
        self.reward = 0

        # pitch, yaw
        view.render()

    def init_environment(self, horizontal_start=None, vertical_start=None):
        """
        Initializes the environment and sets the camera to a random position
        unless explicit coordinates are given
        :return:
        """
        # Reset environment
        self.camera.azimuth(-self.horizontal_degree)
        self.camera.elevation(-self.vertical_degree)

        self.horizontal_degree = 0
        self.vertical_degree = 0
        self.reward = 0

        self.scene.render()
        # Start with a random position
        if horizontal_start is None:
            self.horizontal_degree = np.random.randint(-30, 30)
            self.vertical_degree = np.random.randint(-15, 15)
        else:
            self.horizontal_degree = horizontal_start
            self.vertical_degree = vertical_start

        self.camera.azimuth(self.horizontal_degree)
        self.camera.elevation(self.vertical_degree)

    def get_camera_pixels(self, filename=None, resize=(64, 64)):
        """
        Returns the view of the camera
        :param filename: optional, to save the camera view into a file
        :param resize:
        :return:
        """
        from mayavi import mlab

        cur_img = mlab.screenshot()
        cur_img = np.asarray(cur_img, dtype='uint8')
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)

        if resize is not None:
            cur_img = cv2.resize(cur_img, resize)

        if filename:
            cv2.imwrite(filename, cur_img)

        cur_img = cur_img/255.0
        return cur_img.reshape(1, 64, 64, 3)

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

        self.camera.azimuth(degrees)
        self.scene.render()

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

        self.camera.elevation(degrees)
        self.scene.render()

    def get_reward(self):
        """
        Evaluates the environment status (1.0 is the max reward when the face is correctly centered)
        :return:
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
