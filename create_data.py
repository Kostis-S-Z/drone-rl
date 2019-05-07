from rendering_env import CameraEnvironment

import numpy as np
import time

deg_mv = 1  # accuracy, by how many degrees to save

save = 1  # 0:json, 1:pickle, 2:csv, 3:txt, 4:numpy


def generate_dataset():
    env = CameraEnvironment()  # initialize mayavi env

    # set environment to the boundaries
    env.init_environment(-90, -60)

    data = {}  # data will be saved in a dictionary{coordinates:values}

    start = time.clock()
    # for every possible horizontal degree
    for hor in range(-90, 91):
        # for every possible vertical degree
        for vert in range(-60, 61):
            # save the image in unsigned integer of 8 bits (0-255)
            value = np.uint8(env.get_camera_pixels())
            data[(hor, vert)] = value
            # make a vertical step
            env.move_vertical(deg_mv)
        print("{}/180".format(hor))
        # make a horizontal step
        env.move_horizontal(deg_mv)
    stop = time.clock()

    # It should take around 500seconds.
    print("Done creating data in {:.2f} seconds. Now saving...").format(stop-start)

    save_data(data)


def save_data(data):
    # Multiple methods of Saving
    if save == 0:
        """ JSON """
        import json

        with open("data.json", "wb") as f:
            json.dump(data, f)
            f.close()

    elif save == 1:
        """ PICKLE """
        import pickle

        with open("data.pkl", "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()

    elif save == 2:
        """ CSV """
        import csv

        w = csv.writer(open("data.csv", "w"))
        for key, val in data.items():
            w.writerow([key, val])

    elif save == 3:
        """ TXT """
        with open("data.txt", "w") as f:
            f.write(str(data))
            f.close()

    else:
        """ NUMPY """
        np.save('data.npy', data)

    print("Done!")


if __name__ == '__main__':
    generate_dataset()
