import numpy as np

def set_random_drone_position():
    xy = 200 * np.random.random_sample(size=(2, 1)) - 100
    z = 10 + 4 * np.random.random_sample(size=(1, 1)) - 2
    
    drone_pos = np.concatenate((xy, z), axis=0)
    return drone_pos

def set_random_drone_velocity():
    vel = 10 * np.random.random_sample(size=(3, 1)) - 5
    return vel

def set_random_drone_angle():
    ang = np.pi / 2 * np.random.random_sample(size=(3, 1))
    return ang

def set_random_drone_angular_velocity():
    angv = 4 * np.random.random_sample(size=(3, 1)) - 2
    return angv

def set_random_coin_position():
    xy = 200 * np.random.random_sample(size=(2, 1)) - 100
    z = 10 + 4 * np.random.random_sample(size=(1, 1)) - 2

    coin_pos = np.concatenate((xy, z), axis=0)
    return coin_pos