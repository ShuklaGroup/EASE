"""
Utils functions for various purposes
"""
import mdtraj as md
import matplotlib.pyplot as plt
import numpy as np
import pickle


def list_to_trajs(traj_list):

    trajs = [pickle.load(open(f, 'rb')) for f in traj_list]

    return trajs

def generate_coordinates(replicates,seed):

    np.random.seed(seed)
    x_values = np.random.uniform(-1.2, -0.8, replicates)
    y_values = np.random.uniform(-0.2, 0.2, replicates)
    coordinates = np.column_stack((x_values, y_values))

    return np.array(coordinates)
