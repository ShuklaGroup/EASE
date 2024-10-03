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


