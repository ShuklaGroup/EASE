"""
Utils functions for various purposes
"""
import mdtraj as md
import matplotlib.pyplot as plt
import numpy as np
import pickle


def load(traj,top):

    return md.load(traj,top=top)

