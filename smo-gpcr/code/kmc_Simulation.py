"""
Definition of a simulation class 
"""
import numpy as np
import os
import deeptime as dt
import scipy
from scipy import interpolate
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
from deeptime.data import triple_well_2d
from matplotlib.collections import LineCollection
import pickle
from tqdm import tqdm
from deeptime.clustering import KMeans



class kMC_simulation:
    """
    Kinetic Monte Carlo Simulation object for villin
    """
    def __init__(self,round_no,root_path,msm_path):
        """Constructor.

        :param round_no: int.
            Round number of adaptive sampling. 
        :param root_path: str.
            Path to save round output
        """

        self.round_no = round_no
        self.root_path = root_path
        self.msm_path = msm_path


    def get_initial_data(self, mc_steps,start_states, num_reps, seed):
        """Generates initial (round=0) data.


        :param mc_steps: int.
            Number of steps.
        :param start_state = int
            state id to seed from
        :param num_reps = int, default = 1
            Number of replicates to run 
        :param seed = int, default = 42
            Seed to fix random parameters

        :return: list.
            List of pkls files of the trajectories.
        """

        traj_list = []
        my_mod = pickle.load(open(self.msm_path,"rb"))
        
        for start_state, rep in tqdm(zip(start_states, range(num_reps))):
	        traj = my_mod.simulate(n_steps=mc_steps,start=start_state,seed=seed)
	        path = f'{self.root_path}/round{self.round_no}/trajs/rep{rep}.out'
	        os.makedirs(f'{self.root_path}/round{self.round_no}/trajs/', exist_ok=True)
	        pickle.dump(traj,open(path,'wb'))
	        traj_list.append(path)

        return traj_list


