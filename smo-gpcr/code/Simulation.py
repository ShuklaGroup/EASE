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



class OSSimulation:
    """
    Kinetic Monte Carlo Simulation object for villin
    """
    def __init__(self,root_path,round_no=0):
        """Constructor.

        :param round_no: int.
            Round number of adaptive sampling. 
        :param root_path: str.
            Path to save round output
        """

        self.round_no = round_no
        self.root_path = root_path


    def get_initial_data(self, mc_steps, num_reps,start_state=20, seed=42):
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
        start_state = np.random.randint(low=1, high=149) 
        traj_list = []
        my_mod = pickle.load(open(f"ground_truth_msm/best_msm/msm_object_clus150_lag300.pkl","rb"))
        
        for rep in tqdm(range(num_reps)):
	        traj = my_mod.simulate(n_steps=mc_steps,start=start_state,seed=seed)
	        path = f'{self.root_path}/round{self.round_no}/trajs/rep{rep}.out'
	        os.makedirs(f'{self.root_path}/round{self.round_no}/trajs/', exist_ok=True)
	        pickle.dump(traj,open(path,'wb'))
	        traj_list.append(path)

        return traj_list

