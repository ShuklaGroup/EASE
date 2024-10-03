"""
Definition of Policy classes for Adaptive Sampling
"""
import pickle
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
#from Utils import *
import deeptime as dt
from tqdm import tqdm
import os
from scipy import interpolate
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import scipy
from deeptime.markov.tools.analysis import eigenvalues
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MaximumLikelihoodMSM, BayesianMSM
from deeptime.decomposition.deep import VAMPNet
from deeptime.util.torch import MLP
from deeptime.util.data import TrajectoryDataset
from Utils import *
# import openmm as mm
# import openmm.app as app
# from simtk.unit import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import mdtraj as md
import itertools
from scipy.stats import entropy
import random

class LeastCounts:
    """
    Least Counts sampling
    """

    def __init__(self, root_path, round_no):
        self.root_path = root_path
        self.round_no = round_no
        self.policy_name = "LeastCounts"

    def get_states(self, traj_list, states):
        """gets state according to policy.

        :param traj_list: Python list.
            List containing paths to trajectory pickles.
        :param states: int.
            Number of states to choose
        :return:
            
        """

        selected_states = []
        trajs = list_to_trajs(traj_list)
        
        all_trajs = np.concatenate(trajs)
      

 
        unique_clusters, cluster_sizes = np.unique(all_trajs, return_counts=True)

        least_counts_states = unique_clusters[np.argsort(cluster_sizes)[:states]]
  

        #least_counts_states = all_trajs[least_counts_states]
        print("Chosen states:",least_counts_states)
        
        return least_counts_states


    def generate_data(self, mc_steps,start_states,seed, num_reps=1):
        """Generates initial (round=0) data.


        :param mc_steps: int.
            Number of steps.
        :param start_states = int
            state id to start from
        :param num_reps = int, default = 1
            Number of replicates to run 
        :param seed = int, default = 42
            Seed to fix random parameters

        :return: list.
            List of pkls files of the trajectories.
        """

        # if seed == None:
        #     seed = np.random.randint(low=1, high=1000)

        traj_list = []
        my_mod = pickle.load(open(f"ground_truth_msm/best_msm/msm_object_clus150_lag300.pkl","rb"))
        #start_states = np.array(list(start_states)*num_reps) # to convert [state] to [state,state,state,...]


        for start_state, rep in tqdm(zip(start_states, range(num_reps))):

            traj = my_mod.simulate(n_steps=mc_steps,start=start_state,seed=seed + rep)
            #print(traj)
            path = f'{self.root_path}/round{self.round_no}/{self.policy_name}/trajs/rep{rep}.out'
            os.makedirs(f'{self.root_path}/round{self.round_no}/{self.policy_name}/trajs/', exist_ok=True)
            pickle.dump(traj,open(path,'wb'))

            traj_list.append(path)

        return traj_list


class RandomSampling(LeastCounts):
    """
    Random sampling
    """

    def __init__(self, root_path, round_no):
        super().__init__(root_path, round_no)
        self.policy_name = "RandomSampling"

    def get_states(self, traj_list=None, states=1):
        """gets state according to policy.

        :param traj_list: Python list.
            List containing paths to trajectory pickles.
        :param states: int.
            Number of states to choose
        :return:
            
        """

        selected_states = []
        trajs = list_to_trajs(traj_list)
        
        all_trajs = np.concatenate(trajs)


        unique_clusters, _ = np.unique(all_trajs, return_counts=True)
        all_states = np.arange(len(unique_clusters))

        # Shuffle the list of states randomly
        np.random.shuffle(all_states)

        # Select the desired number of states
        random_selected_states = all_states[:states]

        random_selected_states = all_trajs[random_selected_states]

        return random_selected_states




class LambdaSampling(LeastCounts):
    """
    Lambda sampling
    """

    def __init__(self, root_path, round_no):
        super().__init__(root_path, round_no)
        self.policy_name = "LambdaSampling"


    def get_states(self, traj_list=None, states=1):
        """Converts selected idx to actual states.

        :param traj_list: Python list.
            List containing paths to trajectory pickles.
        :param states: int.
            Number of states to choose
        :return:
            
        """
        cm, msm = self._make_msm(dtrajs=traj_list)
        best_idx = self._calculate_best_state(cm=cm, msm=msm,states=states)
        traj = np.concatenate(list_to_trajs(traj_list))



        selected_states = traj[best_idx]


        return selected_states 

    def _make_msm(self, dtrajs, best_lag = 1): #, lag = 10):

        dtrajs = list_to_trajs(dtrajs)
        count_model = TransitionCountEstimator(lagtime = best_lag,count_mode = 'sliding').fit_fetch(dtrajs)
        msm = MaximumLikelihoodMSM().fit_fetch(count_model.submodel_largest())

        return count_model , msm

    def _calculate_best_state(self,cm,msm,states):

        t_ij = msm.transition_matrix
        num_states = t_ij.shape[0]
        evs = np.real(eigenvalues(t_ij))
        e = evs[1] 
        A = t_ij - e*np.identity(num_states)
        P,L,U = scipy.linalg.lu(A)
        m=states
        zero = U[0:-1,-1]
        U = U[0:-1,0:-1]
        e_k = np.zeros((num_states))
        e_k[-1] = 1
        
        x_a = np.linalg.solve(U,zero)
        x_a = np.append(x_a,[1])

        x = np.linalg.solve(L.T,e_k)
        sens_mat = np.outer(x,x_a)/np.dot(x, x_a)
        
        diff_arr = []
        var_arr = []
        
        for s in range(num_states):
        
            p_i = t_ij[s,:]
            s_i = sens_mat[:,s]
            w_i = cm.count_matrix[s,:].sum()
        
            qc = np.diag(p_i) - np.outer(p_i,p_i)
            qr = np.matmul(qc,s_i)
            q_i = np.dot(s_i, qr)
        
            diff = ((q_i/(w_i+1))-(q_i/(w_i+m+1)))
            diff_arr.append(diff)
            

            var = ((q_i/(w_i+1)))
            var_arr.append(var)


        best_idx = np.argsort(diff_arr)[-states:]

        return best_idx

