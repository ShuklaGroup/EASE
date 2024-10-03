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
import scipy
from scipy.stats import entropy
from deeptime.plots import plot_implied_timescales
from deeptime.util.validation import implied_timescales
from deeptime.markov.tools.analysis import eigenvalues
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MaximumLikelihoodMSM, BayesianMSM
from deeptime.decomposition.deep import VAMPNet
from deeptime.util.torch import MLP
from deeptime.util.data import TrajectoryDataset
from Utils import *
import openmm as mm
import openmm.app as app
from simtk.unit import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import mdtraj as md
import itertools
import random

from kmc_main_betas import kmc_run


class Gradient:
    """
    class to rank policies on the fly using gradient approach
    """

    def __init__(self, root_path, round_no,kmc_beta):
        self.root_path = root_path
        self.round_no = round_no

        self.kmc_beta = kmc_beta
        print("Imma run kmc now")


    def get_policy(self,old,new):
        
        msm_path = self._make_msm(new=new)
        best_name = kmc_run(root_path=str(self.root_path + "_kmc"),msm_path=msm_path,steps=120,old=old, beta=self.kmc_beta)

        return best_name


    def _make_msm(self,new,lag=1):
        '''
        Makes msm for current round data

        :param feat_path: Python list.
            List containing paths to feature pickles upto round i-1.  


        :return  dtrajs: Python list.
            clustered trajectories
        '''

        dtrajs = list_to_trajs(new)

        all_dtrajs = np.concatenate(dtrajs)
        unique_clusters, _ = np.unique(all_dtrajs, return_counts=True)


        enforce_states = 150 # this can be any reasonable number, wouldnot matter in a real system
        count_model = TransitionCountEstimator(lagtime = lag,count_mode = 'sliding', n_states = enforce_states).fit_fetch(dtrajs)
        msm = MaximumLikelihoodMSM(allow_disconnected = True).fit_from_counts(count_model.count_matrix + (1/enforce_states)).fetch_model()
        


        msm_fold_path = f"{self.root_path}/round{self.round_no}/grad_msm/"
        msm_path = msm_fold_path + f"msm_states_{msm.transition_matrix.shape[0]}.pkl"
        #print(msm_path)
        os.makedirs(msm_fold_path, exist_ok=True)

        pickle.dump(msm,open(msm_path,"wb"))

        return msm_path