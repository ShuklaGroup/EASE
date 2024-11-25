"""
Definition of evaluation classes
"""
import pickle
import numpy as np
from numpy import linalg as LA
from Utils import *
import deeptime as dt
from tqdm import tqdm
import os
import glob
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MaximumLikelihoodMSM, BayesianMSM
from deeptime.plots import plot_implied_timescales
from deeptime.util.validation import implied_timescales
from deeptime.clustering import KMeans          
import pandas as pd 
from kmc_Utils import *



class Evaluate:
    """
    evaluation class for each round of sampling
    """
    rounds_data_list = []
    def __init__(self, root_path, round_no, num_reps,msm_path, prev_best_data=None):

        self.root_path = root_path
        self.round_no = round_no
        self.num_reps = num_reps
        self.best_policy_name = None         # this will get replaced
        self.prev_best_data = prev_best_data
        self.msm_path = msm_path



    def rank_policies(self,trajs,beta):
        """Rank Policies.

        :param trajs: Python list.
            List containing paths to kMC trajectories
        :param beta: float.
            Beta parameter to control balance between exploration and convergence 

        :return  best_round_data, df_met, best_policy_name: Python tuple.
            Data sampled from the best (chosen) policy, pandas dataframe containing the results of the ranked policies, name of best policy selected from EASE
            
        """    

        df_met = self._metrics(trajs,beta)
        self.best_policy_name = df_met['result'].idxmin()
        current_best_data = self._get_data_from_policy_name(self.best_policy_name) 
        self._save_current_best_data(current_best_data)
        self.best_round_data = self._update_data(current_best_data)
        return self.best_round_data, df_met,self.best_policy_name

    def _save_current_best_data(self,data):
        traj = np.concatenate(list_to_trajs(data))
        unique_clusters, _ = np.unique(traj, return_counts=True)
        Evaluate.rounds_data_list.append(unique_clusters)


    def _metrics(self,trajs, beta):
        """
        Helper function to calculate metrics (exploration and covergence), returns dataframe containing policies and their respective metrics
        """
        ref_msm = pickle.load(open(self.msm_path,'rb'))

        met_dict = {}
        best_res = 0

        for i, traj in enumerate(trajs):
            name =  self._get_policy_name(traj) 
            #joined_traj = self._update_data(traj)
            cm, msm = self._make_msm(traj_list=traj)
            expl = self._measure_exploration(count_model=cm)
            conv1 = self._rel_entropy(ref_msm=ref_msm, test_msm=msm)
            res = beta * (1 - expl) + (1 - beta)*(conv1) 
            met_dict[name] = {'1 - exploration':1 - expl, 'convergence':np.round(conv1,4), 'result':res}
            self._save_msm_matrix(policy_name=name,test_msm=msm)

        df_met = pd.DataFrame(met_dict).transpose()
        #print(df_met)
        return df_met

    def _get_zero_data(self):

        zero_path = []
        for rep in range(self.num_reps):
            z = f"{self.root_path}/round0/trajs/rep{rep}.out"
            zero_path.append(z)
        return zero_path 

    def _get_data_from_policy_name(self,policy_name):

        data_path = []
        for rep in range(self.num_reps):
            z = f"{self.root_path}/round{self.round_no}/{policy_name}/trajs/rep{rep}.out"
            data_path.append(z)

        return data_path

    def _get_policy_name(self, trajs):
        """
        Helper function to get round 0 data, to be appended to later rounds.
        """
        name = trajs[0].split('/')[2]
        return name

    def _update_data(self, trajs):

        updated_data = []
        if self.round_no == 1:
            updated_data = trajs #+ self._get_zero_data()
        else:
            updated_data = trajs + self.prev_best_data
        
        return updated_data 

    def _make_msm(self, traj_list,best_lag = 1):
        """Creates a Markov State Model

        :param traj_list: Python list.
            List of trajectory paths.
        :param best_lag: int.
            Lagtime for the MSM
        :return: count model object, MSM object.
            Count Model from the transition count estimator (as implemented in Deeptime library), MSM object, the fitted MSM from the dtrajs
        """
        trajs = list_to_trajs(traj_list)
        #clus_path = f'{self.root_path}/round{self.round_no}/clus_pkls/round{self.round_no}_clusObj.pkl'

        dtrajs = trajs #
        enforce_states = int(self.msm_path.split('/')[3].split('_')[2].split('.')[0]) # finding number of states, same as the saved MSM for current round
        print(enforce_states)
        count_model = TransitionCountEstimator(lagtime = best_lag,count_mode = 'sliding', n_states = enforce_states).fit_fetch(dtrajs)
        msm = MaximumLikelihoodMSM(allow_disconnected = True).fit_from_counts(count_model.count_matrix + (1/enforce_states)).fetch_model()

        print(msm.transition_matrix.shape)

        return count_model , msm


    def _measure_exploration(self, count_model):
        """
        Helper function to calculate exploration
        """
        visited = count_model.visited_set 
        total = count_model.n_states

        return np.round(len(visited) / total, 3)


    def _rel_entropy(self, ref_msm, test_msm):
        """
        Helper function to calculate convergence (relative entropy)
        """
        p = ref_msm.transition_matrix
        q = test_msm.transition_matrix


        if p.shape != q.shape:
            raise ValueError("The reference and test msm have different shapes bro.")
        if np.any(q) == 0:
            raise ValueError("The test q has zeros, can't divide ")
        
        ent = 0.0
        
        sv = ref_msm.stationary_distribution

        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                if p[i, j] == 0:
                    c = 0
                else:
                    c = sv[i] * p[i, j] * np.log(p[i, j] / q[i, j])
                ent = ent + c
        
        return np.sum(ent)


    def _frob_norm(self, ref_msm, test_msm):
        """
        Helper function to calculate convergence (frobenius norm, not used in code)
        """
        p = ref_msm.transition_matrix
        q = test_msm.transition_matrix

        diff_mat = p - q 

        return LA.norm(diff_mat)


    def _save_msm_matrix(self,policy_name,test_msm):
        """
        Helper function to save MSM object as pkl.
        """
        q = test_msm.transition_matrix
        path = f'{self.root_path}/matrix_photos/{policy_name}/'
        os.makedirs(path, exist_ok=True)

        pickle.dump(q,open(f'{self.root_path}/matrix_photos/{policy_name}/{self.round_no}_{policy_name}_msm_mat.pkl','wb'))

