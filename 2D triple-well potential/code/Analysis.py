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


class Evaluate:
    """
    evaluation class for each round of sampling
    """
    def __init__(self, root_path, round_no, num_reps, prev_best_data=None):

        self.root_path = root_path
        self.round_no = round_no
        self.num_reps = num_reps
        self.best_policy_name = None         # this will get replaced
        self.prev_best_data = prev_best_data


    def rank_policies(self,trajs,beta=0.5):

        df_met = self._metrics(trajs,beta)
        self.best_policy_name = df_met['result'].idxmin()
        current_best_data = self._get_data_from_policy_name(self.best_policy_name) 
        self.best_round_data = self._update_data(current_best_data)
        self._prepare_next_round(traj_list=self.best_round_data)

        ### Implement end of round plotting etc. here
        #print(self.best_policy_name)
        #print(df_met)
        #print(self.best_round_data)

        return self.best_round_data, df_met

    def _prepare_next_round(self,traj_list,c=100):
        """Cluster the trajectories in c clusters.

        :param traj_list: Python list.
            List containing paths to trajectory pickles.
        :param c: int. default = 100.
            Number of clusters 

        :return  clus_path, d_path: Python tuple.
            Path to cluster object, path to dtraj
            
            """    


        path = f'{self.root_path}/round{self.round_no}/clus_pkls/'
        os.makedirs(path, exist_ok=True)


        trajs =list_to_trajs(traj_list)

        cluster_obj = KMeans(n_clusters=c,  # place c cluster centers
                                init_strategy='kmeans++',  # kmeans++ initialization strategy
                                n_jobs=8,fixed_seed=True).fit_fetch(np.concatenate(trajs))

        dtrajs = [cluster_obj.transform(x) for x in trajs]
        d_path = path+f"round{self.round_no}_dtraj.pkl"
        clus_path = path+f"round{self.round_no}_clusObj.pkl"

        pickle.dump(dtrajs, open(d_path,'wb'))
        pickle.dump(cluster_obj,open(clus_path,'wb'))

        return clus_path, d_path        




    def _metrics(self,trajs, beta):

        ref_msm = pickle.load(open('ground_truth_msm/best_msm/msm_object_clus400_lag10.pkl','rb'))

        met_dict = {}
        best_res = 0

        for i, traj in enumerate(trajs):
            name =  self._get_policy_name(traj) 
            joined_traj = self._update_data(traj)
            cm, msm = self._make_msm(traj_list=joined_traj)
            expl = self._measure_exploration(count_model=cm)
            conv1 = self._rel_entropy(ref_msm=ref_msm, test_msm=msm)
            conv2 = self._frob_norm(ref_msm=ref_msm, test_msm=msm)
            res = beta * (1 - expl) + (1 - beta)*(conv1) 
            #met_dict[name] = {'exploration':np.round(expl,4), 'convergence(ent) ':np.round(conv1,4), 'convergence(frob) ':np.round(conv2,4) }
            met_dict[name] = {'1 - exploration':1 - expl, 'convergence':np.round(conv1,4), 'result':res}
            self._save_msm_matrix(policy_name=name,test_msm=msm)

        df_met = pd.DataFrame(met_dict).transpose()
        print(df_met)
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

        name = trajs[0].split('/')[2]
        return name

    def _update_data(self, trajs):

        updated_data = []
        if self.round_no == 1:
            updated_data = trajs + self._get_zero_data()
        else:
            updated_data = trajs + self.prev_best_data
        
        return updated_data 

    def _make_msm(self, traj_list,best_lag = 10): #, lag = 10):

        #lags = np.arange(1, 40, 1)
        #models = []

        trajs = list_to_trajs(traj_list)
        clus_path = f'ground_truth_msm/best_msm/clus_400_obj.pkl'
        clus = pickle.load(open(clus_path,'rb'))
        dtrajs = clus.transform(np.concatenate(trajs))

        # for lag in tqdm(lags):
        #     count_model = TransitionCountEstimator(lag, 'effective').fit_fetch(dtrajs)
        #     msm = BayesianMSM().fit_fetch(count_model.submodel_largest())
        #     models.append(msm)
        # its_data = implied_timescales(models)
        # fig, ax = plt.subplots(1, 1)
        # plot_implied_timescales(its_data, n_its=4, ax=ax)
        # ax.set_yscale('log')
        # ax.set_title(f'Implied timescales')
        # ax.set_xlabel('lag time (steps)')
        # ax.set_ylabel('timescale (steps)')
        #plt.show()        
        #fig.savefig(f'ground_truth_msm/its_plots/ITS_plot_clus{k}.jpg',dpi=300)

        enforce_states = int(clus_path.split('/')[2].split('_')[1]) 
        count_model = TransitionCountEstimator(lagtime = best_lag,count_mode = 'sliding', n_states = enforce_states).fit_fetch(dtrajs)
        msm = MaximumLikelihoodMSM(allow_disconnected = True).fit_from_counts(count_model.count_matrix + (1/enforce_states)).fetch_model()

        return count_model , msm


    def _measure_exploration(self, count_model):

        visited = count_model.visited_set 
        total = count_model.n_states

        return np.round(len(visited) / total, 3)


    def _rel_entropy(self, ref_msm, test_msm):

        p = ref_msm.transition_matrix
        q = test_msm.transition_matrix

        if p.shape != q.shape:
            raise ValueError("The reference and test msm have different shapes bro.")
        if np.any(q) == 0:
            raise ValueError("The test q has zeros, can't divide")
        
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

        p = ref_msm.transition_matrix
        q = test_msm.transition_matrix

        diff_mat = p - q 

        return LA.norm(diff_mat)


    def _save_msm_matrix(self,policy_name,test_msm):

        q = test_msm.transition_matrix
        path = f'{self.root_path}/matrix_photos/{policy_name}/'
        os.makedirs(path, exist_ok=True)

        pickle.dump(q,open(f'{self.root_path}/matrix_photos/{policy_name}/{self.round_no}_{policy_name}_msm_mat.pkl','wb'))


class Single_Evaluate(Evaluate):
    """
    evaluation class for single policy at each round of sampling
    """
    def __init__(self, root_path, round_no, num_reps,policy_name, prev_best_data=None):
        super().__init__(root_path, round_no, num_reps, prev_best_data=None)

        self.policy_name = policy_name


    def rank_policies(self,trajs):

        df_met = self._metrics(trajs)
        self.best_policy_name = self.policy_name
        current_best_data = self._get_data_from_policy_name(self.best_policy_name) 
        self.best_round_data = self._update_data(current_best_data)
        self._prepare_next_round(traj_list=self.best_round_data)

        ### Implement end of round plotting etc. here
        #print(self.best_policy_name)
        #print(df_met)
        #print(self.best_round_data)

        return self.best_round_data, df_met

