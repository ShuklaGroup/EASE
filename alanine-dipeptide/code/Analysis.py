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
import itertools
import re


class Evaluate:
    """
    Evaluation class for each round of sampling
    """

    best_policy_list = []
    best_data_list = []


    def __init__(self, root_path, round_no,n_steps, num_reps, prev_best_data=None):

        self.root_path = root_path
        self.round_no = round_no
        self.num_reps = num_reps
        self.best_policy_name = None         # this will get replaced
        self.prev_best_data = prev_best_data
        self.n_steps = n_steps


    def rank_policies(self,feat_paths,beta=0.5):
        """Rank Policies.

        :param feat_list: Python list.
            List containing paths to features (dihedrals) pickles.
        :param beta: float. default = 0.5.
            Beta parameter to control balance between exploration and convergence 

        :return  best_data_list, best_round_data, df_met: Python tuple.
            List containing data appended from all previous rounds, data sampled from the best (chosen) policy, pandas dataframe containing the results of the ranked policies
            
        """    

        df_met = self._metrics(feat_paths,beta)
        self.best_policy_name = df_met['result'].idxmin()
        self.__class__.best_policy_list.append(self.best_policy_name)
        current_best_data = self._get_data_from_policy_name(self.best_policy_name) 
        self.best_round_data = self._update_data(current_best_data)
        self._prepare_next_round(feat_list=self.best_round_data)
        self._return_traj_list(self.__class__.best_policy_list)

        return self.__class__.best_data_list, self.best_round_data, df_met

    def _return_traj_list(self,best_policy_list):
        """
        Helper function to get round 0 saved data and append to list of best data, including the current best data
        """
        zero_list = []
        for rep in range(self.num_reps):
            path =  f"{self.root_path}/round0/trajs/rep{rep}.dcd"
            zero_list.append(path)


        def sort_file_paths(file_paths):
            return sorted(file_paths, key=lambda path: (
                int(re.search(r'round(\d+)', path).group(1)), 
                int(re.search(r'rep(\d+)', path).group(1))
            ))

        def include(name,rs):
            l = []
            for r in range(rs):
                p = path =  f"{self.root_path}/round{self.round_no}/{name}/trajs/rep{r}.dcd"
                l.append(p)
            return l
        b_list = include(name=best_policy_list[self.round_no -1],rs=self.num_reps)
        result = b_list + zero_list
        result = sort_file_paths(result)
        #result = [item for sublist in result for item in sublist]
        self.__class__.best_data_list.extend(result)
         

    def _metrics(self,feat_paths, beta):
        """
        Helper function to calculate metrics (exploration and covergence), returns dataframe containing policies and their respective metrics
        """
        ref_msm = pickle.load(open('ground_truth_msm/best_msm/msm_object_clus300_lag5.pkl','rb'))

        met_dict = {}
        best_res = 0

        for i, feat_path in enumerate(feat_paths):
            name =  self._get_policy_name(feat_path) 
            joined_feat_path = self._update_data(feat_path)
            cm,msm = self._make_msm(feat_path_list=joined_feat_path)

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


    def _get_policy_name(self, feat_path):
        """
        Helper function to infer policy name from feature path.
        """
        name = feat_path[0].split('/')[2]
        return name    
 
    def _update_data(self, feat_paths):
        """
        Append best policy feature data to previous best data list
        """

        updated_data = []
        if self.round_no == 1:
            updated_data = feat_paths + self._get_zero_data()
        else:
            updated_data = feat_paths + self.prev_best_data
        
        return updated_data 

    def _get_zero_data(self):
        """
        Helper function to get round 0 data, to be appended to later rounds.
        """

        zero_path=[]
        z = f"{self.root_path}/round0/trajs/round0.ft"
        zero_path.append(z)
        return zero_path 


    def _make_msm(self, feat_path_list,best_lag = 15):
        """Creates a Markov State Model

        :param feat_path_list: Python list.
            List of featurized trajectory paths.
        :param best_lag: int.
            Lagtime for the MSM
        :return: count model object, MSM object.
            Count Model from the transition count estimator (as implemented in Deeptime library), MSM object, the fitted MSM from the dtrajs
        """

        lags = np.arange(1, 40, 1)
        models = []

        feats = [pickle.load(open(x,'rb')) for x in feat_path_list]
        feats = list(itertools.chain(*feats))

        clus_path = f'ground_truth_msm/best_msm/clus_300_obj.pkl'
        cluster_obj = pickle.load(open(clus_path,'rb'))

        dtrajs = [cluster_obj.transform(x) for x in feats]

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
        # plt.show()        
        # fig.savefig(f'ground_truth_msm/its_plots/ITS_plot_clus{k}.jpg',dpi=300)

        enforce_states = int(clus_path.split('/')[2].split('_')[1]) 
        count_model = TransitionCountEstimator(lagtime = best_lag,count_mode = 'sliding', n_states = enforce_states).fit_fetch(dtrajs)
        msm = MaximumLikelihoodMSM(allow_disconnected = True).fit_from_counts(count_model.count_matrix + (1/enforce_states)).fetch_model()

        return count_model , msm

    def _measure_exploration(self, count_model):
        """
        Helper function to calculate exploration
        """

        visited = count_model.visited_set 
        total = count_model.n_states
        print(f"Visited states {len(visited)} and Total states {total}")
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


    def _get_data_from_policy_name(self,policy_name):

        data_path = []
        z = f"{self.root_path}/round{self.round_no}/{policy_name}/trajs/round{self.round_no}.ft"
        data_path.append(z)

        return data_path

    def _prepare_next_round(self,feat_list):
        """Cluster the trajectories in c clusters.

        :param traj_list: Python list.
            List containing paths to trajectory pickles.
        :param c: int. default = 100.
            Number of clusters 

        :return  clus_path, d_path: Python tuple.
            Path to cluster object, path to dtraj
            
            """    

        vol = (self.round_no + 1)*self.num_reps * self.n_steps 
        c = int(np.sqrt(vol)/2)

        if c < 50: c=50
        elif c>300: c=300
        

        print(f"States for round {self.round_no} are {c}")
        feats = [pickle.load(open(x,'rb')) for x in feat_list]
        feats = list(itertools.chain(*feats))

        path = f'{self.root_path}/round{self.round_no}/clus_pkls/'
        os.makedirs(path, exist_ok=True)
                                                                                                                                                      

        
        cluster_obj = KMeans(n_clusters=c,  # place c cluster centers
                             init_strategy='kmeans++',  # kmeans++ initialization strategy
                             n_jobs=8,fixed_seed=True).fit_fetch(feats)

        dtrajs = [cluster_obj.transform(x) for x in feats]
        d_path = path+f"round{self.round_no}_dtraj.pkl"
        clus_path = path+f"round{self.round_no}_clusObj.pkl"

        pickle.dump(dtrajs, open(d_path,'wb'))
        pickle.dump(cluster_obj,open(clus_path,'wb'))
        #print(f"=============================Dumped at {clus_path} ===========================================")

        return clus_path, d_path    

