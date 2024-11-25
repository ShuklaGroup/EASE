from kmc_Simulation import kMC_simulation
from kmc_Policies import LeastCounts, RandomSampling, LambdaSampling 
from kmc_Analysis import Evaluate
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import os
import matplotlib.pyplot as plt
import seaborn as sns
from Utils import*
from tqdm import tqdm
from collections import Counter
import sys


def kmc_run(root_path,msm_path,steps,old,beta):
        """ Gets the best policy according to EASE

        :param root_path: str.
            Path to save the data
        :param msm_path: str.
            Path to saved MSM (because these are kMC trajectories, which require a kinetic model for generation)
        :param steps: int.
            Number of steps to run.
        :param old: Python list.
            List containing paths to feature/trajectory pickles upto round i-2.  
        :param beta: float.
            Beta for choosing convergence vs exploration in kMC space
        :return most_common_name,policies[most_common_name]: Python tuple
            Best policy name (str), best policy (adaptive sampling class object)
        """
    policies = {
    'LeastCounts': 0,
    'RandomSampling': 1,
    'LambdaSampling': 2
    }
    ROOT_PATH = root_path
    MSM_PATH = msm_path
    REPLICATES = 5
    old = list_to_trajs(old)
    trajs_0 = old

    r_no = 1
    betas = [beta]
    STEPS=steps


    names = []
    for _ in range(20):
        for BETA in betas:

            SEED = np.random.randint(low=1, high=1e6)

            lc_obj = LeastCounts(root_path=ROOT_PATH, round_no=r_no,msm_path=MSM_PATH)
            st_lc_obj = lc_obj.get_states(traj_list=trajs_0,states=REPLICATES)
            trajs_obj_lc = lc_obj.generate_data(mc_steps=STEPS,start_states=st_lc_obj,num_reps=REPLICATES,seed=SEED) 

            rs_obj = RandomSampling(root_path=ROOT_PATH, round_no=r_no,msm_path=MSM_PATH)
            st_rs_obj = rs_obj.get_states(traj_list=trajs_0,states=REPLICATES)
            trajs_obj_rs = rs_obj.generate_data(mc_steps=STEPS,start_states=st_rs_obj,num_reps=REPLICATES,seed=SEED) 

            ld_obj = LambdaSampling(root_path=ROOT_PATH, round_no=r_no,msm_path=MSM_PATH)
            st_ld_obj = ld_obj.get_states(traj_list=trajs_0,states=REPLICATES)
            trajs_obj_ld = ld_obj.generate_data(mc_steps=STEPS,start_states=st_ld_obj,num_reps=REPLICATES,seed=SEED) 


            ev_obj =  Evaluate(root_path=ROOT_PATH, round_no=r_no,num_reps=REPLICATES,msm_path=MSM_PATH,prev_best_data=None)
            _,_, best_name = ev_obj.rank_policies(trajs=[trajs_obj_lc,trajs_obj_rs,trajs_obj_ld],beta=BETA) #,trajs_obj_ld
            names.append(best_name)
    # Count occurrences of each string
    string_counts = Counter(names)

    # Get the most common string
    most_common_name = string_counts.most_common(1)[0][0]
    print(most_common_name)
    return most_common_name,policies[most_common_name]


