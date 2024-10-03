from Simulation import ToySimulation
from Policies import LeastCounts, RandomSampling , LambdaSampling, MaxEntSampling
from Analysis import Evaluate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns
from Utils import*
from tqdm import tqdm
import mdshare
import pickle
from deeptime.plots import plot_implied_timescales, plot_energy2d
from deeptime.util import energy2d
import os




# Round 0

ROOT_PATH = "sim_output_betas" 
REPLICATES = 5 
STEPS = 2000
ROUND_NO = 0
TOP = "alanine_dipeptide_vacuum_new.pdb"
SAVE_RATE = 10

print(f"=============================ROUND 0===========================================")

sim_0 = ToySimulation(root_path = ROOT_PATH, round_no=ROUND_NO,top_file=TOP,n_steps=500)
trajs_0 = sim_0.get_initial_data(num_reps=REPLICATES,save_rate=10)
feat_0 = sim_0.cluster(trajs_0,num_reps=REPLICATES) 



# Sampling


#betas = [0.0,0.2,0.4,0.6,0.8,1]
betas = [0.8,1]
ROUND_REPS = 20
ROUNDS = range(1, 51,1)  # Adjust the range accordingly

for BETA in betas:
    exp_list = []
    conv_list = []
    tot_list = []
    name_list = []

    for _ in tqdm(range(ROUND_REPS)):

        print(f"=============================ROUND 0===========================================")

        sim_0 = ToySimulation(root_path = ROOT_PATH, round_no=ROUND_NO,top_file=TOP,n_steps=STEPS)
        trajs_0 = sim_0.get_initial_data(num_reps=REPLICATES,save_rate=10)
        feat_0 = sim_0.cluster(trajs_0,num_reps=REPLICATES) 

        r_t_list = trajs_0
        r_ft_list = feat_0
      
        best_exp = []
        best_conv = []
        best_tot = []
        best_name = []
        prev_best_data = None





        for r_no in tqdm(ROUNDS):
            print(f"=============================ROUND {r_no}===========================================")
            lc_obj = LeastCounts(n_steps=STEPS,root_path=ROOT_PATH,save_rate=SAVE_RATE, round_no=r_no,num_reps=REPLICATES)
            st_lc_obj = lc_obj.get_states(feat_path=r_ft_list, traj_list = r_t_list)
            trajs_obj_lc = lc_obj.generate_data(states_list=st_lc_obj) 
            feat_path_obj_lc = lc_obj.compute_dihedral_features_for_trajectories(trajs_obj_lc)

            rs_obj = RandomSampling(n_steps=STEPS,root_path=ROOT_PATH,save_rate=SAVE_RATE, round_no=r_no,num_reps=REPLICATES)
            st_rs_obj = rs_obj.get_states(feat_path=r_ft_list, traj_list = r_t_list)
            trajs_obj_rs = rs_obj.generate_data(states_list=st_rs_obj) 
            feat_path_obj_rs = rs_obj.compute_dihedral_features_for_trajectories(trajs_obj_rs)

            ld_obj = LambdaSampling(n_steps=STEPS,root_path=ROOT_PATH,save_rate=SAVE_RATE, round_no=r_no,num_reps=REPLICATES)
            st_ld_obj = ld_obj.get_states(feat_path=r_ft_list, traj_list = r_t_list)
            trajs_obj_ld = ld_obj.generate_data(states_list=st_ld_obj) 
            feat_path_obj_ld = ld_obj.compute_dihedral_features_for_trajectories(trajs_obj_ld)

            # mv_obj = MaxEntSampling(n_steps=STEPS,root_path=ROOT_PATH,save_rate=SAVE_RATE, round_no=r_no,num_reps=REPLICATES)
            # st_mv_obj = mv_obj.get_states(feat_path=r_ft_list, traj_list = r_t_list)
            # trajs_obj_mv = mv_obj.generate_data(states_list=st_mv_obj) 
            # feat_path_obj_mv = mv_obj.compute_dihedral_features_for_trajectories(trajs_obj_mv)


            ev_obj = Evaluate(root_path=ROOT_PATH, round_no=r_no,num_reps=REPLICATES,n_steps=STEPS,prev_best_data=r_ft_list)
            r_t_list, r_ft_list, df = ev_obj.rank_policies(feat_paths=[feat_path_obj_lc, feat_path_obj_rs, feat_path_obj_ld],beta=BETA) #, feat_path_obj_mv])

            prev_best_data = r_ft_list

            best_exp.append(df.loc[ev_obj.best_policy_name][0])
            best_conv.append(df.loc[ev_obj.best_policy_name][1])
            best_tot.append(df.loc[ev_obj.best_policy_name][2])
            best_name.append(ev_obj.best_policy_name)



        exp_list.append(best_exp)
        conv_list.append(best_conv)
        tot_list.append(best_tot)
        name_list.append(best_name)

    os.makedirs(f'{ROOT_PATH}/analysis/', exist_ok=True)
    pickle.dump(exp_list,open(f'{ROOT_PATH}/analysis/exp_{BETA}.pkl','wb'))    
    pickle.dump(conv_list,open(f'{ROOT_PATH}/analysis/conv_{BETA}.pkl','wb'))  
    pickle.dump(tot_list,open(f'{ROOT_PATH}/analysis/tot_{BETA}.pkl','wb'))  
    pickle.dump(name_list,open(f'{ROOT_PATH}/analysis/name_{BETA}.pkl','wb'))  

