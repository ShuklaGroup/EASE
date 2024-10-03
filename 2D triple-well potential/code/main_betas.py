from Simulation import ToySimulation
from Policies import LeastCounts, RandomSampling, LambdaSampling
from Analysis import Evaluate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Utils import*
from tqdm import tqdm




# Round 0

ROOT_PATH = "sim_output_betas" 
REPLICATES = 5 
STEPS = 300
ROUND_NO = 0
COORDS = generate_coordinates(REPLICATES,seed=12)

sim_0 = ToySimulation(root_path = ROOT_PATH, round_no=ROUND_NO)
trajs_0 = sim_0.get_initial_data(md_steps=STEPS,initial_coords=COORDS, num_reps=REPLICATES, seed = 4000)
clus_0, dtraj_0 = sim_0.cluster(trajs_0,num_reps=REPLICATES) 

# Round 1

## i. Least Counts

lc_1 = LeastCounts(root_path=ROOT_PATH, round_no=1)
st_lc_1 = lc_1.get_states(traj_list=trajs_0)
trajs_1_lc = lc_1.generate_data(md_steps=STEPS,initial_coords=st_lc_1,num_reps=5) 

## ii. Random Sampling

rs_1 = RandomSampling(root_path=ROOT_PATH, round_no=1)
st_rs_1 = rs_1.get_states(traj_list=trajs_0)
trajs_1_rs = rs_1.generate_data(md_steps=STEPS, initial_coords=st_rs_1, num_reps=5)

## iii. Lambda Sampling
ld_1 = LambdaSampling(root_path=ROOT_PATH, round_no=1)
st_ld_1 = ld_1.get_states(traj_list=trajs_0)
trajs_1_ld = ld_1.generate_data(md_steps=STEPS, initial_coords=st_ld_1, num_reps=5)

## iv. Evaluation 

ev = Evaluate(root_path=ROOT_PATH, round_no=1,num_reps=REPLICATES) 
r1_list, df = ev.rank_policies(trajs=[trajs_1_lc,trajs_1_rs,trajs_1_ld])
best_exp1 = (df.loc[ev.best_policy_name][0])
best_conv1 = (df.loc[ev.best_policy_name][1])
best_tot1 = (df.loc[ev.best_policy_name][2])
best_name1 = (ev.best_policy_name)

# # Sampling


betas = [0.5]
ROUND_REPS = 50
ROUNDS = range(2, 100,1)  # Adjust the range accordingly

for BETA in betas:
    exp_list = []
    conv_list = []
    tot_list = []
    name_list = []

    for _ in tqdm(range(ROUND_REPS)):
        r_list = r1_list
        best_exp = []
        best_conv = []
        best_tot = []
        best_name = []

        for r_no in tqdm(ROUNDS):
            lc_obj = LeastCounts(root_path=ROOT_PATH, round_no=r_no)
            st_lc = lc_obj.get_states(traj_list=r_list)
            traj_lc = lc_obj.generate_data(md_steps=STEPS, initial_coords=st_lc, num_reps=REPLICATES)

            rs_obj = RandomSampling(root_path=ROOT_PATH, round_no=r_no)
            st_rs = rs_obj.get_states(traj_list=r_list)
            traj_ls = rs_obj.generate_data(md_steps=STEPS, initial_coords=st_rs, num_reps=REPLICATES)

            ld_obj = LambdaSampling(root_path=ROOT_PATH, round_no=r_no)
            st_ld = ld_obj.get_states(traj_list=r_list)
            traj_ld = ld_obj.generate_data(md_steps=STEPS, initial_coords=st_ld, num_reps=REPLICATES)

            ev_obj = Evaluate(root_path=ROOT_PATH, round_no=r_no, num_reps=REPLICATES, prev_best_data=r_list)
            r_list, df = ev_obj.rank_policies(trajs=[traj_lc, traj_ls, traj_ld],beta=BETA)

            best_exp.append(df.loc[ev_obj.best_policy_name][0])
            best_conv.append(df.loc[ev_obj.best_policy_name][1])
            best_tot.append(df.loc[ev_obj.best_policy_name][2])
            best_name.append(ev_obj.best_policy_name)



        exp_list.append(best_exp)
        conv_list.append(best_conv)
        tot_list.append(best_tot)
        name_list.append(best_name)

    pickle.dump(exp_list,open(f'{ROOT_PATH}/analysis/exp_{BETA}.pkl','wb'))    
    pickle.dump(conv_list,open(f'{ROOT_PATH}/analysis/conv_{BETA}.pkl','wb'))  
    pickle.dump(tot_list,open(f'{ROOT_PATH}/analysis/tot_{BETA}.pkl','wb'))  
    pickle.dump(name_list,open(f'{ROOT_PATH}/analysis/name_{BETA}.pkl','wb'))  
