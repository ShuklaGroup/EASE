from Simulation import OSSimulation
from Policies import LeastCounts
from Analysis import Evaluate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Utils import*
from tqdm import tqdm
import os


# Round 0

ROOT_PATH = "sim_output_LeastCounts" 
REPLICATES = 5
STEPS = 5
policy_name = "LeastCounts"



ROUND_REPS = 30
ROUNDS = range(1, 51,1)  # Adjust the range accordingly

exp_list = []
conv_list = []
tot_list = []
name_list = []
STEPS = 30
betas = [0.5]#,0.2,0.4,0.5,0.6,0.8,1.0]
for BETA in betas:
    for _ in tqdm(range(ROUND_REPS)):
        print(f"=============================ROUND 0===========================================")

        sim_0 = OSSimulation(root_path = ROOT_PATH, round_no=0)
        trajs_0 = sim_0.get_initial_data(mc_steps=STEPS,num_reps=REPLICATES, seed = 42)

        r_t_list = trajs_0
        best_exp = []
        best_conv = []
        best_tot = []
        best_name = []
        prev_best_data = None
      
        for r_no in tqdm(ROUNDS):
            SEED = np.random.randint(low=1, high=1000) 

            print(f"=============================ROUND {r_no}===========================================")

            lc_obj = LeastCounts(root_path=ROOT_PATH, round_no=r_no)
            st_lc_obj = lc_obj.get_states(traj_list=r_t_list,states=REPLICATES)
            trajs_obj_lc = lc_obj.generate_data(mc_steps=STEPS,start_states=st_lc_obj,num_reps=REPLICATES,seed=SEED) 
            

            ev_obj =  Evaluate(root_path=ROOT_PATH, round_no=r_no,num_reps=REPLICATES,prev_best_data=prev_best_data)
            r_t_list, df = ev_obj.rank_policies(trajs=[trajs_obj_lc],beta=BETA)

            prev_best_data = r_t_list

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

