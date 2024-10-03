from Simulation import ToySimulation
from Policies import LeastCounts
from Analysis import Evaluate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Utils import*
from tqdm import tqdm
import time

# Round 0

ROOT_PATH = "sim_output_LeastCounts" 
REPLICATES = 5 
STEPS = 300
ROUND_NO = 0
COORDS = generate_coordinates(REPLICATES,seed=12)
policy_name = 'LeastCounts'
TIME = []

sim_0 = ToySimulation(root_path = ROOT_PATH, round_no=ROUND_NO)
trajs_0 = sim_0.get_initial_data(md_steps=STEPS,initial_coords=COORDS, num_reps=REPLICATES, seed = 4000)
clus_0, dtraj_0 = sim_0.cluster(trajs_0,num_reps=REPLICATES) 


# Round 1
lc_1 = LeastCounts(root_path=ROOT_PATH, round_no=1)
st_lc_1 = lc_1.get_states(traj_list=trajs_0)
trajs_1_lc = lc_1.generate_data(md_steps=STEPS,initial_coords=st_lc_1,num_reps=5) 

## Evaluation 
ev = Evaluate(root_path=ROOT_PATH, round_no=1,num_reps=REPLICATES) 
r1_list, df = ev.rank_policies(trajs=[trajs_1_lc])


# Sampling

ROUND_REPS = 3
ROUNDS = range(2, 50,1)  # Adjust the range accordingly

exp_list = []
conv_list = []
tot_list = []
name_list = []

for _ in range(ROUND_REPS):
    r_list = r1_list
    best_exp = []
    best_conv = []
    best_tot = []
    best_name = []

    start = time.time()    
    for r_no in tqdm(ROUNDS):
        lc_obj = LeastCounts(root_path=ROOT_PATH, round_no=r_no)
        st_lc = lc_obj.get_states(traj_list=r_list)
        traj_lc = lc_obj.generate_data(md_steps=STEPS, initial_coords=st_lc, num_reps=REPLICATES)

        ev_obj = Evaluate(root_path=ROOT_PATH, round_no=r_no, num_reps=REPLICATES, prev_best_data=r_list)
        r_list, df = ev_obj.rank_policies(trajs=[traj_lc])

        best_exp.append(df.loc[ev_obj.best_policy_name][0])
        best_conv.append(df.loc[ev_obj.best_policy_name][1])
        best_tot.append(df.loc[ev_obj.best_policy_name][2])
        best_name.append(ev_obj.best_policy_name)
    
    end = time.time()
    TIME.append(end-start)
    
    
    exp_list.append(best_exp)
    conv_list.append(best_conv)
    tot_list.append(best_tot)
    name_list.append(best_name)

pickle.dump(exp_list,open(f'{ROOT_PATH}/analysis/exp_p1.pkl','wb'))    
pickle.dump(conv_list,open(f'{ROOT_PATH}/analysis/conv_p1.pkl','wb'))  
pickle.dump(tot_list,open(f'{ROOT_PATH}/analysis/tot_p1.pkl','wb'))  
