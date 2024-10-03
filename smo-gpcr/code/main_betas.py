from Simulation import OSSimulation
from Policies import LeastCounts, RandomSampling, LambdaSampling,MaxEntSampling
from Analysis import Evaluate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Utils import*
from tqdm import tqdm
import os


# Round 0
#z = 2.576
ROOT_PATH = "sim_output_betas_new" 
REPLICATES = 5
STEPS = 5




ROUND_REPS = 30
ROUNDS = range(1, 21,1)  # Adjust the range accordingly
# Sampling


betas = [0.5]#,0.0,0.2,0.4,0.6,0.8,1.0]
STEPS=80

import time  # Import for adding delay
import os

for BETA in betas:
    exp_list = []
    conv_list = []
    tot_list = []
    name_list = []

    for _ in tqdm(range(ROUND_REPS)):
        SEED = np.random.randint(low=1, high=1e6)
        print(f"=============================ROUND 0===========================================")

        sim_0 = OSSimulation(root_path = ROOT_PATH, round_no=0)
        trajs_0 = sim_0.get_initial_data(mc_steps=STEPS, num_reps=REPLICATES, seed=SEED)

        r_t_list = trajs_0
        best_exp = []
        best_conv = []
        best_tot = []
        best_name = []
        prev_best_data = None

        for r_no in tqdm(ROUNDS):
            success = False  # Flag to indicate whether the round completed successfully
            while not success:
                try:
                    print(f"============================={r_no}===========================================")
                    print(f"BETA = {BETA}")
                    SEED = np.random.randint(low=1, high=1e6)

                    lc_obj = LeastCounts(root_path=ROOT_PATH, round_no=r_no)
                    st_lc_obj = lc_obj.get_states(traj_list=r_t_list, states=REPLICATES)
                    trajs_obj_lc = lc_obj.generate_data(mc_steps=STEPS, start_states=st_lc_obj, num_reps=REPLICATES, seed=SEED)

                    rs_obj = RandomSampling(root_path=ROOT_PATH, round_no=r_no)
                    st_rs_obj = rs_obj.get_states(traj_list=r_t_list, states=REPLICATES)
                    trajs_obj_rs = rs_obj.generate_data(mc_steps=STEPS, start_states=st_rs_obj, num_reps=REPLICATES, seed=SEED)

                    ld_obj = LambdaSampling(root_path=ROOT_PATH, round_no=r_no)
                    st_ld_obj = ld_obj.get_states(traj_list=r_t_list, states=REPLICATES)
                    trajs_obj_ld = ld_obj.generate_data(mc_steps=STEPS, start_states=st_ld_obj, num_reps=REPLICATES, seed=SEED)

                    ev_obj = Evaluate(root_path=ROOT_PATH, round_no=r_no, num_reps=REPLICATES, prev_best_data=prev_best_data)
                    r_t_list, df = ev_obj.rank_policies(trajs=[trajs_obj_lc, trajs_obj_rs, trajs_obj_ld], beta=BETA)
                    prev_best_data = r_t_list

                    best_exp.append(df.loc[ev_obj.best_policy_name][0])
                    best_conv.append(df.loc[ev_obj.best_policy_name][1])
                    best_tot.append(df.loc[ev_obj.best_policy_name][2])
                    best_name.append(ev_obj.best_policy_name)

                    success = True  # If no exception, mark round as successful

                # in case of errors of sampling per round is not enough for a converged MSM definition, would not occur in real simulations
                except np.linalg.LinAlgError as e:
                    print(f"Linear Algebra Error in round {r_no}: {e}. Retrying after delay...")
                    #time.sleep(5)  # Add a 5-second delay before retrying
                except ValueError as e:
                    print(f"Value Error in round {r_no}: {e}. Retrying after delay...")
                    #time.sleep(5)
                except Exception as e:
                    print(f"An unexpected error occurred in round {r_no}: {e}. Retrying after delay...")
                    #time.sleep(5)

        exp_list.append(best_exp)
        conv_list.append(best_conv)
        tot_list.append(best_tot)
        name_list.append(best_name)

    os.makedirs(f'{ROOT_PATH}/analysis/', exist_ok=True)
    time.sleep(1)
    pickle.dump(exp_list, open(f'{ROOT_PATH}/analysis/exp_{BETA}.pkl', 'wb'))
    time.sleep(1)
    pickle.dump(conv_list, open(f'{ROOT_PATH}/analysis/conv_{BETA}.pkl', 'wb'))
    time.sleep(1)
    pickle.dump(tot_list, open(f'{ROOT_PATH}/analysis/tot_{BETA}.pkl', 'wb'))
    time.sleep(1)
    pickle.dump(name_list, open(f'{ROOT_PATH}/analysis/name_{BETA}.pkl', 'wb'))
    time.sleep(1)
