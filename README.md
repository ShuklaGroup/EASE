# Optimizing Adaptive Sampling via Policy Ranking

Code for ensemble ranking-based adaptive sampling. This repository contains the **code implementation** for the paper:  
**[Optimizing Adaptive Sampling via Policy Ranking](https://arxiv.org/pdf/2410.15259)**  
*(arXiv:2410.15259v1 [q-bio.BM], October 20, 2024)*  
![Framework Overview](figures/fig_2.png)
This work introduces a **modular framework** for adaptive sampling, utilizing **metric-driven ranking** to dynamically identify the most effective sampling policies. By systematically evaluating and ranking an ensemble of policies, this approach enables **policy switching across rounds**, significantly improving convergence speed and sampling performance.

## Highlights:
- **Adaptive Sampling Ensemble:** Dynamically selects the optimal policy based on real-time data.
- **Improved Efficiency:** Outperforms single-policy sampling by exploring conformational space more effectively.
- **Versatility:** Integrates any adaptive sampling policy, making it highly flexible and modular.
- **On-the-Fly Algorithms:** Includes novel algorithms to approximate ranking and decision-making during simulations.


## Code Description:
The code implements the **policy ranking** framework described in the paper. Here, we describe the source files, the classes implemented and major routines for the benefit of the users, in context of the 2D potentials.

**Sampling scripts**:
-`single_LC.py`: This script implements the sampling run for Least Counts policy. 
-`single_RS.py`: This script implements the sampling run for Random Sampling policy. 
-`single_LD.py`: This script implements the sampling run for Lambda Sampling policy. 
-`main_betas.py': This script implements the sampling run for the policy ranking framework.  

**Core scripts**:
`Simulation.py`: This script implements the simulation class for the physical system.
 -`Class: ToySimulation`
        -`func: get_initial_data`: generates initial simulation (round=0) data for the system.
        -`func: cluster`: performs clustering of the trajectory data.
       - `func: plot_trajs`: plots trajectories for the 2D potential.

`Policies.py`: This script contains implementations of the adaptive sampling policies.
 -`Class: Least Counts` 
        -`func: _center_states`: creates a dictionary of representative states. In simpler words it defines which states belong to which cluster.
        -`func: _select_states`: this is the routine that implements the algorithm logic, i.e. for Least Counts it chooses the idx of the least visited states.
        -`func: get_states`: Converts the chosen state idx to states.
       - `func: generate_data`: generates trajectory data for the system.
 -`Class: Random Policy`
 -`Class: Lambda Sampling`
 Note: Both Random Policy and Lambda Sampling (daughter classes) inherit from Least Counts (parent class), therefore only `func: _select_states` needs to be implemented for these classes, besides any helper routines for it. 

`Analysis.py`: This script implements the analysis class for the system.
 -`Class: Evaluate`
        -`func: _measure_exploration': computes the exploration metric (ratio of visited states to the total number of states).
        -`func: _rel_entropy` : implementation of relative entropy as in eq(3) of the paper.
        -`func: _make_msm`: makes a markov state model at each round to compare to the ground truth model, also note that the number of states is enforced to ensure comparison, in addition a uniform prior is added.
        -`func: _metrics`: routine to compile the metrics and implement the objective function according to the beta value (exploitation vs exploration).
        -`func: rank_policies`: routine to rank the policies using the previously defined routines.
        

 
