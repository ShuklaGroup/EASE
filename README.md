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


## **Code Description**  
The code implements the **policy ranking** framework described in the paper. Here, we describe the source files, the classes implemented, and major routines for the benefit of the users, in the context of the 2D potentials.  

### **Sampling Scripts**  
- **`single_LC.py`**: Implements the sampling run for the Least Counts policy.  
- **`single_RS.py`**: Implements the sampling run for the Random Sampling policy.  
- **`single_LD.py`**: Implements the sampling run for the Lambda Sampling policy.  
- **`main_betas.py`**: Implements the sampling run for the policy ranking framework.  

### **Core Scripts**  
#### **`Simulation.py`**: Implements the simulation class for the physical system.  
- **Class: `ToySimulation`**  
  - **`get_initial_data()`**: Generates initial simulation (round=0) data for the system.  
  - **`cluster()`**: Performs clustering of the trajectory data.  
  - **`plot_trajs()`**: Plots trajectories for the 2D potential.  

#### **`Policies.py`**: Contains implementations of the adaptive sampling policies.  
- **Class: `LeastCounts`**  
  - **`_center_states()`**: Creates a dictionary of representative states. In simpler words, it defines which states belong to which cluster.  
  - **`_select_states()`**: Implements the algorithm logic; for Least Counts, it chooses the index of the least visited states.  
  - **`get_states()`**: Converts the chosen state indices to states.  
  - **`generate_data()`**: Generates trajectory data for the system.  
- **Class: `RandomPolicy`**  
- **Class: `LambdaSampling`**  

> **Note:** Both `RandomPolicy` and `LambdaSampling` (subclasses) inherit from `LeastCounts` (parent class). Therefore, only **`_select_states()`** needs to be implemented for these classes, along with any necessary helper routines.  

#### **`Analysis.py`**: Implements the analysis class for the system.  
- **Class: `Evaluate`**  
  - **`_measure_exploration()`**: Computes the exploration metric (ratio of visited states to the total number of states).  
  - **`_rel_entropy()`**: Implements relative entropy as in Eq. (3) of the paper.  
  - **`_make_msm()`**: Constructs a Markov state model at each round for comparison with the ground truth model. The number of states is enforced to ensure consistency, and a uniform prior is added.  
  - **`_metrics()`**: Compiles metrics and implements the objective function according to the beta value (exploitation vs. exploration).  
  - **`rank_policies()`**: Ranks the policies using the previously defined routines.

## **Quick Start Guide**  
```ruby
adrija = candidate
```
