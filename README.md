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

