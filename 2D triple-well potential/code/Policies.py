"""
Definition of Policy classes for Adaptive Sampling
"""
import pickle
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from Utils import *
import deeptime as dt
from tqdm import tqdm
import os
from scipy import interpolate
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import scipy
from deeptime.markov.tools.analysis import eigenvalues
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MaximumLikelihoodMSM, BayesianMSM


class LeastCounts:
    """
    Least Counts sampling
    """

    def __init__(self, root_path, round_no):
        self.root_path = root_path
        self.round_no = round_no
        self.policy_name = "LeastCounts"


    def _select_states(self, clus, traj_list, states):
        """Select states according to Least Counts policy

        :param clus: str.
            Path to clustering object pickles.
        :param traj_list: Python list.
            List containing paths to trajectory pickles.
        :param states: int.
            Number of states to choose
        :return: Python list.
        	Returns list of selected state idx
        	.0
        """

        selected_states = []
        state_dict = self._center_states(clus, traj_list)

        clus_obj = pickle.load(open(clus, 'rb'))
        trajs = list_to_trajs(traj_list)
        dtrajs = [clus_obj.transform(x) for x in trajs]

        all_trajs = np.concatenate(trajs)
        all_dtrajs = np.concatenate(dtrajs)
        
        unique_clusters, cluster_sizes = np.unique(all_dtrajs, return_counts=True)

        least_counts_clusters = unique_clusters[np.argsort(cluster_sizes)[:states]]
        for cluster in least_counts_clusters:
            selected_states.append(state_dict[cluster])

        return selected_states

    def _center_states(self, clus, traj_list):
        """Creates a dictionary of representative states

        :param clus: str.
            Path to clustering object pickles.
        :param traj_list: Python list.
            List containing paths to trajectory pickles.
       
        :return: Python dict.
        	Returns dict. With cluster ids as keys and representative state idx as values.
        """

        traj = np.concatenate(list_to_trajs(traj_list))
        clus_obj = pickle.load(open(clus, 'rb'))
        clus_num = len(clus_obj.cluster_centers)

        states, dis = pairwise_distances_argmin_min(clus_obj.cluster_centers, traj)

        state_dict = {}
        for cl, st in zip(np.arange(0, clus_num), states):
            state_dict[cl] = st

        return state_dict

    def get_states(self, traj_list=None, states=5):
        """Converts selected idx to actual states.

        :param traj_list: Python list.
            List containing paths to trajectory pickles.
        :param states: int.
            Number of states to choose
        :return:
        	
        """
        path = f'{self.root_path}/round{self.round_no - 1}/clus_pkls/'
        clus = path+f"round{self.round_no -1}_clusObj.pkl"

        idx = np.array(self._select_states(clus, traj_list, states))
        traj = np.concatenate(list_to_trajs(traj_list))

        states = traj[idx]

        return states

    def generate_data(self, md_steps ,initial_coords,num_reps = 1,timestep = 1e-5, n_steps= 100, seed = None):
        """Generates trajectories.

        :param md_steps: int.
            Number of steps.
        :param initial_coords = numpy array. 
            Numpy array containing starting states
        :param num_reps = int, default = 1
            Number of replicates to run 
        :param seed = int, default = 42
            Seed to fix random parameters

        :return: list.
            List of pkls files of the trajectories.
        """

        if seed == None:
        	seed = np.random.randint(low=1, high=1000)



        self.model = dt.data.triple_well_2d(h =timestep, n_steps =n_steps)

        traj_list = []
        for coord, rep in tqdm(zip(initial_coords, range(num_reps))):
            traj =  self.model.trajectory(coord, md_steps, seed = seed + rep, n_jobs = 1)
            path = f'{self.root_path}/round{self.round_no}/{self.policy_name}/trajs/rep{rep}.out'
            os.makedirs(f'{self.root_path}/round{self.round_no}/{self.policy_name}/trajs/', exist_ok=True)
            pickle.dump(traj,open(path,'wb'))
            traj_list.append(path)

        return traj_list



    def plot_trajs(self, trajs,save_path=None):
        system = self.model
        trajs = list_to_trajs(trajs)

        x = np.arange(-2, 2, 0.01)
        y = np.arange(-1, 2, 0.01)
        xy = np.meshgrid(x, y)
        V = system.potential(np.dstack(xy).reshape(-1, 2)).reshape(xy[0].shape)

        fig, ax = plt.subplots(1, 1)
        ax.set_title(f"Round {self.round_no}, {self.policy_name}")

        cb = ax.contourf(x, y, V, levels=20, cmap='coolwarm')

        # Define custom colors for each trajectory
        custom_colors = ['red', 'green', 'blue', 'orange', 'purple']

        for idx, (traj, color) in enumerate(zip(trajs, custom_colors)):

            x = np.r_[traj[:, 0]]
            y = np.r_[traj[:, 1]]
            f, u = scipy.interpolate.splprep([x, y], s=0, per=False)
            xint, yint = scipy.interpolate.splev(np.linspace(0, 1, 50000), f)

            points = np.stack([xint, yint]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            coll = LineCollection(segments, color=color)
            coll.set_linewidth(1)
            coll.set_label(f'Trajectory {idx + 1}')  # Added for legend
            ax.add_collection(coll)

            # Plot star at the starting point
            ax.scatter(traj[0, 0], traj[0, 1], marker='*', color='black', s=100, zorder=10)

        fig.colorbar(cb)

        # Added for legend
        ax.legend()
        if save_path:
        	plt.savefig(save_path,dpi=300)
        plt.show()

    def plot_trajs_animation(self, trajs):
        system = self.model
        trajs = [pickle.load(open(f, 'rb')) for f in trajs]

        x = np.arange(-2, 2, 0.01)
        y = np.arange(-1, 2, 0.01)
        xy = np.meshgrid(x, y)
        V = system.potential(np.dstack(xy).reshape(-1, 2)).reshape(xy[0].shape)

        fig, ax = plt.subplots(1, 1)
        ax.set_title("Trajectories Animated on the Potential Landscape")

        cb = ax.contourf(x, y, V, levels=20, cmap='coolwarm')

        custom_colors = ['red', 'green', 'blue', 'orange', 'purple']

        lines = []
        step_text = ax.text(0.02, 0.02, '', transform=ax.transAxes, fontsize=16)  # Text annotation for step count

        for traj, color in zip(trajs, custom_colors):
            line, = ax.plot([], [], color=color, label=f'Trajectory')
            lines.append(line)

            # Plot star at the starting point
            ax.scatter(traj[0, 0], traj[0, 1], marker='*', color='black', s=100, zorder=10)

        def update(frame):
            for idx, (line, traj) in enumerate(zip(lines, trajs)):
                x = traj[:frame, 0]
                y = traj[:frame, 1]
                line.set_data(x, y)

            # Update step count text
            step_text.set_text(f'Step: {frame + 1}/{len(trajs[0])}')

            return lines + [step_text]

        ani = FuncAnimation(fig, update, frames=len(trajs[0]), interval=10, repeat=False)

        fig.colorbar(cb)
        ax.legend()

        # Ensure that the animation object persists
        plt.show()

        return ani


class RandomSampling(LeastCounts):
    """
    Random sampling
    """

    def __init__(self, root_path, round_no):
        super().__init__(root_path, round_no)
        self.policy_name = "RandomSampling"

    def _select_states(self, clus, traj_list, states):
        """Select states according to Random Sampling policy

        :param clus: str.
            Path to clustering object pickles.
        :param traj_list: Python list.
            List containing paths to trajectory pickles.
        :param states: int.
            Number of states to choose
        :return: Python list.
            Returns list of selected state idx.
        """

        selected_states = []
        state_dict = self._center_states(clus, traj_list)

        clus_obj = pickle.load(open(clus, 'rb'))
        trajs = list_to_trajs(traj_list)
        dtrajs = [clus_obj.transform(x) for x in trajs]

        all_trajs = np.concatenate(trajs)
        all_dtrajs = np.concatenate(dtrajs)

        unique_clusters, _ = np.unique(all_dtrajs, return_counts=True)
        all_states = np.arange(len(unique_clusters))

        # Shuffle the list of states randomly
        np.random.shuffle(all_states)

        # Select the desired number of states
        random_selected_states = all_states[:states]

        for cluster in random_selected_states:
            selected_states.append(state_dict[cluster])

        return selected_states

class LambdaSampling(LeastCounts):
    """
    Lambda sampling
    """

    def __init__(self, root_path, round_no):
        super().__init__(root_path, round_no)
        self.policy_name = "LambdaSampling"

    def _select_states(self, clus, traj_list, states):
        """Select states according to Random Sampling policy

        :param clus: str.
            Path to clustering object pickles.
        :param traj_list: Python list.
            List containing paths to trajectory pickles.
        :param states: int.
            Number of states to choose
        :return: Python list.
            Returns list of selected state idx.
        """

        selected_states = []
        state_dict = self._center_states(clus, traj_list)

        clus_obj = pickle.load(open(clus, 'rb'))
        trajs = list_to_trajs(traj_list)
        dtrajs = [clus_obj.transform(x) for x in trajs]

        cm, msm = self._make_msm(dtrajs=dtrajs)

        best_i = self._calculate_best_state(cm=cm, msm=msm)
        selected_states = [state_dict[best_i]]*states

        return selected_states 


    def _make_msm(self, dtrajs, best_lag = 10): #, lag = 10):

        #lags = np.arange(1, 40, 1)
        #models = []

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
        #plt.show()        
        #fig.savefig(f'ground_truth_msm/its_plots/ITS_plot_clus{k}.jpg',dpi=300)

        count_model = TransitionCountEstimator(lagtime = best_lag,count_mode = 'sliding').fit_fetch(dtrajs)
        msm = MaximumLikelihoodMSM().fit_fetch(count_model.submodel_largest())

        return count_model , msm

    def _calculate_best_state(self,cm,msm):

        t_ij = msm.transition_matrix
        num_states = t_ij.shape[0]
        evs = np.real(eigenvalues(t_ij))
        e = evs[1] 
        m=5
        A = t_ij - e*np.identity(num_states)
        P,L,U = scipy.linalg.lu(A)
        
        zero = U[0:-1,-1]
        U = U[0:-1,0:-1]
        e_k = np.zeros((num_states))
        e_k[-1] = 1
        
        x_a = np.linalg.solve(U,zero)
        x_a = np.append(x_a,[1])

        x = np.linalg.solve(L.T,e_k)
        
        sens_mat = np.outer(x,x_a)/np.dot(x, x_a)
        
        diff_arr = []
        var_arr = []
        
        for s in range(num_states):
        
            p_i = t_ij[s,:]
            s_i = sens_mat[:,s]
            w_i = cm.count_matrix[s,:].sum()
        
            qc = np.diag(p_i) - np.outer(p_i,p_i)
            qr = np.matmul(qc,s_i)
            q_i = np.dot(s_i, qr)
        
            diff = ((q_i/(w_i+1))-(q_i/(w_i+m+1)))
            diff_arr.append(diff)
            

            var = ((q_i/(w_i+1)))
            var_arr.append(var)

        print(sum(var_arr))
        best_i = np.argmax(diff_arr)

        return best_i