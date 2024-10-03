"""
Definition of a simulation class for toy potential
"""
import numpy as np
import os
import deeptime as dt
import scipy
from scipy import interpolate
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
from deeptime.data import triple_well_2d
from matplotlib.collections import LineCollection
import pickle
from tqdm import tqdm
from deeptime.clustering import KMeans
from Utils import *

from matplotlib.colors import ListedColormap
purple=tuple(['#6247aa','#815ac0','#a06cd5','#b185db','#d2b7e5'])
blue=tuple(['#2c7da0','#468faf','#61a5c2','#89c2d9','#a9d6e5'])
green=tuple(['#718355', '#87986a', '#97a97c', '#a3b18a', '#cfe1b9'])
orange=tuple(['#ffb700','#ffc300','#ffd000','#ffdd00','#ffea00'])
red=tuple(['#f25c54', '#f27059', '#f4845f', '#f79d65', '#f7b267'])
larger=tuple(['#f7b267','#f7b267','#f7b267','#f7b267','#f7b267'])
anh_colors = purple+blue+green+orange+red+larger
anh_cmap = ListedColormap(anh_colors)

class ToySimulation:
    """
    Simulation object triple-well 2D 
    """
    def __init__(self,root_path, round_no=0, timestep = 1e-5, n_steps= 50):
        """Constructor.

        :param round_no: int.
            Round number of adaptive sampling. 
        :param timestep: float * unit, default = 1e-15, The implementation uses an Euler-Maruyama integrator. (Deeptime Lib)
            Integration timestep  
        :param n_steps: int, default = 100
            Number of integration steps between each evaluation. 
        :param root_path: str.
            Path to save round output
        """

        self.round_no = round_no
        self.timestep = timestep
        self.n_steps = n_steps
        self.root_path = root_path
        self.model = dt.data.triple_well_2d(h = self.timestep, n_steps = self.n_steps)


    def get_initial_data(self, md_steps,initial_coords,num_reps = 1, seed = 42):
        """Generates initial (round=0) data.


        :param md_steps: int.
            Number of steps.
        :param initial_coords = numpy array. default = np.array([[-1,0]])
            Numpy array containing starting states
        :param num_reps = int, default = 1
            Number of replicates to run 
        :param seed = int, default = 42
            Seed to fix random parameters

        :return: list.
            List of pkls files of the trajectories.
        """
        self.md_steps=md_steps
        traj_list = []
        for coord, rep in tqdm(zip(initial_coords, range(num_reps))):
            traj =  self.model.trajectory(coord, self.md_steps, seed = seed + rep, n_jobs = 1)
            path = f'{self.root_path}/round{self.round_no}/trajs/rep{rep}.out'
            os.makedirs(f'{self.root_path}/round{self.round_no}/trajs/', exist_ok=True)
            pickle.dump(traj,open(path,'wb'))
            traj_list.append(path)

        return traj_list


    def cluster(self, traj_list,num_reps):
        """Cluster the trajectories.

        :param traj_list: Python list.
            List containing paths to trajectory pickles.
        :param num_reps = int.
            Number of replicates to run    
        :return  clus_path, d_path: Python tuple.
            Path to cluster object, path to dtraj
    
            
            """    
        vol = num_reps * self.md_steps 

        c = int(np.sqrt(vol))


        if c < 50: c=50
        elif c>100: c=100

        path = f'{self.root_path}/round{self.round_no}/clus_pkls/'
        os.makedirs(path, exist_ok=True)


        trajs = [pickle.load(open(t,'rb')) for t in traj_list]

        cluster_obj = KMeans(n_clusters=c,  # place c cluster centers
                                init_strategy='kmeans++',  # kmeans++ initialization strategy
                                n_jobs=8,progress=tqdm,fixed_seed=True).fit_fetch(np.concatenate(trajs))

        dtrajs = [cluster_obj.transform(x) for x in trajs]
        d_path = path+f"round{self.round_no}_dtraj.pkl"
        clus_path = path+f"round{self.round_no}_clusObj.pkl"

        pickle.dump(dtrajs, open(d_path,'wb'))
        pickle.dump(cluster_obj,open(clus_path,'wb'))

        return clus_path, d_path

    def plot_trajs(self, trajs):
        system = self.model
        trajs = [pickle.load(open(f,'rb')) for f in trajs]

        x = np.arange(-2, 2, 0.01)
        y = np.arange(-1.5, 2.8, 0.01)
        xy = np.meshgrid(x, y)
        V = system.potential(np.dstack(xy).reshape(-1, 2)).reshape(xy[0].shape)

        fig, ax = plt.subplots(1, 1)
        #ax.set_title("Example of trajectories in the potential landscape")

        cb = ax.contourf(x, y, V, levels=20, cmap=anh_cmap)

        # Define custom colors for each trajectory
        custom_colors = ['white']

        for idx, (traj, color) in enumerate(zip(trajs, custom_colors)):

            x = np.r_[traj[:, 0]]
            y = np.r_[traj[:, 1]]
            f, u = scipy.interpolate.splprep([x, y], s=0, per=False)
            xint, yint = scipy.interpolate.splev(np.linspace(0, 1, 50000), f)

            points = np.stack([xint, yint]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            coll = LineCollection(segments, color=color,alpha=0.4)
            coll.set_linewidth(1)
            #coll.set_label(f'Trajectory {idx + 1}')  # Added for legend
            ax.add_collection(coll)

            # Plot star at the starting point
            ax.scatter(traj[0, 0], traj[0, 1], marker='*', color='black', s=100, zorder=10)

        fig.colorbar(cb)

        # Added for legend
        #ax.legend()
        plt.savefig("research_plots/ref.png",dpi=400)
        plt.show()

    def plot_trajs_animation(self, trajs):
        system = self.model
        trajs = [pickle.load(open(f, 'rb')) for f in trajs]

        x = np.arange(-2, 2, 0.01)
        y = np.arange(-1, 2, 0.01)
        xy = np.meshgrid(x, y)
        V = system.potential(np.dstack(xy).reshape(-1, 2)).reshape(xy[0].shape)

        fig, ax = plt.subplots(1, 1)
        ax.set_title("Round 0 initial simulations")

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
        ani.save("lunch/policy_group_lunch.gif", dpi=300, writer=PillowWriter(fps=25))
        print('hahaha')

        fig.colorbar(cb)
        ax.legend()

        # Ensure that the animation object persists
        plt.show()

        return ani


