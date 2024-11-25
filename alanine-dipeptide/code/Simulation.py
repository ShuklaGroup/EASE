"""
Definition of a simulation class for alanine dipeptide
"""
import numpy as np
import os
import deeptime as dt
import scipy
from scipy import interpolate
from matplotlib.animation import FuncAnimation, PillowWriter, ArtistAnimation
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from deeptime.data import quadruple_well_asymmetric
from matplotlib.collections import LineCollection
import pickle
from tqdm import tqdm
from deeptime.clustering import KMeans
#from Utils import *
import openmm as mm
import openmm.app as app
from simtk.unit import *
import torch
import mdtraj as md


class ToySimulation:
    """
    Simulation object 
    """
    def __init__(self,root_path,top_file, round_no=0, n_steps= 100, forcefield="amber14/protein.ff14SB.xml", temp=300 * kelvin, press=1 * bar,
                 nonbondedMethod=app.NoCutoff, constraints=app.HBonds, collision_freq=1 / picosecond,
                 timestep=0.002 * picosecond, platform="CUDA",implicit_solvent="implicit/gbn2.xml"):
        """Constructor.
        """

        self.round_no = round_no
        self.timestep = timestep
        self.n_steps = n_steps
        self.root_path = root_path
        #self.model = dt.data.quadruple_well_asymmetric(h = self.timestep, n_steps = self.n_steps)
        self.top_file = top_file
        self.forcefield = forcefield
        self.temp = temp
        self.press = press  # Ignored for this class, but may be used in derived classes
        self.nonbondedMethod = nonbondedMethod
        self.constraints = constraints
        self.collision_freq = collision_freq
        self.timestep = timestep
       # if torch.cuda.is_available() and platform == "CUDA":
       #     print("Hehe CUDA is available and will be utilized, fast af lolxxx", flush=True)
        self.platform = platform if torch.cuda.is_available() else "CPU"
        self.implicit_solvent = implicit_solvent


    def _set_system(self):
        """Convenience function to set the system before running a simulation. The reason this is not done in __init__()
        is that this method creates a fresh Integrator object (otherwise we get an error because the OpenMM Integrator
        is already bound to a context).

        :return: None.
        """
        pdb = app.PDBFile(self.top_file)
        forcefield = app.ForceField(self.forcefield, self.implicit_solvent)
        system = forcefield.createSystem(pdb.topology,
                                         nonbondedMethod=self.nonbondedMethod,
                                         constraints=self.constraints)
        integrator = mm.LangevinIntegrator(self.temp, self.collision_freq, self.timestep)
        self.system = system
        self.topology = pdb.topology
        self.integrator = integrator
        self.init_pos = pdb.positions

    def _run_single(self,save_rate,rep,velocities=None,positions=None):


        """Runs an OpenMM simulation.

        :param positions: openmm.vec3.Vec3.
            Initial atomic positions for simulation.
        :param n_steps: int.
            Number of steps.
        :param save_rate: int.
            Save rate for trajectory frames.
        :param velocities: openmm.vec3.Vec3 (optional).
            Initial velocities. If not set, then they are sampled from a Boltzmann distribution.
        :return: None.
        """
        self._set_system()

        positions = self.init_pos
        path = f'{self.root_path}/round{self.round_no}/trajs/'
        file_path = path+f'rep{rep}.dcd'
        os.makedirs(path, exist_ok=True)
        self._set_system()
        mm.Platform.getPlatformByName(self.platform)
        simulation = app.Simulation(self.topology, self.system, self.integrator)
        simulation.context.setPositions(positions)
        if velocities:
            simulation.context.setVelocities(velocities)
        simulation.reporters.append(app.DCDReporter(file_path, save_rate))
        simulation.step(self.n_steps)

        return file_path

 
    def _compute_dihedral_features(self,traj):
        """
        Helper function to compute dihedrals (uses MDTraj library)
        """
        phi = md.compute_phi(traj)[1].reshape(-1, 1)
        psi = md.compute_psi(traj)[1].reshape(-1, 1)
        dihedral_features = np.concatenate((phi, psi), axis=1)

        return dihedral_features


    def _compute_dihedral_features_for_trajectories(self,traj_list):
        """
        Helper function to append computed dihedrals for the whole trajectory list.
        """
        dihedral_features_list = []
        for traj_file in traj_list:
            traj = md.load(traj_file, top=self.top_file)
            dihedral_features = self._compute_dihedral_features(traj)
            dihedral_features_list.append(dihedral_features)
        return dihedral_features_list


    def get_initial_data(self,num_reps,save_rate,velocities=None,positions=None):
        """ Get initial data (intended for round 0).

        :param num_reps: int.
            Number of replicates to run    
        :param save_rate: int.
            Frame save rate
        :param veclocities: Vec3 (OpenMM).
            Velocities to initialize (not used)    
        :param veclocities: Vec3 (OpenMM).
           Positions to initialize (not used)    
        :return: Python list.
            List of path to trajectories
        """

        traj_list = []
        for rep in tqdm(range(num_reps)):
            traj = self._run_single(save_rate=save_rate,rep=rep,velocities=velocities,positions=positions)
            traj_list.append(traj)
            
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
        vol = (self.round_no + 1)*num_reps * self.n_steps  
        c = int(np.sqrt(vol))

        if c < 50: c=50
        elif c>1000: c=1000
        print(f"States for round {self.round_no} are {c}")

        traj_path = f'{self.root_path}/round{self.round_no}/trajs/'
        file_path = traj_path+f'round{self.round_no}.ft'

        path = f'{self.root_path}/round{self.round_no}/clus_pkls/'
        os.makedirs(path, exist_ok=True)

        feats = self._compute_dihedral_features_for_trajectories(traj_list)
        pickle.dump(feats, open(file_path,'wb'))
       # print(f"haha dumped here {file_path}")

        cluster_obj = KMeans(n_clusters=c,  # place c cluster centers
                                init_strategy='kmeans++',  # kmeans++ initialization strategy
                                n_jobs=8,progress=tqdm,fixed_seed=True).fit_fetch(feats)


        dtrajs = [cluster_obj.transform(x) for x in feats]
        d_path = path+f"round{self.round_no}_dtraj.pkl"
        clus_path = path+f"round{self.round_no}_clusObj.pkl"

        pickle.dump(dtrajs, open(d_path,'wb'))
        pickle.dump(cluster_obj,open(clus_path,'wb'))

        return [file_path] 

