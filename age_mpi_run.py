from mpi4py import MPI
from time import time
from os import mkdir, rename
import adopted_RL_experiment
import adopted_RL_experiment_fixed_training_env
import networked_RL
import age_RL
import sys
import qlearning
import Qlearning_multithread 
import Qlearning_centeralized
import sarsa_multithread
import Qlearning_withQueue

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    batch_size = 128

    if rank == 0:
        print(comm.Get_size())
        sys.stdout.flush()
        start_time = time()
        dir_name = '%d_%.1f'%(batch_size,start_time)
        dir_name_age = '%d_%.1f_age'%(batch_size,start_time)
        mkdir(dir_name)
        mkdir(dir_name_age)

    if rank != 0:
        dir_name = None
        max_eps = .1
        env = 'WindyGridworld-v0'
        num_episodes = 1000
        gamma = .99
        alpha = .5
    
    dir_name = comm.bcast(dir_name,root=0)
    
    if rank != 0:
        print('hey this is me ',rank)
        learner_policies = ["FCFS","LCFS","1 out of 2","1 out of 5"]
        #learner_policies = ["stop and go"]
        sys.stdout.flush()
        for policy in learner_policies:
            print("******************** New Policy *****************",policy)
            Q = Qlearning_withQueue.Qlearning(env,max_eps,num_episodes,gamma,alpha,policy)
            episode_rewards = Q.stats.episode_rewards[::5]
            episode_lengths = Q.stats.episode_lengths[::5]
            learner_age = Q.learner_age[::10]
            #myPlotting.plot_episode_stats(Q.stats)
            with open("%s/%d.txt"%(dir_name,rank), 'a+') as f:
                f.write("%s\n"%policy)
                for reward, length in zip(episode_rewards,episode_lengths):
                    f.write("%.2f,%2f\n"%(reward,length))
            with open("%s_age/%d.txt"%(dir_name,rank), 'a+') as ff:
                ff.write("%s\n"%policy)
                ff.write("Number of steps:%d\n"%Q.step_cnt)
                for item in learner_age:
                    ff.write("%d\n"%item)
        rename("%s_age/%d.txt"%(dir_name,rank),"%s_age/%d_complete.txt"%(dir_name,rank))
                    
