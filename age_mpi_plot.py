from mpi4py import MPI
import matplotlib.pyplot as plt
import sys
from os import listdir
import heapq
import matplotlib.style as style


def get_succeed_ranks(dir_name):
    succeed_ranks = []
    for f in listdir(dir_name):
        f = f.replace('_','.')
        if "complete" in f.split('.'):
            succeed_ranks.append(int(f.split('.')[0]))
    return succeed_ranks
    #return [10]
    #return [int(f.split('.')[0]) for f in listdir(dir_name)]

if __name__ == "__main__":
    print('Hi')
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    dir_name = "128_1616444118.8"
    age_dir_name = "128_1616444118.8_age"
    RB_size = [1000000]
    batch_size = 128
    num_eval_episods = 100
    avg_period_min = 4
    total_run_min = 240
    
    succeed_ranks = get_succeed_ranks(age_dir_name)
    
    if rank in succeed_ranks:
        print("Hi ", rank)
        sys.stdout.flush()
        with open("%s/%d.txt"%(dir_name,rank)) as f:
            rewards, temp_rewards = [], []
            lengths, temp_lengths = [], []
            for line in f:
                if line in ["FCFS\n","LCFS\n","1 out of 2\n","1 out of 5\n","1 out of 20\n","1 out of 100\n","stop and go\n"]:
                    if temp_rewards:
                        rewards.append(temp_rewards)
                        lengths.append(temp_lengths)
                        temp_rewards = []
                        temp_lengths = []
                else:
                    temp_rewards.append(float(line.split(',')[0]))
                    temp_lengths.append(float(line.split(',')[1].rstrip('\n')))
            rewards.append(temp_rewards)
            lengths.append(temp_lengths)
        #comm.send([rewards,lengths],dest=0)
        num_steps = []
        with open("%s_age/%d_complete.txt"%(dir_name,rank)) as ff:
            age, temp_age = [], []
            for line in ff:
                if line in ["FCFS\n","LCFS\n","1 out of 2\n","1 out of 5\n","1 out of 20\n","1 out of 100\n","stop and go\n"]:
                    if temp_age:
                        age.append(temp_age)
                        temp_age = []
                elif "Number of steps" in line:
                    num_steps.append(int(line.split(':')[1]))
                else:
                    temp_age.append(int(line))
            age.append(temp_age)
        comm.send([rewards,lengths,age,num_steps],dest=0)

    if rank == 0:
        colors = ['crimson','darkolivegreen','darkgoldenrod','darkcyan','cornflowerblue','chocolate']
        all_rewards = []
        all_lengths = []
        concat_age = []
        num_updates = []
        learner_policies = ["FCFS","LCFS","1 out of 2","1 out of 5"]
        #learner_policies = ["stop and go"]
        cnt_valid_data = 0
        print('begin')
        sys.stdout.flush()
        for node in succeed_ranks:
            msg = comm.recv(source=node)
            rewards, lengths, age, num_steps = msg[0], msg[1], msg[2], msg[3]
            print("num steps", num_steps)
            concat_age.append(age)
            num_updates.append([len(x) for x in age])
            if not all_rewards and len(rewards)==len(learner_policies):
                all_rewards = rewards
                all_lengths = lengths
                num_steps_sum = num_steps
                cnt_valid_data += 1
            elif len(rewards)==len(learner_policies):
                print(len(all_rewards[0]),len(rewards[0]))
                all_rewards = [[x+y for (x,y) in zip(all_rewards[i],rewards[i])] for i in range(len(learner_policies))]
                all_lengths = [[x+y for (x,y) in zip(all_lengths[i],lengths[i])] for i in range(len(learner_policies))]
                num_steps_sum = [x+y for (x,y) in zip(num_steps_sum,num_steps)]
                cnt_valid_data += 1
        avg_steps = [x/len(succeed_ranks) for x in num_steps_sum]
        avg_num_updates = [sum([num_updates[i][j] for i in range(len(num_updates))])/len(num_updates) for j in range(len(learner_policies))]
        min_age_length = min([min([len(x) for x in age]) for age in concat_age])
        all_ages = []
        #print([len(concat_age[20][x]) for x in range(len(learner_policies))])
        for k in range(len(learner_policies)):
            all_ages.append([sum([concat_age[i][k][j] for i in range(len(concat_age))])/len(concat_age) for j in range(min_age_length)])
        HP = ["Windy GridWorld","1-1/t",.99,.1]
        all_rewards = [[x/cnt_valid_data for x in y] for y in all_rewards]
        all_lengths = [[x/cnt_valid_data for x in y] for y in all_lengths]
        episodes = [x*5 for x in range(len(all_rewards[0]))]
        print('before')
        sys.stdout.flush()
        plt.figure()
        style.use('seaborn')
        for i in range(len(learner_policies)):
            plt.plot(episodes,all_rewards[i],'-',color=colors[i],lw=2,label=learner_policies[i])
        #plt.plot(episodes,all_lengths,'-',color=colors[i],lw=.25,alpha=.2)
        #plt.plot(avg_times[2:],high_conf[i][3:],'-',color=colors[i],lw=.25,alpha=.2)
        #plt.fill_between(avg_times[2:],low_conf[i][3:],high_conf[i][3:],color=colors[i],alpha=.1)
        plt.legend()
        plt.xlabel("Episodes")
        plt.ylabel("Episode Rewards")
        #plt.grid(True)
        #left = int(0.05*len(episodes))
        #right = int(0.9*len(episodes))
        plt.yscale('symlog')
        plt.grid(True) 
        #plt.title(r'Env: %s,$\epsilon=1-1/t$,$\gamma$=%.2f,$\alpha$=%.2f'%(HP[0],HP[2],HP[3]))
        plt.savefig("%s_rewards.pdf"%dir_name)

        plt.figure()
        time_steps = [x*10 for x in range(min_age_length)]
        for i in range(len(learner_policies)):
            plt.plot(time_steps,all_ages[i],'-',color=colors[i],lw=2,label=learner_policies[i])
        plt.legend()
        plt.xlabel("Time steps")
        plt.ylabel("Average age of an update")
        #plt.ylim(bottom=0,top=20)
        plt.yscale('log')
        plt.grid(True)
        #plt.title(r'Env: %s,$\epsilon=1-1/t$,$\gamma$=%.2f,$\alpha$=%.2f'%(HP[0],HP[2],HP[3]))
        plt.savefig("%s_age.pdf"%dir_name)

        plt.figure()
        plt.bar(learner_policies,avg_num_updates,color=[colors[i] for i in range(len(learner_policies))])
        plt.ylabel("Average number of updates")
        #plt.title(r'Env: %s,$\epsilon=1-1/t$,$\gamma$=%.2f,$\alpha$=%.2f'%(HP[0],HP[2],HP[3]))
        plt.savefig("%s_updates.pdf"%dir_name)

        plt.figure()
        plt.bar(learner_policies,avg_steps,color=[colors[i] for i in range(len(learner_policies))])
        plt.ylabel("Average number of steps")
        #plt.title(r'Env: %s,$\epsilon=1-1/t$,$\gamma$=%.2f,$\alpha$=%.2f'%(HP[0],HP[2],HP[3]))
        plt.savefig("%s_steps.pdf"%dir_name)
        
