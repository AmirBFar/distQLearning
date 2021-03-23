import gym
import numpy as np
import itertools
from collections import defaultdict
import myPlotting
import gym_gridworlds
import gym_maze
from gym.envs.toy_text.blackjack import BlackjackEnv
from gym.envs.registration import register
import time, threading
import sys

env = register(
            id='maze-sample-10x10-v1',
            entry_point='gym_maze.envs:MazeEnvSample10x10',
            max_episode_steps=10000,
            kwargs={'enable_render': False}
        )

def epsilon_greedy(Q,num_actions):
    
    def policy(state,eps):
        action_probs = [eps/(num_actions) for _ in range(num_actions)]
        greedy_action = Q[state].index(max(Q[state]))
        action_probs[greedy_action] += 1-eps
        return action_probs
    return policy

class Qlearning():
    def __init__(self,env,eps,num_episodes,gamma,alpha,policy):

        self.env = gym.make(env)
        #self.env.enable_render = False
        self.eps = eps
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.alpha = alpha
        self.learner_policy = policy
        self.Q = defaultdict(lambda: [0 for _ in range(self.env.action_space.n)])
        self.temp_buffer = []
        self.stats = myPlotting.EpisodeStats(
            episode_lengths = np.zeros(self.num_episodes),
            episode_rewards = np.zeros(self.num_episodes))
        self.flag = True
        self.step_cnt = 0
        self.learner_age = []
        #actor_thread = threading.Thread(target=self.qLearning)
        learner_thread = threading.Thread(target=self.learner)
        actor_thread = threading.Thread(target=self.qLearning)
        learner_thread.start()
        actor_thread.start()
        
        actor_thread.join()
        learner_thread.join()

    def learner(self):
        update_cnt = 0
        update_age = 0
        last_update_time_step = 0
        while self.flag:
            #print("I am learner")
            if self.temp_buffer:
                #print("update",update_cnt,"learner's age",self.step_cnt-last_update_time_step,"time step",self.step_cnt,"last",last_update_time_step)
                print('This is ',self.learner_policy)
                sys.stdout.flush()
                if self.learner_policy == "LCFS":
                    exp = self.temp_buffer.pop()
                    #print('This is '%self.learner_policy)
                else:
                    exp = self.temp_buffer.pop(0)
                #exp = self.temp_buffer.pop(-1)
                #self.learner_age.append(self.step_cnt-last_update_time_step)
                #print("update count is", last_update_time_step)
                #sys.stdout.flush()
                next_greedy_action = self.Q[exp[3]].index(max(self.Q[exp[3]]))                                                                                                                    
                td_target = exp[2] + self.gamma*self.Q[exp[3]][next_greedy_action]                                                                                                                           
                td_delta = td_target - self.Q[exp[0]][exp[1]]                                                                                                                                                    
                self.Q[exp[0]][exp[1]] = (1-self.alpha)*self.Q[exp[0]][exp[1]] + self.alpha*td_target
                #self.learner_age.append(self.step_cnt-last_update_time_step)
                update_time_step = exp[4]
                if update_time_step > 4:
                    time.sleep(.001)
                #print("update step is", update_time_step)
                #print("update step is", update_time_step, "and actor step is", self.step_cnt," and age of update is",self.step_cnt-update_time_step)
                print("update count is",update_cnt)
                sys.stdout.flush()
                update_age = self.step_cnt-update_time_step
                self.learner_age.append(self.step_cnt-update_time_step)
                update_cnt += 1
                #time.sleep(.001)
            #time.sleep(.2)
        print('The total number of updates is ',update_cnt)
        

    def qLearning(self):
        policy = epsilon_greedy(self.Q,self.env.action_space.n)
        print("I am actor")
        #self.step_cnt = 0
        if self.learner_policy not in ["FCFS","LCFS"]:
            window = int(self.learner_policy.split(' ')[3])
        for i in range(self.num_episodes):
            if i%10 == 0:
                print('episode %d started'%i)
                print('buffer length is',len(self.temp_buffer))
            #state = tuple(self.env.reset()) ### for the environments with mutable state representation
            state = self.env.reset()
            for t in itertools.count():
                action_probs = policy(state,self.eps)
                action = np.random.choice(np.arange(
                    len(action_probs)),
                    p = action_probs)
                next_state, reward, done, _ = self.env.step(action)
                #next_state = tuple(next_state) ### for the enviroments with mutable state representation
                #print("step",self.step_cnt)
                self.step_cnt += 1
                print("step count is", self.step_cnt)
                sys.stdout.flush()
                if self.learner_policy in ["FCFS","LCFS"]:
                    self.temp_buffer.append([state,action,reward,next_state,self.step_cnt])
                else:
                    if t%window == 0:
                        self.temp_buffer.append([state,action,reward,next_state,self.step_cnt])

                self.stats.episode_rewards[i] += reward
                self.stats.episode_lengths[i] = t
                #time.sleep(.01)

                if done:
                    #if self.eps-1/self.num_episodes > 0:
                    #    self.eps -= 1/self.num_episodes
                    break
                state = next_state
        self.flag = False
        print('The total number of steps is ',self.step_cnt)

if __name__ == "__main__":
    #env = gym.make('WindyGridworld-v0')
    #env = gym.make('CartPole-v0')
    env = 'WindyGridworld-v0'
    Q = Qlearning(env,1,5000,.99,.1)
    print(Q.stats.episode_rewards[::10])
    print(Q.stats.episode_lengths[::10])
    myPlotting.plot_episode_stats(Q.stats)
