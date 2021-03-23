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


def epsilon_greedy(Q,num_actions):
    
    def policy(state,eps):
        action_probs = [eps/(num_actions) for _ in range(num_actions)]
        try:
            greedy_action = Q[state].index(max(Q[state]))
        except:
            print(max(Q[state]))
            print(Q[state])
            greedy_action = 0
        action_probs[greedy_action] += 1-eps
        return action_probs
    return policy
class Sarsa():
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
        actor_thread = threading.Thread(target=self.qLearning)
        learner_thread = threading.Thread(target=self.learner)
        actor_thread.start()
        learner_thread.start()
        
        actor_thread.join()
        learner_thread.join()

    def learner(self):
        
        update_cnt = 0
        policy = epsilon_greedy(self.Q,self.env.action_space.n)
        
        while self.flag:
            
            if self.temp_buffer:
                
                if self.learner_policy == "LCFS":
                    exp = self.temp_buffer.pop(-1)
                else:
                    exp = self.temp_buffer.pop(0)
                
                next_action = exp[4]
                td_target = exp[2] + self.gamma*self.Q[exp[3]][next_action]                                                                                                                           
                td_delta = td_target - self.Q[exp[0]][exp[1]]                                                                                                                                                    
                self.Q[exp[0]][exp[1]] = (1-self.alpha)*self.Q[exp[0]][exp[1]] + self.alpha*td_target  
                
                update_cnt += 1
           
        print('The total number of updates is ',update_cnt)
        

    def qLearning(self):
        
        policy = epsilon_greedy(self.Q,self.env.action_space.n)
        print("I am actor")
        step_cnt = 0
        
        if self.learner_policy not in ["FCFS","LCFS"]:
            window = int(self.learner_policy.split(' ')[3])
            
        for i in range(self.num_episodes):
           
            state = self.env.reset()
            action_probs = policy(state,self.eps)
            action = np.random.choice(np.arange(
                    len(action_probs)),
                    p = action_probs)
            for t in itertools.count():
                
                next_state, reward, done, _ = self.env.step(action)
                step_cnt += 1
                
                action_probs = policy(next_state,self.eps)
                next_action = np.random.choice(np.arange(
                    len(action_probs)),
                    p = action_probs)
                
                if self.learner_policy in ["FCFS","LCFS"]:
                    self.temp_buffer.append([state,action,reward,next_state,next_action])
                else:
                    if t%window == 0:
                        self.temp_buffer.append([state,action,reward,next_state,next_action])

                self.stats.episode_rewards[i] += reward
                self.stats.episode_lengths[i] = t

                if done:
                    if self.eps-1/self.num_episodes > 0:
                        self.eps -= 1/self.num_episodes
                    break
                state = next_state
                action = next_action
                
        self.flag = False
        print('The total number of steps is ',step_cnt)

if __name__ == "__main__":

    env = 'WindyGridworld-v0'
    Q = Qlearning(env,1,5000,.99,.1)
    print(Q.stats.episode_rewards[::10])
    print(Q.stats.episode_lengths[::10])
    myPlotting.plot_episode_stats(Q.stats)
