import gym
import numpy as np
from baselines import deepq

# playing the random agent for 10k episodes 5 times and saving a numpy array

envs = ["Acrobot-v1"]#,["LunarLander-v2"]

def main():
    for enviro in envs:    
        env = gym.make(enviro)
        act = deepq.load("ABbest_mean_rew.pkl")
        plot_array = []
        for run in range(5):
            save_arr = []
            for i in range(100):
                obs, done = env.reset(), False
                episode_rew = 0
                
                while not done:
                    #env.render()
                    obs, rew, done, _ = env.step(act(obs[None])[0])#env.step(env.action_space.sample())#
                    episode_rew += rew
                
                save_arr += [episode_rew]
                
                if i % 10 == 0:
                    print(i,":  Episode reward", episode_rew, '  run -- ',run)
                    #print (len(save_arr))
            save_arr = np.array(save_arr)
            plot_array += [save_arr]
        plot_array = np.array(plot_array)
        
        #print('save as ',"random{}.npy".format(enviro[:4]))
        np.save("/mnt/data/bestmean{}.npy".format(enviro[:4]),plot_array)
        print ('shape of saved array: ',plot_array.shape)
        '''
        plt.figure(figsize=(15,5))
        sns.tsplot(plot_array)
        plt.ylim(-600,-100)
        plt.show()
        '''
        
if __name__ == '__main__':
    main()
