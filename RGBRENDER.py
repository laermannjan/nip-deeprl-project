import gym
import numpy as np
import matplotlib.pyplot as plt
env = gym.make('CartPole-v0')
env.reset()
'''
for _ in range(10):
    env.render()
    print('step',env.render('rgb_array').shape)
    env.step(env.action_space.sample()) # take a random action
'''
obs1 = env.render('rgb_array')
env.step(env.action_space.sample()) # take a random action
obs2 = env.render('rgb_array')
env.step(env.action_space.sample()) # take a random action
obs3 = env.render('rgb_array')
env.step(env.action_space.sample()) # take a random action
obs4 = env.render('rgb_array')

print ('OBSERVATION SHAPEEEEEEE    ', obs4.shape)

train_arr = np.zeros((800,1200,3))*1. ### 3 for rgb..think about grayscale [:,:,0]

print ('TRAIN ARRAY SHAPEEEEE    ',train_arr.shape )
train_arr[:obs1.shape[0],:obs1.shape[1],:] = obs1*1.
train_arr[:obs1.shape[0],obs1.shape[1]:,:] = obs2*1.
train_arr[obs1.shape[0]:,:obs1.shape[1],:] = obs3*1.
train_arr[obs1.shape[0]:,obs1.shape[1]:,:] = obs4*1.

train_obs = train_arr

plt.figure(:
plt.imshow(-train_obs[:,:,:])
plt.show()