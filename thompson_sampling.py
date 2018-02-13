# Thompson Sampling

# Data set represends 10 different ads and 10000 different ad rotations
# 1 = ad was sucsessful
# 0 = ad was NOT sucsessful
# Goal: To find best ad to run

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

def thompsonSampling(dataset):

    # N: Number of ads ran, d: number of different ads
    N = dataset.shape[0] # 10000
    d = dataset.shape[1] # 10
    
    # Array of what add was selected in a particular rotation
    ads_selected = []
    
    # Number of times an ad got a reward of 1 up to round n
    numbers_of_rewards_1 = [0] * d
    
    # Number of times an ad got a reward of 0 up to round n
    numbers_of_rewards_0 = [0] * d
    
    # Total of rewards obtained
    total_reward = 0
    
    # array of reward values at each rotation
    rewards = np.zeros(N)
    
    # Running average of last 100 rewards
    running_avg_rewards = np.zeros(N)
    
    # Array for ploting number of times each ad was ran for a given rotation
    bandit_selection = np.zeros((d, N))
    
    # Preform Thompson Sampling
    for n in range(0, N):
        ad = 0
        max_random = 0
        
        
        for i in range(0, d):
            # Random draws taken form the beta distrabution
            random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
            # We will pick the ad with the maximum random draw
            if random_beta > max_random:
                max_random = random_beta
                ad = i
        ads_selected.append(ad)
        
        # Update the reward trackers
        reward = dataset.values[n, ad]
        if reward == 1:
            numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
        else:
            numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
        
        # Update the reward given    
        total_reward = total_reward + reward
        rewards[n] = reward
        
        # Update the Running average
        if n > 100:
            running_avg_rewards[n] = rewards[np.arange(n-499,n+1)].mean()
        
        # Update the add selection for each ad
        if n > 0:
            for i in range(0, d):
                bandit_selection[i][n] = bandit_selection[i][n-1]
            bandit_selection[ad][n] = bandit_selection[ad][n-1] + 1
            
    print('Thompson Sampling - Average total reward:', total_reward/N)
        
    return ads_selected, running_avg_rewards, bandit_selection

if __name__ == '__main__':
    
    # Importing the dataset
    dataset = pd.read_csv('../DataSets/Ads_CTR_Optimisation.csv')
    
    ads_selected, running_avg_rewards, bandit_selection = thompsonSampling(dataset)
    
    # Histogram of amount of times and ad was selected
    plt.hist(ads_selected)
    plt.title('Histogram of ads selections')
    plt.xlabel('Ads')
    plt.ylabel('Number of times each ad was selected')
    plt.show()
    
    # Running average of rewards
    plt.plot(running_avg_rewards)
    plt.title('Running Average Reward (last 500)')
    plt.xlabel('Ads')
    plt.ylabel('rewards')
    plt.show()
    
    # Number of times each ad was ran 
    for i in range(0,dataset.shape[1]):
        plt.plot(bandit_selection[i], label='Ad %d' %(i+1))
    plt.title('Number of times ad has been selected to run.')
    plt.xlabel('Number of time distributed')
    plt.ylabel('Number of time ad ran')
    plt.legend()
    plt.show()

