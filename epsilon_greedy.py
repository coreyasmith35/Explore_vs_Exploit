# Epsilon Greedy

# Data set represends 10 different ads and 10000 different ad rotations
# 1 = ad was sucsessful
# 0 = ad was NOT sucsessful
# Goal: To find best ad to run

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Implementing Random Selection
def epsilon_greedy(dataset, eps = .05):

    # N: Number of ads ran, d: number of different ads
    N = dataset.shape[0] # 10000
    d = dataset.shape[1] # 10
    
    # Array of what ad was selected in a particular rotation
    ads_selected = []
    
    num_ad_selected = np.zeros(d)
    num_ad_win = np.zeros(d)
    
    # Total of rewards obtained
    total_reward = 0
    
    # array of reward values at each rotation
    rewards = np.zeros(N)
    
    avg_rewards = np.zeros(d)
    
    # Running average of last 100 rewards
    running_avg_rewards = np.zeros(N)
    
    # Array for ploting number of times each ad was ran for a given rotation
    bandit_selection = np.zeros((d, N))
    
    # Preform Epsilon Greedy
    for n in range(0, N):

        if avg_rewards.sum() == 0:
            ad = np.random.choice(d)
        elif np.random.random() < eps:
            ad = np.random.choice(d)
        else:
            ad = np.argmax(avg_rewards)
             
        ads_selected.append(ad)
        reward = dataset.values[n, ad]
        
        num_ad_selected[ad] = num_ad_selected[ad] + 1
        num_ad_win[ad] = num_ad_win[ad] + reward
        avg_rewards[ad] = num_ad_win[ad] / num_ad_selected[ad]
        total_reward = total_reward + reward
        
        # Update the Running average
        rewards[n] = reward
        running_avg_rewards[n] = rewards[np.arange(n-499,n+1)].mean()
        
        if n > 0:
            for i in range(0, d):
                bandit_selection[i][n] = bandit_selection[i][n-1]
            bandit_selection[ad][n] = bandit_selection[ad][n-1] + 1
        
    print('Epsilon Greedy(',eps,') - Average total reward:', total_reward/N)
    
    return ads_selected, running_avg_rewards, bandit_selection


if __name__ == '__main__':
    
    # Importing the dataset
    dataset = pd.read_csv('../DataSets/Ads_CTR_Optimisation.csv')
    
    ads_selected, running_avg_rewards, bandit_selection = epsilon_greedy(dataset)
    
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
    

    _, eps25, _ = epsilon_greedy(dataset,eps=.25)
    _, eps50, _ = epsilon_greedy(dataset,eps=.5)
    
    # Comparing Epsilon Values
    plt.plot(running_avg_rewards, label = 'eps 0.05')
    plt.plot(eps25, label = 'eps 0.25')
    plt.plot(eps50, label = 'eps 0.50')
    plt.title('Comparing Epsilon Values, running avg')
    plt.xlabel('Ads')
    plt.ylabel('rewards(avg last 500')
    plt.legend()
    plt.show()
