# Comparing different reinforcement learning algorithms
# EXPLORE vs EXPLOIT

# Algorithms:
#   - Random Selection
#   - Epsilon Greedy
#   - Upper Confidence Bound (UBC)
#   - Tompson Sampling


# About the dataset:
#   Data set represends 10 different ads and 10000 different ad rotations
#   1 = ad was sucsessful
#   0 = ad was NOT sucsessful
#   Goal: To find best ad to run

import matplotlib.pyplot as plt
import pandas as pd

from random_selection import randomChoice
from epsilon_greedy import epsilon_greedy
from upper_confidence_bound import ubc
from thompson_sampling import thompsonSampling

# Importing the dataset
dataset = pd.read_csv('../DataSets/Ads_CTR_Optimisation.csv')

_, rand_rar, _ = randomChoice(dataset)
_, eps_rar, _ = epsilon_greedy(dataset)
_, ubc_rar, _ = ubc(dataset)
_, ts_rar, _ = thompsonSampling(dataset)

plt.plot(rand_rar, label = 'Random Selection')
plt.plot(eps_rar, label = 'Epsilon Greedy')
plt.plot(ubc_rar, label = 'Upper Confidence Bound')
plt.plot(ts_rar, label = 'Tompson Sampling')
plt.title('Running Average Reward (last 500)')
plt.xlabel('Ad rotations')
plt.ylabel('rewards')
plt.legend()
plt.show()

