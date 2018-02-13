# Explore vs Exploit Problem

### Algorithms:
- Random Selection
- Epsilon Greedy
- Upper Confidence Bound (UBC)
- Tompson Sampling

### About the data:
The data represents 10 different adds run over 10000 different times. 

|Ad 1|Ad 2|Ad 3|Ad 4|Ad 5|Ad 6|Ad 7|Ad 8|Ad 9|Ad 10|
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|1|0|0|0|1|0|0|0|1|0|
|0|0|0|0|0|0|0|0|1|0|
|0|0|0|0|0|0|0|0|0|0|

Each row represents a different ad rotion, we are to select one ad to run during each rotation. Depending on the ad we choose we will receive:

    - 1 represents a successful ad
    - 0 represents an unsuccessful ad	

### Goal:
Find the most successful ad while maximizing our reward.

Random Selection - Average total reward: 0.1248

Epsilon Greedy( 0.05 ) - Average total reward: 0.2589
Epsilon Greedy( 0.25 ) - Average total reward: 0.2189
Epsilon Greedy( 0.5 ) - Average total reward: 0.1937

Thompson Sampling - Average total reward: 0.2585

Upper Confidence Bound - Average total reward: 0.2178