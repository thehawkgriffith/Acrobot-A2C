# Acrobot-A2C
The Advantage Actor Critic algorithm implementation for the OpenAI Gym's Acrobot-v1 environment.


**My Hyperparameters:**

  Number of trajectories = 1

  Learning rate for policy parameters: 3e-4

  Gamma: 0.99

  Actor network architecture: (FC-layer(input_shape, 256), FC-layer(256, n_actions))
  
  Critic network architecture: (FC-layer(input_shape, 512), FC-layer(512, 1))

  Advantage function: **R(t) - V(s_t)**
  
  c (Weightage of the critic MSE error in the criterion): 0.01 
  
  Number of steps T per actor-critic update: 5



**The Rewards (Y-Axis) vs Episodes plot (X-Axis):**
![PLOT](/rewards.png)
