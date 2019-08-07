import torch
import torch.nn as nn
import numpy as np
import gym
import matplotlib.pyplot as plt
from compytent.utils import Logger

logger = Logger('Acrobot-v1', '/home/shahbaz/Research/Experimental Records/A2C/')


class PolicyNet(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
            nn.Softmax()
        )

    def forward(self, x):
        x = torch.FloatTensor([x]).cuda()
        x = x.view(1, -1)
        return self.fc(x)


class Baseline(nn.Module):

    def __init__(self, input_shape):
        super(Baseline, self).__init__()
        shape = 1
        for dim in input_shape:
            shape *= dim
        self.fc = nn.Sequential(
            nn.Linear(shape, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = torch.FloatTensor([x]).cuda()
        x = x.view(1, -1)
        return self.fc(x)


class Agent:

    def __init__(self, policy_net, baseline_net):
        self.policy_net = policy_net
        self.baseline = baseline_net

    def train(self, env, num_traj, iterations, gamma, base_epochs):
        mapping = {0: -1, 1: 0, 2: 1}
        iter_rewards = []
        for iter in range(iterations):
            trajectories = []
            ITER_REW = 0
            for _ in range(num_traj):
                rewards = []
                log_probs = []
                states = []
                s = env.reset()
                done = False
                while not done:
                    a = self.policy_net(s)
                    states.append(s)
                    a2 = a.detach().cpu().numpy()
                    vec = [0, 1, 2]
                    u = np.random.choice(vec, 1, replace=False, p=a2[0])
                    log_probs.append(a[0][u])
                    del a
                    sp, r, done, _ = env.step(mapping[u[0]])
                    ITER_REW += r
                    rewards.append(r)
                    # env.render()
                    s = sp
                trajectories.append({'log_probs': log_probs, 'rewards': rewards, 'states': states})
            self.update_parameters(trajectories, gamma)
            logger.write('episode: %d     reward: %.3f' % (iter, ITER_REW/num_traj))
            print("ITERATION:", iter + 1, "AVG REWARD:", ITER_REW / num_traj)
            iter_rewards.append(ITER_REW/num_traj)
        return iter_rewards

    def update_parameters(self, trajectories, gamma):
        c = 0.01
        k = 5
        loss = torch.tensor([0]).float().cuda()
        optim = torch.optim.Adam(list(self.policy_net.parameters()) + list(self.baseline.parameters()), lr=3e-4)
        for trajectory in trajectories:
            for t in range(len(trajectory['rewards'])):
                r_t = torch.tensor([0]).float().cuda()
                log_prob = trajectory['log_probs'][t]
                temp = trajectory['rewards'][t:t + k]
                for i, reward in enumerate(temp[:-1]):
                    r_t += gamma ** i * reward
                critic_estimate = self.baseline(trajectory['states'][t])[0]
                r_t += gamma ** i * self.baseline(trajectory['states'][i+1])[0]
                advantage_t = r_t - critic_estimate
                loss += (-log_prob * advantage_t) + (c * (critic_estimate - r_t) ** 2)
                if t % k == 0:
                    optim.zero_grad()
                    loss.backward()
                    # print("\nPolicy Gradients\n")
                    # for name, param in self.policy_net.named_parameters():
                    #     print(name, param.grad.data.sum())
                    # print("\nCritic Gradients\n")
                    # for name, param in self.baseline.named_parameters():
                    #     print(name, param.grad.data.sum())
                    optim.step()
                    loss = torch.tensor([0]).float().cuda()


def main():
    env = gym.make('Acrobot-v1')
    policy_net = PolicyNet(env.observation_space.shape, env.action_space.n).to(torch.device('cuda'))
    base_net = Baseline(env.observation_space.shape).to(torch.device('cuda'))
    # policy_net.load_state_dict(torch.load('./policynet'))
    # base_net.load_state_dict(torch.load('./basenet'))
    agent = Agent(policy_net, base_net)
    rews = agent.train(env, 1, 100, 0.99, 5)
    plt.plot(rews)
    plt.show()
    logger.save_plot('/home/shahbaz/Research/Experimental Records/A2C/', 'Acrobot-v1', rews)


main()