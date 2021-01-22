import argparse

import numpy as np
import cv2

import gym
import torch
import torch.nn as nn

from multi_car_racing_test import MultiCarRacing

parser = argparse.ArgumentParser(description='Test the PPO agent for the CarRacing-v0')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 12)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

NUM_AGENTS = 2

class Env():
    """
    Test environment wrapper for CarRacing 
    """

    def __init__(self):
        self.env = MultiCarRacing(NUM_AGENTS)
        self.env = gym.wrappers.Monitor(self.env, 'recordings_separate', force=True)
        self.env.seed(args.seed)
        self.reward_threshold = 900

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * args.img_stack
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        for i in range(args.action_repeat):
            img_rgb, reward, die, _ = self.env.step(action)
            # don't penalize "die state"
            if die:
                reward += 100
            # green penalty
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == args.img_stack
        return np.array(self.stack), total_reward, done, die

    def render(self, *arg):
        return self.env.render(*arg)

    @staticmethod
    def rgb2gray(rgb, norm=True):
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray

    @staticmethod
    def reward_memory():
        count = 0
        length = 100
        history = np.zeros((length, NUM_AGENTS))

        def memory(reward):
            nonlocal count
            history[count,:] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory


class Net(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self):
        super(Net, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(args.img_stack, 8, kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
        )  # output shape (256, 1, 1)
        self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return (alpha, beta), v


class Agent():
    """
    Agent for testing
    """

    def __init__(self):
        self.net = Net().float().to(device)

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        action = alpha / (alpha + beta)

        action = action.squeeze().cpu().numpy()
        return action

    def load_param(self, name):
        self.net.load_state_dict(torch.load('param/ppo_net_params_' + name + '.pkl'))


if __name__ == "__main__":
    agent1 = Agent()
    agent2 = Agent()
    agent1.load_param('agent1')
    agent2.load_param('agent2')
    env = Env()

    # state = env.reset()
    # video_file = 'output_test_single.avi'
    # video_obs = env.render('rgb_array')
    # video_h = 2 * video_obs.shape[1]
    # video_w = video_obs.shape[2]

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # vid_writer = cv2.VideoWriter(video_file, fourcc, 24, (video_w, video_h))

    training_records = []
    running_score = 0
    # state = env.reset()
    for i_ep in range(1):
        score = 0
        state = env.reset()
        agent1_state = state[:,0,...]
        agent2_state = state[:,1,...]

        for t in range(30000):
            action1 = agent1.select_action(agent1_state)
            action2 = agent2.select_action(agent2_state)
            action1 = action1 * np.array([2., 1., 1.]) + np.array([-1., 0., 0.])
            action2 = action2 * np.array([2., 1., 1.]) + np.array([-1., 0., 0.])
            action = np.vstack((action1, action2))
            state_, reward, done, die = env.step(action)
            if args.render:
                env.render()
            score += reward

            # video_obs = env.render('rgb_array')
            # video_obs = np.concatenate(video_obs)
            # video_caption = ["single_agent"]

            # for line_idx, line in enumerate(video_caption):
            #     video_obs = cv2.putText(video_obs, line,
            #     (10, int(line_idx * video_h / 2 + 40)),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.7, (0, 0, 255), 3)
            # vid_writer.write(video_obs)

            state = state_
            if done or die:
                break

        print('Ep {}\tScore: {}\t'.format(i_ep, score))

    # vid_writer.release()

