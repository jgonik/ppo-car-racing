import argparse

import cv2
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from utils import DrawLine
from multi_car_racing import MultiCarRacing

parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--vis', action='store_true', help='use visdom')
parser.add_argument(
    '--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

NUM_AGENTS = 2
COUNTER = 0
writer = SummaryWriter()

transition = np.dtype([('s', np.float64, (args.img_stack, 96, 96)), ('a', np.float64, (3,)), ('a_logp', np.float64),
                       ('r', np.float64), ('s_', np.float64, (args.img_stack, 96, 96))])


class Env():
    """
    Environment wrapper for CarRacing 
    """
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 10
    buffer_capacity, batch_size = 2000, 128

    def __init__(self):
        self.env = MultiCarRacing(NUM_AGENTS)
        # self.env.seed(args.seed)
        self.net = Net().double().to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)
        self.training_step = 0
        self.reward_threshold = 800
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)

    def update(self):
        self.training_step += 1

        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(device)
        a = torch.tensor(self.buffer['a'], dtype=torch.double).to(device)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(device)
        
        print('s shape:', s.shape)

        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(device).view(-1, 1)

        with torch.no_grad():
            target_v = r + args.gamma * self.net(s_)[1]
            adv = target_v - self.net(s)[1]
            # adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):
                alpha, beta = self.net(s[index])[0]
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.net(s[index])[1], target_v[index])
                loss = action_loss + 2. * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def store(self, transition):
        self.buffer[COUNTER] = transition
        COUNTER += 1
        if COUNTER == self.buffer_capacity:
            COUNTER = 0
            return True
        else:
            return False

    def save_param(self):
        torch.save(self.net.state_dict(), 'param/multi_ppo_net_params.pkl')

    def select_action(self, state):
        state = torch.from_numpy(state).double().to(device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp

    def reset(self):
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * args.img_stack  # four frames for decision
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        for i in range(args.action_repeat):
            img_rgb, reward, die, _ = self.env.step(action)
            # don't penalize "die state"
            if die:
                reward += 100
            # green penalty
            for i in range(NUM_AGENTS):
                if np.mean(img_rgb[i, :, :, 1]) > 185.0:
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

    def render(self, arg):
        return self.env.render(arg)

    @staticmethod
    def rgb2gray(rgb, norm=True):
        # rgb image -> gray [0, 1]
        images = []
        for index in range(rgb.shape[0]):
            image = rgb[index,...]
            gray = np.dot(image[..., :], [0.299, 0.587, 0.114])
            if norm:
                # normalize
                gray = gray / 128. - 1.
            images.append(gray)
        return np.vstack(images)

    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
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
    Agent for training
    """

    def __init__(self, env):
        self.env = env


if __name__ == "__main__":
    NUM_EPISODES = 3000
    LOG_INTERVAL = 200
    env = Env()

    # state = env.reset()
    # video_file = 'output_mac_multi.avi'
    # video_obs = env.render('rgb_array')
    # video_h = 2 * video_obs.shape[1]
    # video_w = video_obs.shape[2]

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # vid_writer = cv2.VideoWriter(video_file, fourcc, 24, (video_w, video_h))

    training_records = []
    running_score = np.zeros((NUM_AGENTS,))
    state = env.reset()
    for i_ep in range(NUM_EPISODES):
        score = np.zeros((NUM_AGENTS,))
        state = env.reset()
        state = np.reshape(state, (NUM_AGENTS, 4, 96, 96))
        
        for agent_id in range(NUM_AGENTS):
            for t in range(1000):
                actions = []
                a_logps = []
                for i in range(NUM_AGENTS):
                    agent_state = state[i,...]
                    action, a_logp = env.select_action(agent_state)
                    actions.append(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
                    a_logps.append(a_logp)
                state_, reward, done, die = env.step(np.vstack(actions))
                state_ = np.reshape(state_, (NUM_AGENTS, 4, 96, 96))
                if args.render:
                    env.render()
                if env.store((state[agent_id,...], actions[agent_id], a_logps[agent_id], reward[agent_id], state_[agent_id,...])):
                    print('updating agent ' + agent_id)
                    env.update()
                score += reward

                # video_obs = env.render('rgb_array')
                # video_obs = np.concatenate(video_obs)
                # video_caption = ["agent_" + str(i) for i in range(NUM_AGENTS)]

                # for line_idx, line in enumerate(video_caption):
                #     video_obs = cv2.putText(video_obs, line,
                #     (10, int(line_idx * video_h / 2 + 40)),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7, (0, 0, 255), 3)
                # vid_writer.write(video_obs)

                state = state_
                if done or die:
                    break
        running_score = running_score * 0.99 + score * 0.01
        for i in range(NUM_AGENTS):
            writer.add_scalar("running_score_" + str(i), running_score[i], i_ep)
            writer.add_scalar("last_score_" + str(i), score[i], i_ep)

        if i_ep % LOG_INTERVAL == 0:
            print('Ep {}\tLast score: {}\tMoving average score: {}'.format(i_ep, score, running_score))
            env.save_param()
        if all(agent_score > env.reward_threshold for agent_score in running_score):
            print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
            break

    vid_writer.release()
