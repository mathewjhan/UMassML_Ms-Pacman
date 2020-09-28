import gym
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import random
from collections import deque
from PIL import Image
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.util import crop
import numpy as np

IMAGE_RESIZE = (86, 80)
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.99995
EPISODES = 500
BATCH_SIZE = 32
GAMMA = 0.9

# i have no idea wat im doing lol

# Flatten 2D output
class Lin_View(nn.Module):
        def __init__(self):
                super(Lin_View, self).__init__()
        def forward(self, x):
                return x.view(x.size()[0], -1)

class DQN_Agent:
    def __init__(self):
        self.action_space = 9
        self.observation_space = IMAGE_RESIZE
        self.exploration_rate = EXPLORATION_MAX
        self.memory = deque(maxlen=500000)
        self.lr = 10e-3

        # slightly modified deepmind architecture
        self.model = nn.Sequential(
                nn.Conv2d(1,32,8,4),
                nn.LeakyReLU(),
                nn.Conv2d(32,64,4,2),
                nn.LeakyReLU(),
                nn.Conv2d(64,64,3,1),
                nn.LeakyReLU(),
                Lin_View(),
                nn.Linear(64*7*6,9)
            )

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)

        if(torch.cuda.is_available()):
            self.model.cuda()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Epsilon greedy
    def act(self, state):
        if(np.random.rand() < self.exploration_rate):
            return random.randrange(self.action_space)

        state = torch.from_numpy(state).reshape(1, 1, *(IMAGE_RESIZE)).cuda().float().detach()
        q_values = self.model(state)
        values, indices = torch.max(q_values, 1)
        return indices.item()

    # Experience replay
    def replay(self):

        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)

        for state, action, reward, state_next, terminal in batch:

            s = torch.from_numpy(state).reshape(1, 1, *(IMAGE_RESIZE)).cuda().float().detach()
            sn = torch.from_numpy(state_next).reshape(1, 1, *(IMAGE_RESIZE)).cuda().float().detach()

            # Penalize no reward a little bit to encourage
            # more movement (not heavily tested)
            temp = reward
            if(temp == 0):
                temp -= 1
            q_update = temp

            if(not terminal):
                q_update = temp + GAMMA * torch.max(self.model(sn), 1)[0].item()

            outputs = self.model(s)

            target = torch.zeros(1,9).cuda()
            target[:,action] = q_update

            self.optimizer.zero_grad()
            loss = self.loss_function(outputs, target)
            loss.backward()
            self.optimizer.step()

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

# Create average image of past 4 frames
# track motion of pacman/ghosts
def buffered_image(deque):
    first = np.zeros(IMAGE_RESIZE)
    for im in deque:
        first += im
    return ((1/len(deque)) * first).astype(np.uint8)

def run():
    env = gym.make('MsPacman-v0')

    # use pygame to play game
    #play(env)

    # Keep track of rewards
    rew_list = list()
    rAll = 0

    # Keep a buffer for previous 4 frames
    # One buffer for current state and
    # another for the next state
    dqn = DQN_Agent()
    buff_cur = deque(maxlen=4)
    buff_next = deque(maxlen=4)

    # Training
    for i in range(EPISODES):

        s = env.reset()
        d = False

        # Rescale image to 84x80 and convert to grayscale
        # cuts out bottom part of the image with score
        s = (rgb2gray(resize(s[0:172, 0:160], IMAGE_RESIZE, anti_aliasing=True, order=0))*255).astype(np.uint8)

        t = 0
        next_time = 0

        buff_next.append(s)

        while(d != True):
            env.render()

            buff_cur.append(s)

            if(len(buff_cur) != 4):
                a = env.action_space.sample()
            else:
                buff_s = buffered_image(buff_cur)
                a = dqn.act(buff_s)

                # Debugging
                # img = Image.fromarray(buff_s)
                # img.save('test.png', 'PNG')

            s1,r,d,_ = env.step(a)

            s1 = (rgb2gray(resize(s1[0:172, 0:160], IMAGE_RESIZE, anti_aliasing=True, order=0))*255).astype(np.uint8)
            buff_next.append(s1)

            if(len(buff_next) != 4 or len(buff_cur) != 4):
                dqn.remember(s, a, r, s1, d)
            else:
                buff_s1 = buffered_image(s1)
                dqn.remember(buff_s, a, r, buff_s1, d)


            # Every 2-4 frames experience replay to speed up training
            # instead of every single frame
            if(t == next_time):
                t = 0
                next_time = random.randint(2,4)
                dqn.replay()
            else:
                t += 1

            rAll += r
            s = s1

        rew_list.append(rAll/(i+1))
        env.render()

        print("Episode: ", i)
        print("Reward sum: ", rAll/(i+1))

    print("Final reward sum: ", rAll/EPISODES)
    plt.plot(rew_list)
    plt.show()

if(__name__ == '__main__'):
    run()
