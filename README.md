# OpenAI Gym Ms-Pacman DQN
My submission for the UMass ACM ML Ms-Pacman contest. Won 1st Place.

## DQN Architecture
```
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
```
