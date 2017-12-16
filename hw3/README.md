## HW 2 - Video Captioning

### Information

Code Deadline (GitHub): 12/16/2017 23:59

### Instructions

To do the testing with my best models:
```bash
python3 test.py --test_pg --test_dqn
```

To train policy gradient agent for Pong: (PG)
```bash
python3 main.py --train_pg
```

To train advantage actor critic policy gradient agent for Pong: (PG A2C)
```bash
python3 main.py --train_pg_a2c # currently not able to train with TA's sample code
```

To train double dueling deep-q-network agent for Breakout: (Dueling DDQN)
```bash
python3 main.py --train_dqn
```

### Used Packages

1. TensorFlow 1.3.0
2. Keras 2.0.7
3. Numpy 1.13.3

### About Models

#### Policy Gradient for Pong

```python
state_size = (80, 80, 1)
action_size = 3 # reduced from self.env.action_space.n=6
gamma = 0.99
learning_rate = 0.00005
optimizer = 'rmsprop'
```

#### Advantage Actor Critic Policy Gradient for Pong

```python
state_size = (80, 80, 1)
action_size = 3 # reduced from self.env.action_space.n=6
value_size = 1
gamma = 0.99
learning_rate = 0.0005
optimizer = 'adam'
```

#### Deep-Q-Network for Breakout

```python
state_size = (84, 84)
state_length = 4
model_input_shape = (84, 84, self.state_length)
action_size = self.env.action_space.n
memory_size = 10000
        
gamma = 0.99
epsilon_init = 1.0
epsilon_min = 0.05
exploration_steps = 1000000
epsilon_step = (self.epsilon_init - self.epsilon_min) / self.exploration_steps
learning_rate = 0.0001

initial_replay_size = 10000
replay_interval = 4
target_update_interval = 1000
save_interval = 50000

QN = 'dqn'
DUEL = 'avg' # 'none', 'avg', 'max', or 'naive'
optimizer = 'adam'
```
