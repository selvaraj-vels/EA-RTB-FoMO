# File: models/rl_bid_shading.py
# Purpose: Reinforcement Learning for bid shading in EA-RTB

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# -----------------------
# DQN Network
# -----------------------
class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=7):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # 7 possible shading ratios
        )
    def forward(self, x):
        return self.fc(x)

# -----------------------
# Replay Memory
# -----------------------
class ReplayMemory:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = []
    
    def push(self, transition):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# -----------------------
# Environment Step Function
# -----------------------
def step(state, action_idx, bid_base, fomo_score, p_win, ctr, floor_price):
    shading_ratios = [-0.2,-0.15,-0.1,-0.05,0,0.05,0.1]
    delta = shading_ratios[action_idx]
    bid = bid_base*(1+delta)
    
    # Determine cost
    cost = max(floor_price, bid)  # simplified for first-price auction
    
    # Reward function (CTR - cost + expected surplus)
    gamma = 1.0  # CTR weight
    lambda_s = 0.5  # surplus weight
    expected_surplus = p_win * (ctr - bid)
    
    reward = gamma*ctr - cost + lambda_s*expected_surplus
    next_state = np.array([floor_price, p_win, fomo_score, ctr], dtype=np.float32)
    done = False  # for single-step simplification
    return next_state, reward, done

# -----------------------
# Training Loop (simplified)
# -----------------------
if __name__ == "__main__":
    input_dim = 4  # floor price, p_win, FoMO, recent CTR
    output_dim = 7  # 7 shading actions
    dqn = DQN(input_dim, output_dim=output_dim)
    optimizer = optim.Adam(dqn.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    memory = ReplayMemory()
    
    # Dummy example for testing
    for episode in range(10):
        state = np.array([1.0, 0.5, 0.7, 0.03], dtype=np.float32)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_values = dqn(state_tensor)
        action_idx = torch.argmax(action_values).item()
        
        next_state, reward, done = step(state, action_idx, bid_base=1.0, fomo_score=0.7, 
                                        p_win=0.5, ctr=0.03, floor_price=1.0)
        
        # Store in memory
        memory.push((state, action_idx, reward, next_state, done))
        
        # Sample and train (batch size = 1 for demo)
        batch = memory.sample(1)
        for s, a, r, ns, d in batch:
            s_tensor = torch.tensor(s, dtype=torch.float32)
            ns_tensor = torch.tensor(ns, dtype=torch.float32)
            q_values = dqn(s_tensor)
            q_target = q_values.clone().detach()
            q_target[a] = r + 0.99*torch.max(dqn(ns_tensor))
            loss = criterion(q_values, q_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Episode {episode} - Action {action_idx}, Reward {reward}")
