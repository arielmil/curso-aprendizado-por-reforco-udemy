# Importação das bibliotecas

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

# Criação da arquitetura da rede neural
class Network(nn.Module):
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        
        # 5 -> 30 -> 3 - full connection (dense)
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values
    
# Implementação do replay de experiência
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    # D,E,F,G,H
    # 4 valores: último estado, novo estado, última ação, última recompensa
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
            
    def sample(self, batch_size):
        # ((1,2,3), (4,5,6)) -> ((1,4), (2,5), (3,6))
        batch = random.sample(self.memory, batch_size)
        batch_state, batch_next_state, batch_action, batch_reward = zip(*batch)
        batch_state = torch.cat(batch_state, 0)
        batch_next_state = torch.cat(batch_next_state, 0)
        batch_action = torch.cat(batch_action, 0)
        batch_reward = torch.cat(batch_reward, 0)
        return batch_state, batch_next_state, batch_action, batch_reward
        
# Implementação de Deep Q-Learning
class Dqn():
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.zeros(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        
    def select_action(self, state):
        # softmax(1,2,3) -> (0.04, 0.11, 0.85) -> (0, 0.02, 0.98)
        with torch.no_grad():
            logits = self.model(state) * 100  # T = 7
            probs = F.softmax(logits, dim=1)
        action = probs.multinomial(num_samples=1)
        return action.item()
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()
        
    def update(self, reward, new_signal):
        new_state = torch.tensor(new_signal, dtype=torch.float32).unsqueeze(0)
        self.memory.push((
            self.last_state,
            new_state,
            torch.LongTensor([int(self.last_action)]),
            torch.tensor([self.last_reward], dtype=torch.float32),
        ))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1.)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('Carregado com sucesso')
        else:
            print('Erro ao carregar')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            
        
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        