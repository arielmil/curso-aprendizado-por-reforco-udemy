#Neste arquivo definirei a rede neural conforme o curso fez, porém no próximo (ai_me.py), irei implementar a minha própria maneira.

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

#Criação da arquitetura da rede neural, terá 5 entradas, e 3 saidas.
class Network(nn.Module):

    #input_size: Quantos neuronios de entrada temos, nb_action: Quantas ações temos (Número de neuronios de saída)
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action

        #Terá uma camada oculta com 30 neuronios
        # Um neuronio de uma camada, contecta-se com todos os neuronios da próxima camada (Full connected)

        #Faz a ligação entre a camada de entrada e os 30 neuronios da camada oculta
        self.fc1 = nn.Linear(input_size, 30)
        #Faz a ligação entre a camada de oculta e os nb_action neuronios da camada de saida
        self.fc2 = nn.Linear(30, nb_action)
         
    def forward(self, state):
        # Passa state para a primeira camada como input, e aplica relu nela.
        x = F.relu(self.fc1(state))
        # Passa x que foi calculado na linha acima para a camada oculta
        # Posteriormente será aplicado softmax no resultado de forward.
        q_values = self.fc2(x)
        return q_values
    
    # Implementação do replay de experiencia:
    # A classe deve implementar a lógica de criar uma amostra de n eventos (Serão 100000)
    # Um evento é definido como um conjunto de 4 valores:
        # ultimo estado, novo estado, ultima ação, ultima recompensa
        # aonde a ultima ação leva a mudança do ultimo estado para o novo estado, e gera a ultima recompensa.
    # Herda de object, classe base no python.
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)

        #Se a memoria estiver cheia, deleta o 1o estado
        if len(self.memory) >= self.capacity:
            del self.memory[0]

    # Retorna uma amostra de eventos de tamanho batch_size
    def sample(self, batch_size):

        # Pega aleatoriamente batch_size eventos da memoria.
        # o asterisco antes de random faz com que os valores retornados por random (uma especie de lista), sejam tirados da lista,
        # e passados para zip como argumentos separados (serão tuplas)
        # zip então pega o enesimo elemento de cada tupla, e os une em uma única tupla
        # por exemplo: random.sample retornara [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        # o asterisco fara: (1, 2, 3), (4, 5, 6), (7, 8, 9)
        # zip fara um iterador contendo: (1, 4, 7), (2, 5, 8), (3, 6, 9) 
        samples = zip(*random.sample(self.memory, batch_size))

        # lambda: cria uma funcao anonima que recebe x.
        # map aplica a funcao anonima a cada item do iteravel samples
        # torch.cat concatena uma sequencia de tensores ao longo do eixo 0.

        # Exemplo simples:
        # random.sample(...) retorna: [(s1, a1, r1), (s2, a2, r2), (s3, a3, r3)] 
        # zip(random.sample(...)) retorna: [(s1, s2, s3), (a1, a2, a3), (r1, r2, r3)]
        # Cada item de samples é uma tupla do mesmo tipo de dado,
        # A linha faz:
        # para cada tupla (s1, s2, s3), concatena todos eles em um tensor batch
        # para cada tupla (a1, a2, a3), concatena todos eles em outro tensor batch
        # para cada tupla (r1, r2, r3), concatena todos eles em outro tensor batch
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
        

'''
para testar sample:

rm = ReplayMemory(5)

state1 = 1
state2 = 2
state3 = 3
state4 = 4

prev_states = [state1, state2, state3, state4]

state5 = 5
state6 = 6
state7 = 7
state8 = 8

next_states = [state5, state6, state7, state8]

action1 = 9
action2 = 10
action3 = 11
action4 = 12

actions = [action1, action2, action3, action4]

reward1 = 13
reward2 = 14
reward3 = 15
reward4 = 16

rewards = [reward1, reward2, reward3, reward4]

for value in range(4):
    prev_state = torch.tensor(prev_states[value], dtype=torch.float).unsqueeze(0)
    next_state = torch.tensor(next_states[value], dtype=torch.float).unsqueeze(0)
    action = torch.tensor(actions[value], dtype=torch.float).unsqueeze(0)
    reward = torch.tensor(rewards[value], dtype=torch.float).unsqueeze(0)

    rm.push((prev_state, next_state, action, reward))

iterator = rm.sample(3)

for item in iterator:
    print(item)

#retorno:

#tensor([3., 4., 1.])
#tensor([7., 8., 5.])
#tensor([11., 12.,  9.])
#tensor([15., 16., 13.])'''

# Implementação do Deep-Q-Learning:
class Dqn():
    def __init__(self, input_size, nb_action, gamma, lr=0.001):
        self.gamma = gamma
        #Learning rate
        self.lr = lr
        #Uma amostra de n recompensas associados aos ultimos n estados que o agente esteve
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = lr)

        # torch.Tensor(input_size) cria um tensor 1D com input_size elementos (valores nao inicializados).
        # unsqueeze(0) adiciona uma dimensao no inicio, transformando de shape (input_size,) para (1, input_size).

        #Logica:
            # last_state guarda o ultimo estado observado pelo agente.
            # unsqueeze(0) deixa o tensor com formato de batch (1 amostra), que e o formato esperado pela rede.
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state, temperature = 7):
        # volatile = True é para não fazer os calculos de gradiente, já que nessa parte não estamos em treinamento.
        # self.model chama a função forward de nn.Module que foi sobreescrita na classe Network.
        # em forward, state é aplicado a primeira camada, e retorna a operação feita na ultima camada
        # em seguida, a funcao softmax é aplicada para converter a saida da rede neural em um array de probabilidades que somam 1
        # temperatura mexe na distribuição probabilistica de softmax
        # quanto menor a temperatura, mais exploração pois as probabilidades ficam mais parecidas
        # quanto maior a temperatura, maior a tendencia da maior ação dominar
        # 7 é considerado um valor "alto", então a escolha de estados fica mais gananciosa
        # fazendo com que o maior q_value seja escolhido na maioria das vezes
        # na pratica esse temperature se torna 1/temperature, ou seja,
        # quanto maior temperature, menor fica 1/temperature,
        # e mais ganancioso fica a escolha de q_value.
        probs = F.softmax(self.model(Variable(state, volatile= True)) * temperature)

        # probs.multinomial pega aleatóriamente um valor retornado pelo softmax acima
        # quanto maior um determinado valor, maior as chances dele ser escolhido para ser retornado
        action = probs.multinomial()
        return action.data[0, 0]
    
    # Parametros retornados pela função sample
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action, debug=False):
        # self.model(batch_state): A rede recebe um batch de estados e retorna os Q-values para todas as ações.
            # Exemplo: batch_state shape: [100, 5] (100 estados, 5 features cada)
            # Saída shape: [100, 3] (100 amostras, 3 Q-values - um para cada ação possível)

        # batch_action.unsqueeze(1): Adiciona uma dimensão ao tensor de ações para o formato que .gather() espera.
        # batch_action: tensor 1D com shape [batch_size] 
            # Exemplo: [100] → [0, 2, 1, 0, 2, 1, 0, ...] (índices de cada Q-value)
        # saida de batch_action.unsqueeze(1): Tensor 2D com shape [batch_size, 1]
            # Exemplo: [100, 1] → [[0], [2], [1], [0], [2], [1], [0], ...]
        # unsqueeze(1) adiciona uma nova dimensão na posição 1 Transforma de [100] para [100, 1]

        # gather(1, batch_action.unsqueeze(1)): Seleciona elementos específicos ao longo de uma dimensão usando índices.
        
        outputs = self.model(batch_state)
        if (debug):
            print("\noutputs antes de .gather: " + str(outputs))
        outputs = outputs.gather(1, batch_action.unsqueeze(1))
        if (debug):
            print("\noutputs depois de .gather e antes de .squeeze(1): " + str(outputs))
        outputs = outputs.squeeze(1)
        if (debug):
            print("\nbatch_action sem unsqueeze(1): " + str(batch_action))
        
        next_outputs = self.model(batch_next_state)
        if (debug):
            print("\nnext_outputs antes de .detach(): " + str(next_outputs))
        next_outputs = next_outputs.detach()
        if (debug):
            print("\nnext_outputs depois de .detach() e antes de .max(1)" + str(next_outputs))
        next_outputs = next_outputs.max(1)
        if (debug):
            print("\nnext_outputs depois de .max(1)" + str(next_outputs))
        next_outputs = next_outputs[0]
        if (debug):
            print("\nnext_outputs completo: " + str(next_outputs))

        target = self.gamma * next_outputs + batch_reward
        if (debug):
            print("\ntarget: " + str(target))

        # TD: Diferença temporal, a função perda irá verificar o quão próximo TD está de zero, isto é:
        # O quão próximo outputs está de target, ambos calculados acima.
        td_loss = F.smooth_l1_loss(outputs, target)
        
        # Zera os gradientes para não acumular entre as iterações
        self.optimizer.zero_grad()

        # Faz a backpropagation usando a função de perda definida
        td_loss.backward()

        # Atualiza os pesos
        self.optimizer.step()