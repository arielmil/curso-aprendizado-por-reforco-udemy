import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Fluxo de execução:

# update roda algumas vezes sendo chamado por map.py
# cada vez que update é chamado, o sinal lido pelos sensores é passado para ele
# em cada uma das chamadas, um estado novo é criado a partir desse sinal
# junto a ultima recompensa, ao ultimo estado, e a ultima acao, essas variaveis são pushadas na memoria do Dqn
# Após isso, os valores do ultimo estado, ultima recompensa e ultima ação são atualizados
# Quando update fizer size_reward_window pushes para a memoria,
# um sample de batch_size é colhido criando batch_state, batch_next_state, batch_action e batch_reward
# Todos esses batches são passados para learn
# Learn pega os q_values que o modelo calculou, associados aos indices contidos em batch_action
# Batch_action é criado com um conjunto de actions agrupados
# Cada actino vem da função select_action
# learn então calcula a formula target = R(S, A) + γ*max(Q(S', A'))
# A função perda faz target - q_values para cada target e cada q_value e tira a media
# Os pesos e biases são calculados com back propagation em cima da função perda e atualizados pelo step no otimizador
# Assim a rede aprende.

#Criação da arquitetura da rede neural, terá 5 entradas, e 3 saidas.
class Network(nn.Module):

    #input_size: Quantos neuronios de entrada temos, nb_action: Quantas ações temos (Número de neuronios de saída)
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()

        self.input_size = input_size
        self.nb_action = nb_action
        hidden_layer = 50

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, nb_action)
        )
         
    def forward(self, state):
        q_values = self.model(state)
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
    def sample(self, batch_size, debug=False):

        # Pega aleatoriamente batch_size eventos da memoria.
        # o asterisco antes de random faz com que os valores retornados por random (uma especie de lista), sejam tirados da lista,
        # e passados para zip como argumentos separados (serão tuplas)
        # zip então pega o enesimo elemento de cada tupla, e os une em uma única tupla
        # por exemplo: random.sample retornara [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        # o asterisco fara: (1, 2, 3), (4, 5, 6), (7, 8, 9)
        # zip fara um iterador contendo: (1, 4, 7), (2, 5, 8), (3, 6, 9) 
        
        
        if (debug):
            samples = list(zip(*random.sample(self.memory, batch_size)))
        else:
            samples = zip(*random.sample(self.memory, batch_size))

        if (debug):
            print("\n\nIteravel samples:\n\n" + str(samples))
            print("\nCada sample contido em samples:")
            for sample in samples:
                print("\n" + str(sample))

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
        # Passa samples para a função anonima abaixo, que concatena cada sample em samples
        # Em uma lista na dimensão 0, para criar um tensor
        # A função map é o que permite aplicar a função anonima a cada sample
        return map(lambda sample: torch.Tensor(torch.cat(sample, 0)), samples)
        
# Implementação do Deep-Q-Learning:
class Dqn():
    def __init__(self, input_size, nb_action, gamma, lr=0.007):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gamma = gamma

        #Learning rate
        self.lr = lr

        #Uma amostra de n recompensas associados aos ultimos n estados que o agente esteve
        self.reward_window = []
        self.model = Network(input_size, nb_action).to(self.device)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = lr)

        # torch.Tensor(input_size) cria um tensor 1D com input_size elementos (valores nao inicializados).
        # unsqueeze(0) adiciona uma dimensao no inicio, transformando de shape (input_size,) para (1, input_size).

        #Logica:
            # last_state guarda o ultimo estado observado pelo agente.
            # unsqueeze(0) deixa o tensor com formato de batch (1, amostra), que e o formato esperado pela rede.
    
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state, temperature = 17, debug = False):
        # torch.no_grad() é para não fazer os calculos de gradiente, já que nessa parte não estamos em treinamento.
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
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            probs = F.softmax(
                self.model(state_tensor).squeeze(0) * temperature,
                dim=0
                )
        # probs.multinomial pega aleatóriamente num_samples indices dos valores retornados pelo softmax acima
        # quanto maior um determinado valor, maior as chances de seu indice ser escolhido para ser retornado
        action = probs.multinomial(num_samples=1)
        if (debug):
            print("\n\naction:\n\n" + str(action))
        return action.data[0]
    
    # Parametros retornados pela função sample
    def learn(self, batch_state, batch_next_state, batch_action, batch_reward, debug=False):

        # Coloca todos os batches na GPU
        batch_state = batch_state.to(self.device)
        batch_next_state = batch_next_state.to(self.device)
        batch_action = batch_action.to(self.device)
        batch_reward = batch_reward.to(self.device)

        # Retorna um tensor com dimensoes nb_action x batch_size e certas propriedades, fazendo feed forward
        outputs = self.model(batch_state)
        if (debug):
            print("\nself.model(batch_state):\n\n" + str(outputs))

        # Pega na dimensão 1 do tensor definido acima, todos os valores nos indices contidos em batch_action
        # OBS: unsqueeze(1) é para mudar o shape do tensor batch_action de [a, b, c, d] para [[a], [b], [c], [d]]
        # isso é esperado pela função gather.
        outputs = outputs.gather(1, batch_action.unsqueeze(1))
        if (debug):
            print("\nself.model(batch_state).gather(1, batch_action.unsqueeze(1)):\n\n" + str(outputs))
        
        # Remove a dimensão extra que .gather na linha acima retorna
        # ou seja, passa O tensor de [[v1], [v2], [v3], [v4]] para [v1, v2, v3, v4]
        outputs = outputs.squeeze(1)
        if (debug):
            print("\nself.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1):\n\n" + str(outputs))
        
        # Retorna um tensor com dimensoes nb_action x batch_size e certas propriedades, fazendo feed forward
        next_outputs = self.model(batch_next_state)
        if (debug):
            print("\nself.model(batch_next_state):\n\n" + str(next_outputs))
        # Pega apenas o tensor (exclui as outras propriedades)
        next_outputs = next_outputs.detach()
        if (debug):
            print("\nself.model(batch_next_state).detach():\n\n" + str(next_outputs))
        # Dentro da dimensão 1, retorna um tensor com o maior dos nb_action valores valor de cada linha 
        # de 1 a batch_size, e certas propriedades
        next_outputs = next_outputs.max(1)
        if (debug):
            print("\nself.model(batch_next_state).detach().max(1):\n\n" + str(next_outputs))
        # Pega apenas o campo contendo os valores do tensor
        next_outputs = next_outputs[0]
        if (debug):
            print("\nself.model(batch_next_state).detach().max(1)[0]:\n\n" + str(next_outputs))

        # Implementa a formula R(S, A) + γ*max(Q(S', A'))
        target = self.gamma * next_outputs + batch_reward
        if (debug):
            print("\nself.gamma * self.model(batch_next_state).detach().max(1)[0] + batch_reward:\n\n" + str(target))

        # TD: Diferença temporal, a função perda irá verificar o quão próximo TD está de zero, isto é:
        # O quão próximo outputs está de target, ambos calculados acima.
        # Implementa a formula TD(S, A) = Q(S, A) - R(S, A) + γ*max(Q(S', A'))
        td_loss = F.smooth_l1_loss(outputs, target)
        
        # Zera os gradientes para não acumular entre as iterações
        self.optimizer.zero_grad()

        # Faz a backpropagation usando a função de perda definida
        td_loss.backward()

        # Atualiza os pesos
        self.optimizer.step()

    # Faz a atualização das variaveis do agente, adiciona um novo evento a memoria e faz o aprendizado
    # last_reward e new_signal alimentam o agente a partir da interface grafica
    def update(self, last_reward, new_signal, size_reward_window = 1000):
        # Cria o novo estado baseado no sinal que chegou para o agente
        # Obs: new_signal é uma lista, por isso não precisa de [new_signal].
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        last_action_tensor = torch.LongTensor([int(self.last_action)])
        last_reward_tensor = torch.Tensor([self.last_reward])
        self.memory.push((self.last_state, new_state, last_action_tensor, last_reward_tensor))
        action = self.select_action(new_state)

        batch_size = 100
        if (len(self.memory.memory) > batch_size):
            # Pega um sample com batch_size (100) elementos aleatorios, e cria os batches abaixo
            # A ordem dos retornos é determinada pela ordem da tupla que foi push na linha 201
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(batch_size)
            # Faz o treinamento
            self.learn(batch_state, batch_next_state, batch_action, batch_reward)
        
        # Atualiza as variaveis
        self.last_action = action
        self.last_state = new_state
        self.last_reward = last_reward
        # Para auxiliar a verificação de como o agente está se saindo
        self.reward_window.append(last_reward)

        if (len(self.reward_window) > size_reward_window):
            del (self.reward_window[0])

        return action
    
    # Faz a media das rewards em reward_window para avaliar o quão bem o agente está performando
    def score(self):
        # + 1.0 para evitar divisão por 0
        return sum(self.reward_window) / len(self.reward_window) + 1.0
    
    # Salva a rede neural com o aprendizado (pesos e biases) que ela fez até o momento
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()},
                    'last_brain.pth')
    
    # Carrega a rede neural salva
    def load(self):
        if (os.path.isfile('last_brain.pth')):
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Carregado com sucesso!")
        else:
            print("Erro ao carregar!")