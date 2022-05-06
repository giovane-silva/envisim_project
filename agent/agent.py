import socket
import json

import csv
import numpy as np
import pandas as pd
import random
from IPython.display import clear_output

class Agent():
    """
    Cria um agente com o objetivo de resolver o mundo de wumpus utilizando o algoritmo de Q-Learning
    """
    
    def __init__(self, stateSpace, actionSpace, server_ip):
        """
        Inicializa as variáveis da classe
        """
        self.stateSpace = stateSpace
        self.actionSpace = actionSpace
        self.msg = ''                               # string com a msg p/ EnviSim
        self.respEnviSim = ''                       # string com resposta do EnviSim
        self.posX = 0                               # opcional (se quiser saber a pos X no grid) ignorem
        self.posY = 0                               # opcional (se quiser saber a pos Y no grid) ignorem
        self.dir = 'n'                              # começa com a direção norte (pode ser random)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # cria socket = sock
        self.sever_IP = server_ip             # IP do servidor, ajustar conforme necessidade
        self.server_port = 15051                    # Port do servidor, ajustar conforme necessidade
        self.server_address = (self.sever_IP, self.server_port)      # server_adress tem IP + porta
        self.results = []                           # Armazena resultados das epochs
        self.prediction_results = []                # Armazena resultados das epochs na avaliação do agente
        self.energy = 200                           # Energia do agente (quantidade de movimentos até morrer)
        self.state = ''                             # Estado do agente
        self.action = ''                            # Ação tomada pelo agente
        self.previous_action = ''                   # Ação anterior tomada pelo agente 
        self.reward = 0                             # Recompensa do agente pela açõa tomada
        self.q_table = np.zeros([len(stateSpace), len(actionSpace)])    # Tabela de action, state value
        self.hasGold = False                        # Indica se agente tem o ouro
        self.outcome = ''                           # Resultado da epoch
        self.n_sameTile = 0                         # Conta quantidade de ações no mesmo quadrado
        self.tile_originalResponse = 'initial'

    def start_server(self):
        """
        Conecta com o servidor
        """
        try:
            self.sock.connect(self.server_address)
            print('Conectado ao server: %s no port: %s' % self.server_address)
        except:
            print('ERRO ao conectar com servidor')

    def get_server_response(self):
        """
        Recebe mensagem do servidor
        """
        try:
            self.respEnviSim = self.sock.recv(256) # recebe até 256 bytes
        except socket.error as e:
            print('Socket error: ', str(e))

    def restart_server(self):
        """
        Reinicia servidor
        """
        self.msg = "{\"call\":[\"restart\",1]}"
        self.send_server_message()

    def send_server_message(self):
        """
        Envia mensagem para o servidor
        """
        try:
            self.sock.sendall(self.msg.encode('utf-8'))  # envia a msg
            self.msg = ''  # limpa a string
        except socket.error as e:
            print('Socket error: ', str(e))

    def update_qtable(self, alpha, gamma):
        """
        Atualiza q-table
        """
        state_index = list(self.stateSpace.keys()).index(self.state)
        action_index = np.where(self.actionSpace == self.action)[0][0]

        next_state = [k for k, v in self.stateSpace.items() if v == True][0]
        next_state_index = list(self.stateSpace.keys()).index(next_state)

        old_value = self.q_table[state_index, action_index]
        next_max = np.max(self.q_table[next_state_index])

        new_value = (1 - alpha) * old_value + alpha * (self.reward + gamma * next_max)                

        self.q_table[state_index, action_index] = new_value


    def get_action(self, epsilon):
        """
        Seleciona ação do agente
        Se o número for menor do que a taxa de exploração ou for a primeira ação do agente, selecionar movimento aleatório
        Caso contrário, seleciona melhor ação
        """
        if (random.uniform(0, 1) < epsilon) or (self.action == 'rst'):
            self.state = [k for k, v in self.stateSpace.items() if v == True][0]
            self.action = np.random.choice(self.actionSpace)
        else:
            self.state = [k for k, v in self.stateSpace.items() if v == True][0]
            state_index = list(self.stateSpace.keys()).index(self.state)
            self.action = self.actionSpace[np.argmax(self.q_table[state_index])]
        
        # Cria mensagem para ser enviada ao servidor de acordo com a ação selecionada
        if self.action == 'move_F':
            self.msg = "{\"move\":[\"forward\",1]}"
        elif self.action == 'call_F':
            self.msg = "{\"call\":[\"forward\",1]}"
        elif self.action == 'rot_L':
            self.msg = "{\"rotate\":[\"left\",2]}"
        elif self.action == 'rot_R':
            self.msg = "{\"rotate\":[\"right\",2]}"
        elif self.action == 'grab':
            self.msg = "{\"act\":[\"grab\",0]}"
        elif self.action == 'leave':
            self.msg = "{\"act\":[\"leave\",0]}"
        else:
            self.msg = "{\"move\":[\"forward\",1]}"
        
            
    def reset_variables(self):
        """
        Reseta as variáveis ao estado inicial
        """
        for key in self.stateSpace.keys():
            self.stateSpace[key] = False
        self.stateSpace['initial'] = True        
        self.done = False
        self.hasGold = False
        self.energy = 200
        self.outcome = ''
        self.n_sameTile = 0
            
    def reset_stateSpace(self):
        """
        Reseta o estado do agente e coloca todos como False
        """
        for key in self.stateSpace.keys():
            self.stateSpace[key] = False
            
    def find_originalResponse(self, jobj):
        if ('sense' in jobj) and (self.action != 'call_F'):
            jrasc = jobj['sense']
            if len(jobj['sense']) == 1:
                if (jrasc[0] == 'boundary') or (jrasc[0] == 'obstruction'):
                    pass
                else:
                    self.tile_originalResponse = jrasc[0]
            elif len(jobj['sense']) == 2:
                if (jrasc[0][0] == 'flash') and (jrasc[0][1] == 'breeze'):
                    self.tile_originalResponse = 'breezeFlash'
                elif (jrasc[0][0] == 'flash') and (jrasc[0][1] == 'stench'):
                    self.tile_originalResponse = 'stenchFlash'
                elif (jrasc[0][0] == 'breeze') and (jrasc[0][1] == 'stench'):
                    self.tile_originalResponse = 'breezeStench'
                else:
                    pass
            elif len(jobj['sense']) > 2:
                self.tile_originalResponse = 'breezeStenchFlash'
            else:
                self.tile_originalResponse = 'nothing'
        else:
            pass

    def translate_server_response(self):
        """
        Traduz a mensagem recebida pelo servidor
        """
        jobj = json.loads(self.respEnviSim) # 1o. transf string recebida em Json object
        
        self.find_originalResponse(jobj)

        if 'sense' in jobj:
            self.reset_stateSpace()
            jrasc = jobj['sense']
            
            if self.action != 'call_F':
                self.reward = -1

                if not self.hasGold:
                    
                    if (len(jrasc) > 0) and ((jrasc[0] == 'boundary') or (jrasc[0] == 'obstruction')):     # veio apenas 1 info no SENSE
                        self.stateSpace[f'{self.tile_originalResponse}Boundary'] = True
                    else:
                        self.stateSpace[self.tile_originalResponse] = True
                        
                else:
                    if (len(jrasc) > 0) and ((jrasc[0] == 'boundary') or (jrasc[0] == 'obstruction')):     # veio apenas 1 info no SENSE
                        self.stateSpace[f'hasGold_{self.tile_originalResponse}Boundary'] = True
                    else:
                        self.stateSpace[f'hasGold_{self.tile_originalResponse}'] = True

            else:
                self.reward = -1

                if not self.hasGold:
                    
                    if len(jobj['sense']) == 1:
                        if (jrasc[0] == 'boundary') or (jrasc[0] == 'obstruction'):     # veio apenas 1 info no SENSE
                            self.stateSpace[f'{self.tile_originalResponse}Boundary'] = True
                        else:
                            self.stateSpace[f'{self.tile_originalResponse}_call_{jrasc[0]}'] = True
                    elif len(jobj['sense']) == 2:
                        if (jrasc[0][0] == 'flash') and (jrasc[0][1] == 'breeze'):
                            value = 'breezeFlash'
                            self.stateSpace[f'{self.tile_originalResponse}_call_{value}'] = True
                        elif (jrasc[0][0] == 'flash') and (jrasc[0][1] == 'stench'):
                            value = 'stenchFlash'
                            self.stateSpace[f'{self.tile_originalResponse}_call_{value}'] = True
                        elif (jrasc[0][0] == 'breeze') and (jrasc[0][1] == 'stench'):
                            value = 'breezeStench'
                            self.stateSpace[f'{self.tile_originalResponse}_call_{value}'] = True
                        else:
                            pass
                    elif len(jobj['sense']) > 2:
                        value = 'breezeStenchFlash'
                        self.stateSpace[f'{self.tile_originalResponse}_call_{value}'] = True
                    else:
                        value = 'nothing'
                        self.stateSpace[f'{self.tile_originalResponse}_call_{value}'] = True
                
                else:
                    if len(jobj['sense']) == 1:
                        if (jrasc[0] == 'boundary') or (jrasc[0] == 'obstruction'):     # veio apenas 1 info no SENSE
                            self.stateSpace[f'hasGold_{self.tile_originalResponse}Boundary'] = True
                        else:
                            self.stateSpace[f'{self.tile_originalResponse}_call_{jrasc[0]}'] = True
                    elif len(jobj['sense']) == 2:
                        if (jrasc[0][0] == 'flash') and (jrasc[0][1] == 'breeze'):
                            value = 'breezeFlash'
                            self.stateSpace[f'hasGold_{self.tile_originalResponse}_call_{value}'] = True
                        elif (jrasc[0][0] == 'flash') and (jrasc[0][1] == 'stench'):
                            value = 'stenchFlash'
                            self.stateSpace[f'hasGold_{self.tile_originalResponse}_call_{value}'] = True
                        elif (jrasc[0][0] == 'breeze') and (jrasc[0][1] == 'stench'):
                            value = 'breezeStench'
                            self.stateSpace[f'hasGold_{self.tile_originalResponse}_call_{value}'] = True
                        else:
                            pass
                    elif len(jobj['sense']) > 2:
                        value = 'breezeStenchFlash'
                        self.stateSpace[f'hasGold_{self.tile_originalResponse}_call_{value}'] = True
                    else:
                        value = 'nothing'
                        self.stateSpace[f'hasGold_{self.tile_originalResponse}_call_{value}'] = True
                            
        elif 'server' in jobj:           # se veio 'server'
            jrasc = jobj['server']     # jrasc tem o conteúdo da msg
            if jrasc == 'restarted':   # se jrasc = 'restarted' o EnviSim foi resetado
                self.action = 'rst'  # se vai rst indica que acabou de fazer restart
            elif jrasc == 'connected': # veio 'connected', tem que ficar conectado...
                self.action = 'connected'    # faz sempre 'restart' depois do connected
            else:
                pass

        # testa se veio uma resposta resultado - msg 'outcome'
        elif 'outcome' in jobj:          # se veio 'outcome'
            jrasc = jobj['outcome']    # = died(perdeu), success(venceu), grabbed, cannot
            if jrasc == 'died':        # se jrasc = 'died' o agente morreu (perdeu)
                self.reset_stateSpace()
                self.reward = -1000
                if self.hasGold:
                    self.stateSpace['hasGold_danger'] = True
                else:
                    self.stateSpace['danger'] = True
                self.outcome = 'died'

            elif jrasc == 'success':   # se jrasc = 'success', agente venceu(SAIU) com ouro
                if self.stateSpace['hasGold_initial']:
                    self.reward = 1000
                    self.outcome = 'success'
                else:
                    self.reward = -10

            elif jrasc == 'grabbed':   # se jrasc == 'grabbed', agente pegou o ouro
                if (not self.hasGold) and (self.tile_originalResponse == 'goal'):
                    self.hasGold = True
                    self.reward = 200
                    self.energy += self.reward
                    self.reset_stateSpace()
                    self.stateSpace['hasGold_goal'] = True
                else:
                    self.reward = -10

            elif jrasc == 'cannot':    # 'cannot', não pode executar o último comando
                self.reward = -10
            else:                      # 'none', EnviSim fez nada com seu comando anterior
                pass

        # testa se veio uma resposta de colisão - msg 'collision': _ _
        elif 'collision' in jobj:
            self.reset_stateSpace()
            self.reward = -1
            if self.hasGold:
                self.stateSpace[f'hasGold_{self.tile_originalResponse}Boundary'] = True
            else:
                self.stateSpace[f'{self.tile_originalResponse}Boundary'] = True

        elif 'direction' in jobj:
            self.reward = -1
            self.reset_stateSpace()
            if self.hasGold:
                self.stateSpace[f'hasGold_{self.tile_originalResponse}'] = True
            else:
                self.stateSpace[f'{self.tile_originalResponse}'] = True
        
        else:
            pass
        
    def check_sameTile(self):
        """
        Verifica se o agente permaneceu no mesmo quadrado
        """
        if self.action == 'rot_L':
            self.n_sameTile += 1
        elif self.action == 'rot_R':
            self.n_sameTile += 1
        elif self.action == 'call_F':
            self.n_sameTile += 1
        elif len([k for k, v in stateSpace.items() if 'boundary' in k and v == True]) > 0:
            self.n_sameTile += 1
        elif self.action == 'move_F':
            self.n_sameTile = 0
        elif (self.action == 'grab') and (self.stateSpace['hasGold_goal']):
            self.n_sameTile = 0
        else:
            pass
        
    def write_results(self, origin, file_name):
        """
        Transfere os resultados para CSV
        """
        if origin == 'train':
            with open(file_name, 'w') as f:
                write = csv.writer(f) 
                write.writerows(self.results)
        else:
            with open(file_name, 'w') as f:
                write = csv.writer(f) 
                write.writerows(self.prediction_results)
            
    def write_qtable(self):
        """
        Transfere a Q-Table para CSV
        """
        pd.DataFrame(self.q_table, columns = self.actionSpace, index = self.stateSpace).to_csv('./results/q_table.csv')

    def fit(self, epochs, alpha, gamma, epsilon):
        """
        Treina o agente
        """
        epochs = epochs
        alpha = alpha
        gamma = gamma
        epsilon = epsilon
        
        self.start_server()

        for i in range(epochs):
            print(f"Iniciando episódio {i}")
            self.reset_variables()
            self.restart_server()
            self.get_server_response()
            self.translate_server_response()
            
            while not self.done:
                self.get_action(epsilon)
                
                if self.energy < 0:
                    if self.hasGold:
                        self.results.append('grabbed gold - died by energy')
                        self.done = True
                    else:
                        self.results.append('died by energy')
                        self.done = True
                else:
                    self.energy -= 1
                    self.send_server_message()
                    self.get_server_response()
                    self.translate_server_response()
                    if (self.outcome == 'died') or (self.outcome == 'success'):
                        self.update_qtable(alpha, gamma)
                        self.done = True
                        if self.hasGold:
                            self.results.append(f'grabbed gold - {self.outcome}')
                        else:
                            self.results.append(self.outcome)
                    else:
                        self.update_qtable(alpha, gamma)
            
            print(self.results[i])
            if i % 10 == 0:
                clear_output(wait=True)
                
        self.write_results(origin = 'train', file_name = './results/episodes_outcomes.csv')
        self.write_qtable()
        
    def predict(self, epochs, epsilon, QTable=None):
        """
        Avalia o agente
        """

        if QTAble:
            self.q_table = QTAble

        self.start_server()

        for i in range(epochs):
            print(f"Iniciando episódio {i}")
            self.reset_variables()
            self.restart_server()
            self.get_server_response()
            self.translate_server_response()

            while not self.done:
                self.get_action(epsilon)

                if self.energy < 0:
                    if self.hasGold:
                        self.prediction_results.append('grabbed gold - died by energy')
                        self.done = True
                    else:
                        self.prediction_results.append('died by energy')
                        self.done = True
                else:
                    self.energy -= 1
                    self.send_server_message()
                    self.get_server_response()
                    self.translate_server_response()
                    if (self.outcome == 'died') or (self.outcome == 'success'):
                        self.done = True
                        if self.hasGold:
                            self.prediction_results.append(f'grabbed gold - {self.outcome}')
                        else:
                            self.prediction_results.append(self.outcome)
                    else:
                        pass

            print(self.prediction_results[i])
            if i % 10 == 0:
                clear_output(wait=True)
                
        self.write_results(origin = 'predict', file_name = './results/prediction_outcomes.csv')