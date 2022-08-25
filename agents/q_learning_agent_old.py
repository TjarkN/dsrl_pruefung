from agents.agent import Agent
import numpy as np
import torch
from torch.utils.data import Dataset



class QLearningAgent(Agent):

    def __init__(self, problem,
                 q_table=None,
                 N_sa=None,
                 gamma=0.99,
                 max_N_exploration=100,
                 R_Max=100,
                 q_table_file="q_table.npy"):
        super().__init__(problem)
        self.actions = problem.get_all_actions()
        self.states = problem.get_all_states()
        if q_table is not None:
            self.q_table = q_table
        else:
            self.q_table = {} # hier die tabelle ersetzt / hier haben wir nur den reinen zustand / qtable wird durch nn ersetzt
        if N_sa is not None:
            self.N_sa = N_sa
        else:
            self.N_sa = {}
        self.gamma = gamma
        self.max_N_exploration = max_N_exploration
        self.R_Max = R_Max
        self.file = q_table_file

    def act(self):
        # perception
        s = self.problem.get_current_state().to_state()
        # lookup in q_table
        action = self.actions[np.argmax(self.q_table[s])]
        return action

    def train(self):
        action = None
        s_new = None
        while True:
            current_state = self.problem.get_current_state()
            r = self.problem.get_reward(current_state)
            s = s_new
            s_new = current_state.to_state()
            if s_new not in self.N_sa.keys():
                self.N_sa[s_new] = np.zeros(len(self.actions))
                self.q_table[s_new] = np.zeros(len(self.actions))
            if action is not None:
                a = self.actions.index(action)
                self.N_sa[s][a] += 1
                self.update_q_values(s, a, r, s_new, self.problem.is_goal_state(current_state)) # bei der aufgabe soll wert mit berechnet werden
            if self.problem.is_goal_state(current_state):
                return self.q_table, self.N_sa
            action = self.choose_GLIE_action(self.q_table[s_new], self.N_sa[s_new])
            # act
            self.problem.act(action)

    # muss überschrieben werden / durch NN (?)
    def update_q_values(self, s, a, r, s_new, is_goal_state):
        #hier remember aufrufen
        if is_goal_state:
            self.q_table[s][a] = self.q_table[s][a] + self.alpha(s, a) * (r-self.q_table[s][a])
        else:
            self.q_table[s][a] = self.q_table[s][a] + self.alpha(s, a) * (r + self.gamma * np.max(self.q_table[s_new]) -
                                                                      self.q_table[s][a])

    def choose_GLIE_action(self, q_values, N_s):
        exploration_values = np.ones_like(q_values) * self.R_Max
        # which state/action pairs have been visited sufficiently
        no_sufficient_exploration = N_s < self.max_N_exploration
        # turn cost to a positive number
        q_values_pos = self.R_Max / 2 + q_values
        # select the relevant values (q or max value)
        max_values = np.maximum(exploration_values * no_sufficient_exploration, q_values_pos)
        # assure that we do not dived by zero
        if max_values.sum() == 0:
            probabilities = np.ones_like(max_values) / max_values.size
        else:
            probabilities = max_values / max_values.sum()
        # select action according to the (q) values
        if np.random.random() < (self.max_N_exploration+0.00001)/(np.max(N_s)+0.00001):
            action = np.random.choice(self.actions, p=probabilities)
        else:
            action_indexes = np.argwhere(probabilities == np.amax(probabilities))
            action_indexes.shape = (action_indexes.shape[0])
            action_index = np.random.choice(action_indexes)
            action = self.actions[action_index]
        return action

    def save_q_table(self):
        np.save(self.file, self.q_table)

    def load_q_table(self):
        self.q_table = np.load(self.file)

    def alpha(self, s, a):
        # learnrate alpha decreases with N_sa for convergence
        alpha = self.N_sa[s][a]**(-1/2)
        return alpha

class ExperienceReplay(Dataset):
    def __init__(self, model,m_memory=100, gamma=0.99,
                 transform=None, target_transform=None):
        self.model = model # äquivalent zur qtable
        self.memory = []
        self.max_memory = m_memory
        self.gamma = gamma
        self.transform = transform
        self.target_transform = target_transform

    def remember(self, e, gameover):
        ## save the state to memory
        # agent gibt hier eine experience rein
        # is goal state or not merken
        # experience
        # nacher neue klasse deep q learning agent
        self.memory.append([e, gameover])
        # we dont want to store infinite memories, so if we
        # have too many, we just delete the oldest ones

        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def update_model(self, model):
        self.model = model

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, idx):
        # get random batch from self.experience_replay
        # tuple of the sars parameter is in the 0 idx of an experience
        s, a, r, s_new = self.memory[idx][0]
        # goal state is in the 1 idx of the experience / bool
        goal_state = self.memory[idx][1]
        # save the state in the features
        features = np.array(s)

        # init labels with old prediction (and later overwrite action a)
        label = self.model[s]

        if goal_state:
            # wenn der state der goal state ist, append reward to position of this action
            label[a] = r
        else:
            # calc reward in connection with nex state and append
            label[a] = r + self.gamma*max(self.model[s_new])

        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            label = self.target_transform(label)

        device = "cpu"

        if torch.cuda.is_available():
            device = "cuda"

        # to device from numpy (only interesting if training takes place on GPU
        features = torch.from_numpy(features).float().to(device)
        label = torch.from_numpy(label).float().to(device)

        return features, label





