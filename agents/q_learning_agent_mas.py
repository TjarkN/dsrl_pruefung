import torch
from torch.utils.data import Dataset

from agents.agent import Agent
import numpy as np
from time import sleep


class QLearningAgentMAS(Agent):

    def __init__(self, problem, q_table=None, N_sa=None, gamma=0.99, max_N_exploration=0, R_Max=500): # todo maybe change ma explo back
        super().__init__(problem)
        self.actions = problem.get_all_actions()
        self.states = problem.get_all_states()
        if q_table is not None:
            self.q_table = q_table
        else:
            self.q_table = np.zeros((len(self.states), (len(self.actions))))
        if N_sa is not None:
            self.N_sa = N_sa
        else:
            self.N_sa = np.zeros((len(self.states), (len(self.actions))))
        self.gamma = gamma
        self.max_N_exploration = max_N_exploration
        self.R_Max = R_Max

    def act(self):

        active = self.problem.plantsim.get_value("sync[\"isPythonActive\",1]")
        while not active:
            sleep(0.01)
            active = self.problem.plantsim.get_value("sync[\"isPythonActive\",1]")

        # perception
        current_state = self.problem.get_current_state()
        if self.problem.is_goal_state(current_state):
            return None
        s = self.states.index(current_state.to_state())
        # lookup in q_table
        action = self.actions[np.argmax(self.q_table[s])]
        return action

    def train(self):
        # states and action are valid only for an autonomous agent with a certain "id"
        # thus states und actions are saved in dictionaries with the id as key
        action = None
        actions = {}
        states = {}

        while True:
            #pause python when plantsim works
            active = self.problem.plantsim.get_value("sync[\"isPythonActive\",1]")
            while not active:
                sleep(0.01)
                active = self.problem.plantsim.get_value("sync[\"isPythonActive\",1]")

            current_state = self.problem.get_current_state()
            r = self.problem.get_reward(current_state)
            if current_state.id in states and current_state.id in actions:
                s = states[current_state.id]
                action = actions[current_state.id]
            else:
                s = None
                action = None
            s_new = self.states.index(current_state.to_state())
            states[current_state.id] = s_new
            if action is not None:
                a = self.actions.index(action)
                self.N_sa[s, a] += 1
                self.update_q_values(s, a, r, s_new, self.problem.is_goal_state(current_state))
                #self.q_table[s, a] = self.q_table[s, a] + self.alpha(s, a) * (r + self.gamma * np.max(self.q_table[s_new]) - self.q_table[s, a])
            if self.problem.is_goal_state(current_state):
                return self.q_table, self.N_sa

            action = self.choose_GLIE_action(self.q_table[s_new], self.N_sa[s_new])
            actions[current_state.id] = action

            # act
            self.problem.act(action)

    def update_q_values(self, s, a, r, s_new, is_goal_state):
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
        q_values_pos = (self.R_Max / 2 + q_values)
        # select the relevant values (q or max value)
        max_values = np.maximum(exploration_values * no_sufficient_exploration, q_values_pos)
        # assure that we do not divide by zero

        if max_values.sum() == 0:
            probabilities = np.ones_like(max_values)
        else:
            probabilities = np.copy(max_values)
        # set not possible actions to zero

        # norming
        probabilities = probabilities / probabilities.sum()
        # select action according to the (q) values
        if np.sum(no_sufficient_exploration) > 0:
            action = np.random.choice(self.actions, p=probabilities)
        else:
            action = self.actions[np.argmax(probabilities)]
        return action

    def save_q_table(self, file):
        np.save(file, self.q_table)

    def load_q_table(self, file):
        self.q_table = np.load(file)

    def alpha(self, s, a):
        # learning rate alpha decreases with N_sa for convergence
        alpha = self.N_sa[s, a] ** (-1 / 2)
        return alpha

# copied from q_learning_agent
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