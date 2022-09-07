from agents.q_learning_agent_mas import QLearningAgentMAS, ExperienceReplay
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class DeepQLearningAgent(QLearningAgentMAS):
    def __init__(self, problem, q_table=None, N_sa=None, gamma=0.99, max_N_exploration=100, R_Max=100, batch_size=10, Optimizer=torch.optim.Adam, loss_fn=nn.MSELoss()):
        super().__init__(problem=problem,
                       q_table=q_table,
                       N_sa=N_sa,
                       gamma=gamma,
                       max_N_exploration=max_N_exploration,
                        R_Max=R_Max)

        if q_table is None:
            all_states = np.array(self.states)
            min_values = np.amin(all_states, axis=0)
            max_values = np.maximum(np.ones_like(self.states[0]), np.amax(all_states, axis=0))
            transform = lambda x: (x-min_values) / (max_values - min_values)
            self.q_table = self.create_model(Optimizer, loss_fn, transform)
        self.batch_size = batch_size
        self.experience_replay = ExperienceReplay(self.q_table, transform=transform)
        self.loss_history = []

    def create_model(self, Optimizer, loss_fn, transform):
        return DeepQTable(len(self.states[0]), len(self.actions), Optimizer, loss_fn, transform)

    def update_q_values(self, s, a, r, s_new, is_goal_state):
        self.experience_replay.remember((s ,a ,r, s_new), is_goal_state)
        train_loader = DataLoader(self.experience_replay, batch_size=self.batch_size, shuffle=True)
        self.loss_history += self.q_table.perform_training(train_loader)

class DeepQTable(nn.Module):

    def __init__(self, number_of_states, number_of_actions, Optimizer, loss_fn, transform):
        super(DeepQTable, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(number_of_states, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, number_of_actions)
        )
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

        self.model.to(self.device)
        self.optimizer = Optimizer(self.model.parameters())
        self.loss_fn = loss_fn
        self.transform = transform

    def __getitem__(self, state):
        state = self.transform(np.array(state))
        state = torch.tensor(state).float().to(self.device)
        return self.model(state).cpu().detach().numpy()

    def __setitem__(self, key, value):
        # ignore setting values
        pass

    def forward(self, x):
        return self.model(x)

    def perform_training(self, dataloader):
        loss_history = []
        (X, y) = next(iter(dataloader))
        # Compute prediction and loss

        pred = self(X)
        loss = self.loss_fn(pred, y)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_history.append(loss)
        return loss_history

    def save_model(self, file):
        torch.save(self.model.state_dict(), file)

    def load_model(self, file):
        self.model.load_state_dict(torch.load(file))

