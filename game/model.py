"""
File: model.py
Description: Contains all models used for training
Modified By: Kevin Lizarazu
Date: 04/21/2024
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# DQN Linear Model
class LinearDQN(nn.Module):
    def __init__(self, input_shape, hidden_shape, output_shape):
        super().__init__()
        self.layer1 = nn.Linear(input_shape, hidden_shape)
        self.layer2 = nn.Linear(hidden_shape, output_shape)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Necessary forward-feeding functino for DQN linear models
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x
    
    # Saving model for inference
    def save(self, filename="model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        filename = os.path.join(model_folder_path, filename)
        torch.save(self.state_dict(), filename)

# Load saved model for inferencing
def load(in_shape, hid_shape, out_shape, filename="model.pth"):
    model_folder_path = "./model"
    filename = os.path.join(model_folder_path, filename)

    model = LinearDQN(in_shape, hid_shape, out_shape)
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model

# Trainer object updates and optimizes model
class QTrainer():
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), self.lr)
        self.loss_fn = nn.MSELoss()

    # Updates and optimizes model every step taken in a game
    def train_step(self, state, action, rew, next_state, done):
        # params can come in different dimensions, avoid by turning into tensors
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        next_state = torch.tensor(next_state, dtype=torch.float)
        rew = torch.tensor(rew, dtype=torch.float)

        # check size of any input to determine if unsqueezing is necessary
        # changes tensor by increasing by a dimension (if dim=0)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            next_state = torch.unsqueeze(next_state, 0)
            rew = torch.unsqueeze(rew, 0)
            done = (done, )

        # current inference with model and target model to update
        # state input is a list ranging from 1 to x steps, which is how many inferences are made
        infer = self.model(state)
        target = infer.clone()

        # iterates over the single sample or list of step samples
        for idx in range(len(done)):
            q_new = rew[idx]

            # if still alive, update the Q value via Bellman update equation
            if not done[idx]:
                q_new = rew[idx] + (self.gamma * torch.max(self.model(next_state[idx])))
            target[idx][torch.argmax(action).item()] = q_new

        # optimize parameters and using loss for backwards propagation
        # important to set gradients to 0 before backpropagation because pytorch accumulates the gradients on subsequent backward passes (useful for RNNs)
        self.optimizer.zero_grad()
        loss = self.loss_fn(target, infer)
        loss.backward()
        self.optimizer.step()