import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DeterministicNetwork(nn.Module):
    def __init__(self, output_dims, input_dims, chkpt_path,
                 output_lower_bound=-1, output_upper_bound=1, hidden_dims=200, alpha=0.0003):
        super(DeterministicNetwork, self).__init__()

        self.checkpoint_file = chkpt_path
        self.model = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, output_dims)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.sigmoid = nn.Sigmoid()
        self.output_lower_bound = output_lower_bound
        self.output_upper_bound = output_upper_bound

    def forward(self, state):
        predictions = self.model(state)
        predictions = (self.output_upper_bound - self.output_lower_bound) * self.sigmoid(predictions) + \
                      self.output_lower_bound

        return predictions

    def save_checkpoint(self):
        T.save(self.model.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, path):
        if path == None:
            self.model.load_state_dict(T.load(self.checkpoint_file))
        else:
            self.model.load_state_dict(T.load(path))


class ProbabilisticNetwork(nn.Module):
    def __init__(self, output_dims, input_dims, chkpt_path,
                 alpha=0.0003, hidden_dims=200, mu_lower_bound=-1, mu_upper_bound=1
                 ):
        super(ProbabilisticNetwork, self).__init__()
        self.output_dims = output_dims
        self.input_dims = input_dims
        self.mu_lower_bound = mu_lower_bound
        self.mu_upper_bound = mu_upper_bound
        self.checkpoint_file = chkpt_path
        self.model = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.Tanh(),
        )
        self.mu = nn.Linear(hidden_dims, output_dims)  # mean for each state prediction
        self.sigma = nn.Linear(hidden_dims, output_dims)  # variance for each state prediction
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.softplus2 = nn.Softplus()

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.max_logvar = T.tensor([np.log(0.25)] * self.output_dims, dtype=float).to(self.device)
        self.min_logvar = T.tensor([np.log(0.25)] * self.output_dims, dtype=float).to(self.device)

    def forward(self, state):
        hidden = self.model(state)

        # constrain mu to range [mu_lower_bound, mu_upper_bound]
        mu = self.mu(hidden)
        mu = (self.mu_upper_bound - self.mu_lower_bound) * self.sigmoid(mu) + self.mu_lower_bound

        # constrain logvar to upper and lower logvar seen in training data
        logvar = self.sigma(hidden)
        logvar = self.max_logvar - self.softplus1(self.max_logvar - logvar)
        logvar = self.min_logvar + self.softplus2(logvar - self.min_logvar)
        var = T.exp(logvar).float()

        return mu, var

    def loss(self, mean_pred, var_pred, observation):
        losses = []
        for i, mean in enumerate(mean_pred):
            dist = T.distributions.multivariate_normal.MultivariateNormal(loc=mean,
                                                                          covariance_matrix=T.diag(var_pred[i]))
            l = -dist.log_prob(observation[i])
            losses.append(l)

        return sum(losses)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, path):
        if path == None:
            self.model.load_state_dict(T.load(self.checkpoint_file))
        else:
            self.model.load_state_dict(T.load(path))
