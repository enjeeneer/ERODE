import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchdiffeq import odeint_adjoint as odeint
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence, Independent


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


### LATENT ODE NETWORKS ###

class ForwardODE(nn.Module):
    """
    Network that represents forward dynamics of environment.
    """

    def __init__(self, param):
        super(ForwardODE, self).__init__()
        self.param = param
        self.hidden_layer = nn.Linear(self.param['latent_dim'], self.param['for_ode_f_dim'])
        self.tanh = nn.Tanh()
        self.hidden_layer_2 = nn.Linear(self.param['for_ode_f_dim'], self.param['for_ode_f_dim'])
        self.tanh2 = nn.Tanh()
        self.output = nn.Linear(self.param['for_ode_f_dim'], self.param['latent_dim'])

    def forward(self, t, input):
        x = input
        x = self.hidden_layer(x)
        x = self.tanh(x)
        x = self.hidden_layer_2(x)
        x = self.tanh2(x)
        output = self.output(x)
        return output


class HistoryODE(nn.Module):
    '''
    Network for learning ODE that governs historic dynamics in encoding space.
    This take previous observations and helps estimate the latent state at the current timestep.
    '''

    def __init__(self, param):
        super(HistoryODE, self).__init__()
        self.param = param
        self.hidden_layer = nn.Linear(self.param['latent_dim'], self.param['enc_ode_f_dim'])
        self.tanh = nn.Tanh()
        self.hidden_layer_2 = nn.Linear(self.param['enc_ode_f_dim'], self.param['enc_ode_f_dim'])
        self.tanh2 = nn.Tanh()
        self.output = nn.Linear(self.param['enc_ode_f_dim'], self.param['latent_dim'])

    def forward(self, t, input):
        x = input
        x = self.hidden_layer(x)
        x = self.tanh(x)
        x = self.hidden_layer_2(x)
        x = self.tanh2(x)
        output = self.output(x)
        return output


class GRU(nn.Module):
    '''
    Updates the latent prediction of EncoderODE to better reflect the true state-action in the history.
    Follows algo 1 from Rubanova et al. (2019).
    '''

    def __init__(self, param):
        super(GRU, self).__init__()
        self.param = param
        self.update_gate = nn.Sequential(
            nn.Linear(self.param['latent_dim'] * 2 + self.param['state_action_dim'], self.param['GRU_unit']),
            nn.Tanh(),
            nn.Linear(self.param['GRU_unit'], self.param['latent_dim']),
            nn.Sigmoid()
        )
        self.reset_gate = nn.Sequential(
            nn.Linear(self.param['latent_dim'] * 2 + self.param['state_action_dim'], self.param['GRU_unit']),
            nn.Tanh(),
            nn.Linear(self.param['GRU_unit'], self.param['latent_dim']),
            nn.Sigmoid()
        )
        self.new_state_gate = nn.Sequential(
            nn.Linear(self.param['latent_dim'] * 2 + self.param['state_action_dim'], self.param['GRU_unit']),
            nn.Tanh(),
            nn.Linear(self.param['GRU_unit'], self.param['GRU_unit'])
        )
        self.new_state_mean = nn.Sequential(
            nn.Linear(self.param['GRU_unit'], self.param['latent_dim'])
        )
        self.new_state_std = nn.Sequential(
            nn.Linear(self.param['GRU_unit'], self.param['latent_dim'])
        )

    def forward(self, mean, std, obs):
        y = T.cat([mean, std, obs], dim=-1).to(self.param['device']).to(
            T.float)  # mean and std should be output of ODE solve, obs is true value
        update = self.update_gate(y)
        reset = self.reset_gate(y)
        y_concat = T.cat([mean * reset, std * reset, obs], -1).to(self.param['device']).to(T.float)

        new_state_hidden = self.new_state_gate(y_concat)
        new_state_mean = self.new_state_mean(new_state_hidden)
        new_state_std = self.new_state_std(new_state_hidden)

        new_mean = (1 - update) * new_state_mean + update * mean
        new_std = (1 - update) * new_state_std + update * std

        return new_mean, new_std.abs()


class OdeRNN(nn.Module):
    '''
    Takes history of observations and estimates current latent state.
    '''

    def __init__(self, param):
        super(OdeRNN, self).__init__()
        self.param = param
        self.ode_func = HistoryODE(param)
        self.gru = GRU(param)

    def forward(self, history, train=True):
        '''
        returns mean and std of z0 (estimate of current latent state) given previous observations
        :param history: array of previous state actions of shape (batch_size, input_traj_length, state_action_dim)
        :return mean, std: mean and std of z0 distribution obtained via ODE-RNN encoding
        '''

        # get mean and std using ODE (aka encoding) from initial conditions of mean = 0 and std = 0
        # these next 7 lines skip explicit encoding by going straight to ode_func which has latent_dims input
        if train:
            # initial guess at latent state is zeros for t_0
            mean0 = T.zeros(self.param['batch_size'], self.param['latent_dim'], device=self.param['device'])
            std0 = T.zeros(self.param['batch_size'], self.param['latent_dim'], device=self.param['device'])

        else:
            mean0 = T.zeros(self.param['particles'], history.shape[1], self.param['latent_dim'],
                            device=self.param['device'])
            std0 = T.zeros(self.param['particles'], history.shape[1], self.param['latent_dim'],
                           device=self.param['device'])

        mean_ode = odeint(func=self.ode_func,
                          y0=mean0,
                          adjoint_method=self.param['solver'],
                          t=T.tensor([0., 1.], dtype=T.float32, device=self.param['device']),
                          # use odeint from t=0 to t=1
                          rtol=self.param['rtol'],
                          atol=self.param['atol']
                          )[1]  # the [1] here gets us the estimate at the next timestep

        # get mean and std of first point in traj by inputting to GRU with first observation
        if train:
            obs = history[:, 0, :].reshape(-1, self.param['state_action_dim'])
        else:
            obs = history[:, :, 0, :]

        mean, std = self.gru(mean=mean_ode, std=std0, obs=obs)

        for i in range(1, history.shape[-2], 1):
            if train:
                obs = history[:, i, :].reshape(-1, self.param['state_action_dim'])
            else:
                obs = history[:, :, i, :]

            mean_ode = odeint(func=self.ode_func,
                              y0=mean,
                              adjoint_method=self.param['solver'],
                              t=T.tensor([i, i + 1], dtype=T.float32, device=self.param['device']),
                              rtol=self.param['rtol'],
                              atol=self.param['atol'])[1]
            mean, std = self.gru(mean=mean_ode, std=std, obs=obs)

        return mean, std

class LatentODE(nn.Module):
    def __init__(self, param):
        super(LatentODE, self).__init__()
        self.param = param
        self.device = param['device']
        self.epochs = param['n_epochs']
        self.state_dim = param['state_dim']
        self.hist_length = param['hist_length']
        self.act_dim = param['act_dim']
        self.state_act_dim = param['state_action_dim']
        self.z0_samples = param['z0_samples']
        self.batch_size = param['batch_size']
        self.latent_dim = param['latent_dim']
        self.obsrvd_std = param['obsrvd_std']
        self.ode_func = ForwardODE(param)
        self.ode_rnn = OdeRNN(param)
        self.decoder = nn.Sequential(
            nn.Linear(param['latent_dim'], param['state_dim'])
        )
        self.encoder = nn.Sequential(
            nn.Linear(param['latent_dim'] + param['state_action_dim'], param['latent_dim'])
        )
        self.mse = nn.MSELoss()
        self.z0_prior = Normal(T.tensor([0.0], device=self.device), T.tensor([1.0], device=self.device))
        self.traj_length = param['traj_length']
        self.kl_cnt = 0

    def predict_next_state(self, history, train=True):
        """
        if train
        :param history: array of shape (batch_size, hist_length + 1 (includes current state), state_action_dim)
        else
        :param history: array of shape (particles, popsize, hist_length + 1 (includes current state), state_action_dim)
        :return:
        """
        if train:
            assert history.shape[1] == self.hist_length + 1  # i.e. includes current state
            input_traj = history[:, :-1, :]
            state_action = T.tile(history[:, -1, :], (self.z0_samples, 1, 1)) # # [z0_samples, batch_size, state_act_dim]
            z0s, z_dists = self.get_z0(input_traj, train=True)
        else:
            assert history.shape[2] == self.hist_length + 1  # i.e. includes current state
            input_traj = history[:, :, :-1, :]
            state_action = T.tensor(history[:, :, -1, :], dtype=T.float)  # # [z0_samples, batch_size, state_act_dim]
            z0s = self.get_z0(input_traj, plan=True)

        eval_points = T.arange(start=0, end=2, step=1, dtype=T.float32, device=self.device)

        z_ = self.encoder(T.cat([z0s, state_action], dim=-1))
        z_next = odeint(func=self.ode_func, y0=z_, method='dopri5', t=eval_points, rtol=self.param['rtol'],
                        atol=self.param['atol'])[1]  # (num_time_points, z0_samples, batch_size, latent_dim)

        pred_states = self.decoder(z_next)

        if train:
            pred_state_mean = T.mean(pred_states, dim=0) # [batch_size, latent_dim]
            pred_state_std = T.std(pred_states, dim=0)

            return pred_state_mean, pred_state_std, z_dists

        else:
            return pred_states

    def get_z0(self, prev_traj, train=False, plan=False):
        '''
        Encodes a trajectory, and runs through HistoryODE solver to obtain estimate for z0 -- latent representation of
        current state.
        :param prev_traj: array of shape (batch_size, input_traj_length, state_act_dim)
        :param train:
        :return:
        '''
        if train:
            mean_z0, logvar_z0 = self.ode_rnn(prev_traj, train=True) # (batch_size, latent_dim) * 2
            std_z0 = T.exp(0.5 * logvar_z0)

            # get current latent state dist
            z0_dists = Normal(loc=mean_z0, scale=std_z0) # mean and std for each dim of latent dim
            z0 = z0_dists.sample(sample_shape=(T.Size([self.z0_samples])))

            return z0, z0_dists

        elif plan:
            mean_z0, logvar_z0 = self.ode_rnn(prev_traj, train=False)  # (batch_size, latent_dim) * 2
            std_z0 = T.exp(0.5 * logvar_z0)

            # get current latent state dist
            z0_dists = Normal(loc=mean_z0, scale=std_z0)  # mean and std for each dim of latent dim
            z0 = z0_dists.sample()

            return z0

        else:
            z0 = self.z0_prior.sample(sample_shape=(self.param['batch_size'], self.param['latent_dim']))

            return z0

    def loss(self, pred_states_mean, real_states, z0_dists, kl_coef):
        '''
        Equation 9 in Rubanova et al. (2019)
        We take in prediction means and std constructed from 5 samples from z0, from here we can construct
        a distribution over our next state predictions which we can compare with real states
        :param pred_traj: array of shape (batch_size, state_dim)
        :param real_traj: array of shape (batch_size, state_dim)
        :param z0_dists: array of torch distributions (batch_size, )
        :param kl_coef: scalar
        :return loss: scalar
        '''

        # kl term
        kl_div = kl_divergence(z0_dists, self.z0_prior)
        kl_div = kl_div.mean(axis=1)

        gaussian = Independent(Normal(loc=pred_states_mean, scale=self.obsrvd_std), 1) # distribution over predicted datapoint in sequence
        log_prob = gaussian.log_prob(real_states) # we get the log prob of the real datapoints if drawn from our predicted distributions (ideally should be close)
        log_likelihood = log_prob / pred_states_mean.shape[0] # expected log prob a.k.a likelihood across batch

        loss = -T.logsumexp(log_likelihood - kl_coef * kl_div, 0)

        return loss

    def trajectory_loss(self, pred_trajs, true_trajs):
        '''
        Compares the
        :param pred_trajs: array of predicted trajectories of shape (particles, N, horizon, state_dim)
        :param true_trajs: array of true trajectories of shape (N, horizon, state_dim)
        :return: loss: mse at each step in H across all trajectories
        '''
        pred_trajs = np.mean(pred_trajs, axis=0) # [N, horizon, state_dim]
        squared_error = np.sum((pred_trajs - true_trajs)**2, axis=2) # [N, horizon]
        absolute_error = pred_trajs - true_trajs # [N, horizon, state_dim]

        mse = np.mean(squared_error, axis=0) # [horizon,]
        std_se = np.std(squared_error, axis=0) # [horizon,]
        mae_state = np.mean(absolute_error, axis=2) # [N, horizon]
        mae = np.mean(mae_state, axis=0) # [horizon,]
        std_ae = np.std(mae_state, axis=0) # [horizon,]

        return mse, std_se, mae, std_ae, absolute_error