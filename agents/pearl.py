import os
import numpy as np
import torch as T
from utils.utils import Normalize
import config.env_configs as env_configs
from components.networks import ProbabilisticNetwork
from components.memory import ModelBasedMemory
from components.optimizers import CEM
from utils.torch_truncnorm import TruncatedNormal
from .base import Base


class Agent(Base):
    def __init__(self, env, steps_per_day, env_name, models_dir, exploration_mins=180, alpha=0.0003,
                 n_epochs=25, batch_size=32, horizon=20, beta=1, theta=1000, phi=1, hist_length=2, particles=20,
                 expl_del=0.05, output_norm_range=[-1, 1], popsize=25, include_grid=True):

        if env_name not in ["MixedUseFanFCU-v0", "SeminarcenterThermostat-v0", "OfficesThermostat-v0",
                            "Apartments2Thermal-v0"]:
            raise ValueError("Invalid environment name, please select from: [\"MixedUseFanFCU-v0\",\
                                \"SeminarcenterThermostat-v0\", \"OfficesThermostat-v0\", \"Apartments2Thermal-v0\"]")

        ### AGENT PROPERTIES ###
        self.agent_name = 'pets'
        self.model_free = False
        self.model_based = True
        self.model_1_path = os.path.join(models_dir, 'model_1.pth')
        self.model_2_path = os.path.join(models_dir, 'model_2.pth')
        self.model_3_path = os.path.join(models_dir, 'model_3.pth')
        self.model_4_path = os.path.join(models_dir, 'model_4.pth')
        self.model_5_path = os.path.join(models_dir, 'model_5.pth')

        ### CONFIGURE ENVIRONMENT-RELATED VARIABLES ###
        self.env_name = env_name
        self.n_steps = 0
        self.env = env
        self.steps_per_day = steps_per_day
        self.minutes_per_step = int(24 * 60 / self.steps_per_day)
        self.exploration_steps = exploration_mins / self.minutes_per_step
        if env_name == "MixedUseFanFCU-v0":
            self.config = env_configs.MixedUse()
        elif env_name == "SeminarcenterThermostat-v0":
            self.config = env_configs.SeminarcenterThermal()
        elif env_name == 'OfficesThermostat-v0':
            self.config = env_configs.Offices()
        elif env_name == 'Apartments2Thermal-v0':
            self.config = env_configs.Apartments2Thermal()
        self.cont_actions = self.config.continuous_actions
        self.temp_reward = self.config.temp_reward
        self.energy_reward_key = self.config.energy_reward[0]
        self.c02_reward_key = self.config.c02_reward[0]
        self.lower_t = self.config.lower_temp_goal
        self.upper_t = self.config.upper_temp_goal
        self.normaliser = Normalize(env, agent=self.agent_name, config=self.config,
                                    steps_per_day=steps_per_day, include_grid=include_grid)
        self.obs_space = self.normaliser.obs_space
        temp_idx = []
        for temp in self.temp_reward:
            idx = self.obs_space.index(temp)
            temp_idx.append(idx)
        self.temp_idx = temp_idx
        self.energy_idx = self.obs_space.index(self.energy_reward_key)
        if include_grid:
            self.c02_idx = self.obs_space.index(self.c02_reward_key)

        ### CONFIGURE AGENT ###
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.beta = beta
        self.theta = theta
        self.phi = phi
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.hist_length = hist_length
        self.horizon = horizon
        self.output_norm_range = output_norm_range
        self.output_norm_low = T.tensor([np.min(self.output_norm_range)], dtype=T.float).to(self.device)
        self.output_norm_high = T.tensor([np.max(self.output_norm_range)], dtype=T.float).to(self.device)
        self.particles = particles
        self.popsize = popsize
        self.act_space = self.normaliser.act_space
        self.act_dim = len(self.act_space)
        self.obs_dim = len(self.obs_space)
        self.time_dim = 4  # number of time features added manually
        self.expl_del = expl_del  # percentage change in actions from one timestep to next during exploration
        lower = self.normaliser.action_lower_bound
        upper = self.normaliser.action_upper_bound
        self.deltas = {}
        for key in lower.keys() & upper.keys():
            delta = (upper[key] - lower[key]) * self.expl_del
            self.deltas[key] = delta
        self.memory = ModelBasedMemory(agent=self.agent_name, batch_size=self.batch_size,
                                       hist_length=self.hist_length, obs_dim=self.obs_dim + self.time_dim,
                                       particles=self.particles, popsize=self.popsize)
        self.include_grid = include_grid

        ### CONFIGURE OPTIMIZER ###
        self.optimizer = CEM(act_dim=self.act_dim, horizon=self.horizon, reward_function=self.reward_function,
                             popsize=self.popsize)
        self.cem_init_mean = T.zeros(size=(self.horizon, self.act_dim), dtype=T.float, requires_grad=False).to(
            self.device)
        self.cem_init_var = T.tile(T.tensor((1 - (-1)) ** 2 / 16, requires_grad=False),
                                   (self.horizon, self.act_dim)).to(self.device)

        ### CONFIGURE MODELS ###
        self.network_input_dims = ((1 + self.hist_length) * (self.obs_dim + self.time_dim)) + self.act_dim

        self.model_1 = ProbabilisticNetwork(input_dims=self.network_input_dims, output_dims=self.obs_dim,
                                            chkpt_path=self.model_1_path, alpha=alpha
                                            )
        self.model_2 = ProbabilisticNetwork(input_dims=self.network_input_dims, output_dims=self.obs_dim,
                                            chkpt_path=self.model_2_path, alpha=alpha,
                                            )
        self.model_3 = ProbabilisticNetwork(input_dims=self.network_input_dims, output_dims=self.obs_dim,
                                            chkpt_path=self.model_3_path, alpha=alpha
                                            )
        self.model_4 = ProbabilisticNetwork(input_dims=self.network_input_dims, output_dims=self.obs_dim,
                                            chkpt_path=self.model_4_path, alpha=alpha
                                            )
        self.model_5 = ProbabilisticNetwork(input_dims=self.network_input_dims, output_dims=self.obs_dim,
                                            chkpt_path=self.model_5_path, alpha=alpha
                                            )

        self.ensemble = [self.model_1, self.model_2, self.model_3, self.model_4, self.model_5]
        self.model_idxs = [np.arange(i, self.particles, len(self.ensemble)) for i in range(len(self.ensemble))]

        super().__init__(self.env, self.normaliser, self.memory, self.config, self.beta, self.theta, self.act_dim,
                         self.energy_reward_key, self.temp_reward, self.lower_t, self.upper_t, self.n_steps,
                         self.deltas, self.phi, self.include_grid, self.c02_reward_key, self.minutes_per_step,
                         self.obs_space, self.cont_actions)

    def trajectory_sampler(self, init_state, act_seqs):
        '''
        Takes action sequences and propogates each P times through models to horizon H.
        :param init_state: numpy array of most recent observation
        :param act_seqs: tensor of action sequences to be evaluated
        :return particle state trajectories: array of trajectories of shape: (particles, popsize, horizon, obs_dim + 4)
        '''
        # detach action sequences from computational graph
        act_seqs = act_seqs.clone().cpu().detach().numpy()

        particle_act_seq = np.tile(act_seqs, (self.particles, 1, 1, 1))  # duplicate action sequence by no particles
        # convert state to tensor and duplicate

        state_tile = np.tile(init_state, (self.particles, self.optimizer.popsize, 1))  # duplicate state
        # instantiate trajectory tensor
        trajs = np.zeros(
                shape=(self.particles, self.optimizer.popsize, self.horizon, self.obs_dim + self.time_dim))

        for i in range(self.horizon):
            action = particle_act_seq[:, :, i, :]
            trajs[:, :, i, :] = state_tile
            input = np.concatenate((action, state_tile, self.memory.previous_sampled), axis=2)

            # store state in memory after input has been created
            self.memory.store_previous_samples(state_tile)

            for j, model in enumerate(self.ensemble):
                model_input = input[self.model_idxs[j]]  # select subset of data for model
                model.float()  # ensure model parameters are floats
                model_input_T = T.tensor(model_input, dtype=T.float, requires_grad=False).to(self.device)

                with T.no_grad():
                    mean_, var_ = model.forward(model_input_T)

                state_ = TruncatedNormal(loc=mean_, scale=var_, a=-2,
                                         b=2).sample()  # a and b stop discontinuity at -1,1

                # ensure state is in normalised region [-1,1]
                state_ = T.where(state_ < self.output_norm_low, self.output_norm_low, state_)
                state_ = T.where(state_ > self.output_norm_high, self.output_norm_high, state_)

                state_ = self.normaliser.update_time(state_tensor=state_, init_date=self.TS_init_date, \
                                                         init_time=self.TS_init_time, TS_step=i)

                state_tile[self.model_idxs[j]] = state_.cpu().detach().numpy()

        return trajs

    def choose_action(self, observation, prev_action, env):
        '''
        Selects action given current state either by random exploration (when n_step < exploration steps) or by
        sampling actions from CEM optimiser in trajectory sampler.
        :param observation: dict output of simulation describing current state of environment
        :param prev_action: array of action taken in environment at (t - 1), used to select new action near to it
        :param env: environment instance used to get current date and time
        :return action_dict: dictionary describing agent's current best estimate of the optimal action given state
        :return action_norm: array of action selected (shape: (act_dim,)), normalised in range [-1,1] for use
                            in model training
        '''
        obs = self.normaliser.outputs(observation, env, for_memory=False)

        if self.n_steps <= self.exploration_steps:
            action_dict, action_norm = self.explore(prev_action)
            model_input = np.concatenate((action_norm, obs, self.memory.previous), axis=0)  # create model input

        else:
            # store date/time for TS
            min, hour, day, month = env.get_date()
            self.TS_init_time = (min, hour)
            self.TS_init_date = (day, month)

            self.memory.previous_sampled = np.tile(self.memory.previous, (self.particles, self.optimizer.popsize, 1))
            actions = self.optimizer.optimal_action(obs, self.cem_init_mean, self.cem_init_var)
            # self.cem_init_mean = actions

            action_dict = self.normaliser.revert_actions(actions[0].cpu().detach().numpy())
            model_input = np.concatenate((actions[0].cpu().detach().numpy(), obs, self.memory.previous))

        self.memory.store_previous(obs)  # store observation in model memory

        return action_dict, model_input

    def reward_function(self, init_state: np.array, act_seqs: T.tensor):
        '''
        Takes popsize action sequences, runs each through a trajectory sampler to obtain P-many possible trajectories
        per sequence and then calculates expected reward of said sequence
        :param init_state: Tensor of initial normalised state of environment of shape (obs_dim,)
        :param act_seqs: Tensor of candidate action sequences of shape (popsize, horizon)
        :return rewards: Tensor of expected reward for each action sequence of shape (popsize,)
        '''

        particle_trajs = self.trajectory_sampler(init_state, act_seqs)
        particle_trajs_revert = self.normaliser.model_predictions_to_tensor(particle_trajs)
        rewards = self.planning_reward(particle_trajs_revert)

        return rewards

    def planning_reward(self, particle_trajs: T.tensor):
        '''
        Takes particles trajectories and calculates expected reward for each action sequence
        :param particle_trajs: Tensor of sampled particle trajectories of shape: (particles, popsize, horizon, act_dim)
        :return exp_reward: Tensor of expected rewards for each action trajectory in popsize. Shape: (popsize,)
        '''
        energy_elements = particle_trajs[:, :, :, self.energy_idx]
        temp_elements = particle_trajs[:, :, :, self.temp_idx]

        temp_penalties = T.minimum((self.lower_t - temp_elements) ** 2,
                                   (self.upper_t - temp_elements) ** 2) * -self.theta
        temp_rewards = T.where((self.lower_t >= temp_elements) | (self.upper_t <= temp_elements), temp_penalties,
                               T.tensor([0.0], dtype=T.double))  # zero if in correct range, penalty otherwise
        temp_sum = T.sum(temp_rewards, axis=[2, 3])  # sum across sensors and horizon
        temp_mean = T.mean(temp_sum, axis=0)  # expectation across particles

        if self.include_grid:
            c02_elements = particle_trajs[:, :, :, self.c02_idx]
            energy_elements_kwh = (energy_elements * (self.minutes_per_step / 60)) / 1000
            c02 = (c02_elements * energy_elements_kwh) * -self.phi
            c02_sum = T.sum(c02, axis=2)
            c02_mean = T.mean(c02_sum, axis=0)
            exp_reward = c02_mean + temp_mean

        else:
            energy_sum = T.sum(energy_elements, axis=2) * -self.beta  # get cumulative energy use across each act seq
            energy_mean = T.mean(energy_sum, axis=0)  # take expectation across particles
            exp_reward = energy_mean + temp_mean

        return exp_reward

    def learn(self):
        '''
        Updated parameters of each learning dynamical model in ensemble and return final losses
        :return mean_loss: mean loss across models
        '''

        losses = []

        for j, model in enumerate(self.ensemble):
            print('...updating ensemble model:', j, '...')
            state_action_arr, obs_array, batches = self.memory.generate_batches()
            for i in range(self.n_epochs):
                if i % 10 == 0:
                    print('learning epoch:', i)

                batch_loss = []
                for batch in batches:
                    state_action_T = T.tensor(state_action_arr[batch], dtype=T.float).to(model.device)
                    obs_T = T.tensor(obs_array[batch], dtype=T.float).to(model.device)

                    # update max/min logvar
                    var = T.var(obs_T, axis=0)
                    var = T.where(var == 0, T.min(var[T.nonzero(var)]), var)  # replace zeroes with minimum var
                    logvar = T.log(var).double()
                    model.max_logvar = T.where(logvar > model.max_logvar, logvar, model.max_logvar)
                    model.min_logvar = T.where(logvar < model.min_logvar, logvar, model.min_logvar)

                    model.optimizer.zero_grad()
                    mu_pred, var_pred = model.forward(state_action_T)
                    loss = model.loss(mu_pred, var_pred, obs_T)
                    loss.backward()
                    model.optimizer.step()

                    MSE = T.nn.MSELoss()
                    mse_loss = MSE(mu_pred, obs_T)
                    batch_loss.append(mse_loss.cpu().detach().numpy())

                # log loss at final epoch
                if i == (self.n_epochs - 1):
                    losses.append(np.mean(batch_loss))

        self.memory.clear_memory()

        # mean MSE across models
        mean_loss = np.mean(losses)

        return mean_loss

    def save_models(self):
        '''
        Saves parameters of each model in ensemble to directory
        '''
        print('... saving models ...')
        for model in self.ensemble:
            model.save_checkpoint()

    def load_models(self, paths: list = None):
        '''
        Loads parameters of pre-trained models from directory
        '''
        print('... loading models ...')
        for i, model in enumerate(self.ensemble):
            model.load_checkpoint(paths[i])
