import os
import numpy as np
import torch as T
from components.networks import LatentODE
from components.memory import ErodeMemory
from components.optimizers import CEM
from utils.utils import Normalize
import config.env_configs as env_configs
from .base import Base

class Agent(Base):
    def __init__(self, env, steps_per_day, env_name, models_dir, explorartion_mins=180, alpha=0.003, n_epochs=10,
                 batch_size=32, horizon=20, beta=1, theta=1000, phi=1, hist_length=3, window_size=2, latent_dim=200,
                 f_ode_dim=100, z0_samples=10, z0_obs_std=0.01, hist_ode_dim=250, GRU_dim=100, particles=20, solver='dopri5',
                 rtol=1e-3, atol=1e-4, include_grid=True, popsize=25, expl_del=0.05, output_norm_range=[-1, 1]):

        ### AGENT PROPERTIES ###
        self.agent_name = 'erode'
        self.model_free = False
        self.model_based = True
        self.model_path = os.path.join(models_dir, 'model.pth')

        ### CONFIGURE ENVIRONMENT-RELATED VARIABLES ###
        self.env_name = env_name
        self.n_steps = 0
        self.env = env
        self.steps_per_day = steps_per_day
        self.minutes_per_step = int(24 * 60 / self.steps_per_day)
        self.exploration_steps = explorartion_mins / self.minutes_per_step
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
        self.normaliser = Normalize(env, agent=self.agent_name, config=self.config, steps_per_day=steps_per_day,
                                    include_grid=include_grid)
        self.obs_space = self.normaliser.obs_space
        self.act_space = self.normaliser.act_space
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
        self.phi = phi
        self.theta = theta
        self.time_dim = 4  # number of time features added manually
        self.lr = alpha
        self.epochs = n_epochs
        self.batch_size = batch_size
        self.horizon = horizon
        self.window_size = 2
        self.popsize = popsize
        self.expl_del = expl_del
        self.output_norm_range = output_norm_range
        self.output_norm_low = T.tensor([np.min(self.output_norm_range)], dtype=T.float).to(self.device)
        self.output_norm_high = T.tensor([np.max(self.output_norm_range)], dtype=T.float).to(self.device)
        self.hist_length = hist_length
        self.latent_dim = latent_dim
        self.f_ode_dim = f_ode_dim
        self.z0_samples = z0_samples
        self.z0_obs_std = z0_obs_std
        self.hist_ode_dim = hist_ode_dim
        self.GRU_dim = GRU_dim
        self.particles = particles
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.state_dim = len(self.obs_space)
        self.act_dim = len(self.act_space)
        self.state_act_dim = self.state_dim + self.time_dim + self.act_dim
        self.memory = ErodeMemory(agent=self.agent_name, batch_size=self.batch_size, hist_length=self.hist_length,
                                       obs_dim=self.state_dim+self.time_dim, state_act_dim=self.state_act_dim)

        ### CONFIGURE MODELS ###
        self.network_input_dims = ((1 + self.window_size) * (self.state_dim + self.time_dim)) + self.act_dim
        self.latent_ode_params = {
            # network hyperparams
            'network_input_dims': self.network_input_dims,
            'models_dir': '../tmp/model_testing',
            'n_epochs': n_epochs,

            # ode-rnn params
            'hist_length': hist_length,
            'latent_dim': latent_dim,
            'enc_ode_f_dim': hist_ode_dim,

            # forward network parameters
            'z0_samples': z0_samples,
            'obsrvd_std': z0_obs_std,
            'for_ode_f_dim': f_ode_dim,

            # gru parameters
            'GRU_unit': GRU_dim,

            # traj sampler params
            'particles': particles,
            'horizon': horizon,

            # ode solver params
            'solver': solver,
            'rtol': rtol,
            'atol': atol,

            # data params
            'batch_size': batch_size,
            'traj_length': hist_length + 2,
            'planning_length': hist_length + horizon + 1,
            'act_dim': self.act_dim,
            'state_dim': self.state_dim,
            'state_action_dim': self.state_act_dim,

            # misc
            'device': self.device,
            'steps_per_day': self.steps_per_day
        }
        self.model = LatentODE(self.latent_ode_params)
        self.optimiser = T.optim.Adamax(self.model.parameters(), lr=alpha)
        self.kl_cnt = 0
        self.kl_coef = 1

        ### CONFIGURE OPTIMIZER ###
        self.CEM = CEM(act_dim=self.act_dim, horizon=self.horizon, reward_function=self.reward_estimator,
                             popsize=self.popsize)
        self.cem_init_mean = T.zeros(size=(self.horizon, self.act_dim), dtype=T.float, requires_grad=False).to(
            self.device)
        self.cem_init_var = T.tile(T.tensor((1 - (-1)) ** 2 / 16, requires_grad=False),
                                   (self.horizon, self.act_dim)).to(self.device)


        super(Agent, self).__init__(self.env, self.normaliser, self.memory, self.config, self.beta, self.theta,
                                    self.act_dim, self.energy_reward_key, self.temp_reward, self.lower_t, self.upper_t,
                                    self.n_steps, self.deltas, self.phi, self.include_grid, self.c02_reward_key,
                                    self.minutes_per_step, self.obs_space, self.cont_actions)

    def plan(self, init_state, act_seqs):
        """
        Performs trajectory sampling using latent odes
        :param init_state: array of more recent observation (state_dim,)
        :param act_seqs: numpy array of action sequences of shape (popsize, horizon, act_dim)
        :return trajs: array of trajectories of shape: (particles, N, horizon, state_dim)
        """

        particle_act_seqs = np.tile(act_seqs, (self.particles, 1, 1, 1)) # [part, popsize, horizon, act_dim]
        state_tile = np.tile(init_state, (self.particles, self.popsize, 1))  # [particles, popsize, state_dim]
        window = np.tile(self.memory.window, (self.particles, self.popsize, 1)) # [part, pop, window_size*state_dim]
        hist = np.tile(self.memory.history, (self.particles, self.popsize, 1, 1)) # [part, pop, hist_length, state_act_dim]


        # initialise trajectory array
        trajs = np.zeros(shape=(self.particles, self.CEM.popsize, self.horizon, self.state_dim + self.time_dim))

        for i in range(self.horizon):
            # store traj
            trajs[:, :, i, :] = state_tile

            # format current state-action
            action = particle_act_seqs[:, :, i, :]
            state_action = np.concatenate((action, state_tile, window), axis=2).reshape(
                            self.particles, self.popsize, 1, self.network_input_dims) # [part, popsize, 1, net_inp_dims]
            input = np.concatenate((hist, state_action), axis=2) # [part, popsize, hist_length + 1, state_act_dim]
            assert input.shape == (self.particles, self.popsize, self.hist_length + 1, self.state_act_dim)

            # predict
            model_input = T.tensor(input, dtype=T.float).to(self.device)
            pred_states = self.model.predict_next_state(history=model_input, train=False)
            assert pred_states.shape == (self.particles, self.popsize, self.state_dim)

            # add time
            pred_states = self.normaliser.update_time(state_tensor=pred_states, init_date=self.TS_init_data,
                                                      init_time=self.TS_init_time, TS_step=i)

            # update window with current state
            window[:, :, self.state_dim:] = window[:, :, :(self.window_size-1)*self.state_dim]
            window[:, :, :self.state_dim] = state_tile

            # update history with current state-action
            hist[:, :, :-1, :] = hist[:, :, 1:, :] # shift memory back one timestep
            hist[:, :, -1, :] = state_action # update action mem (act seqs starts at next state)

            state_tile = pred_states.cpu().detach().numpy()

        return trajs

    def act(self, observation, env, prev_action):
        '''
        Selects action given current state either by random exploration (when n_step < exploration steps) or by
        sampling actions from CEM optimiser in trajectory sampler.
        :param observation: dict output of simulation describing current state of environment
        :param prev_action: array of action taken in environment at (t - 1), used to select new action near to it
        :param env: environment instance used to get current date and time

        '''
        obs = self.normaliser.outputs(observation, env, for_memory=False)

        if self.n_steps <= self.exploration_steps:
            action_dict, action_norm = self.explore(prev_action)
            window = np.concatenate((action_norm, obs, self.memory.window), axis=0).reshape(1,)  # create window
            model_input = np.concatenate((window, self.memory.history), axis=0) # create model input [hist_lenght+1, net_inp_dims]
            assert model_input.shape == (self.hist_length + 1, self.network_input_dims)

        else:
            # store date/time for TS
            min, hour, day, month = env.get_date()
            self.TS_init_time = (min, hour)
            self.TS_init_date = (day, month)

            action = self.optimizer.optimal_action(obs, self.cem_init_mean, self.cem_init_var)
            window = np.concatenate((action, obs, self.memory.window), axis=0)  # create window
            model_input = np.concatenate((window, self.memory.history), axis=0) # create model input [hist_lenght+1, net_inp_dims]
            assert model_input.shape == (self.hist_length + 1, self.network_input_dims)

            action_dict = self.normaliser.revert_actions(action[0].cpu().detach().numpy())

        self.memory.store_previous_state(obs)
        self.memory.store_history(window)

        return action_dict, model_input

    def learn(self, trajs):
        '''
        :param trajs: batched array of input trajectories of shape (n_batches, batch_size, state_action_dim)
        :param traj_batch: array of shape (batch_size, traj_length, state_action_dim)
        :return:
        '''

        print('...updating model parameters...')
        # generate batches


        day_trajs = trajs[(self.day * self.steps_per_day):(self.day + 1) * self.steps_per_day, :, :]
        train_batches = self.data_helper.generate_batches(trajs=day_trajs, batch_size=self.batch_size)
        train_torch = T.from_numpy(train_batches).to(self.device).to(T.float)
        train_input = train_torch[:, :, :-1, :self.state_act_dim]
        train_output = train_torch[:, :, -1, self.act_dim:self.state_act_dim]
        n_batches = train_input.shape[0]

        for epoch in range(self.epochs):
            if epoch % 5 == 0:
                print('learning epoch:', epoch)

            avg_loss = 0.0
            self.kl_coef = 1 - 0.99 ** self.kl_cnt
            self.kl_cnt += 1
            for input_batch, output_batch in zip(train_input, train_output):
                pred_state_mean, pred_state_std, z_dists = self.model.predict_next_state(input_batch)
                loss = self.model.loss(pred_state_mean, output_batch, z_dists, self.kl_coef)

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()
                avg_loss += loss / n_batches

            #print('Epoch{0} | loss = {1:.5f}'.format(epoch, avg_loss))

        #self.save_model()

    def reward_estimator(self, init_state: np.array, act_seqs: T.tensor):
        '''
        Takes popsize action sequences, runs each through a trajectory sampler to obtain P-many possible trajectories
        per sequence and then calculates expected reward of said sequence
        :param init_state: Tensor of initial normalised state of environment of shape (obs_dim,)
        :param act_seqs: Tensor of candidate action sequences of shape (popsize, horizon)
        :return rewards: Tensor of expected reward for each action sequence of shape (popsize,)
        '''

        particle_trajs = self.plan(init_state, act_seqs)
        particle_trajs_revert = self.normaliser.model_predictions_to_tensor(particle_trajs)
        rewards = self.planning_reward(particle_trajs_revert)

        return rewards

    def planning_reward(self, particle_trajs: T.tensor):
        '''
        Takes particles trajectories and calculates expected reward for each action sequence
        :param particle_trajs: Tensor of sampled particle trajectories of shape: (particles, popsize, horizon, state_dim)
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

    def save_model(self):
        '''
        Saves parameters of each model in ensemble to directory
        '''
        print('... saving models ...')
        T.save(self.model.state_dict(), self.model_path)

    def load_models(self):
        '''
        Loads parameters of pre-trained models from directory
        '''
        print('... loading models ...')
        self.model.load_state_dict(T.load(self.model_path))