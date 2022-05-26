import os
import numpy as np
import torch.nn as nn
import torch as T
from components.networks import LatentODE, DeterministicNetwork, Q
from components.memory import ErodeMemory
from components.optimizers import CEM
from utils.utils import Normalize
from utils.torch_truncnorm import TruncatedNormal
import config.env_configs as env_configs
from .base import Base


class Agent(Base):
    def __init__(self, env, steps_per_day, env_name, models_dir, exploration_mins=540, alpha=0.003, n_epochs=10,
                 batch_size=32, horizon=20, beta=1, theta=1000, phi=1, hist_length=1, window_size=2, latent_dim=200,
                 f_ode_dim=100, z0_samples=10, z0_obs_std=0.01, hist_ode_dim=250, GRU_dim=100, particles=20,
                 solver='dopri5',q_dim=200, pi_dim=200, discount=0.99, mix_coeff=0.05,
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
        self.include_grid = include_grid
        self.lower_t = self.config.lower_temp_goal
        self.upper_t = self.config.upper_temp_goal
        self.normaliser = Normalize(self.env, agent=self.agent_name, config=self.config, steps_per_day=steps_per_day,
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
        self.window_size = window_size
        self.popsize = popsize
        self.expl_del = expl_del
        self.output_norm_range = output_norm_range
        self.output_norm_low = T.tensor([np.min(self.output_norm_range)], dtype=T.float).to(self.device)
        self.output_norm_high = T.tensor([np.max(self.output_norm_range)], dtype=T.float).to(self.device)
        lower = self.normaliser.action_lower_bound
        upper = self.normaliser.action_upper_bound
        self.deltas = {}
        for key in lower.keys() & upper.keys():
            delta = (upper[key] - lower[key]) * self.expl_del
            self.deltas[key] = delta
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
        self.state_time_dim = self.state_dim + self.time_dim
        self.act_dim = len(self.act_space)
        self.state_act_dim = self.state_dim + self.time_dim + self.act_dim
        self.network_input_dims = ((1 + self.window_size) * (self.state_dim + self.time_dim)) + self.act_dim
        self.memory = ErodeMemory(agent=self.agent_name, batch_size=self.batch_size, window_size=self.window_size,
                                  hist_length=self.hist_length, obs_dim=self.state_dim + self.time_dim,
                                  net_inp_dims=self.network_input_dims)

        # create discount vector
        disc_list = []
        self.discount = discount
        disc = discount
        for i in range(self.horizon):
            disc_list.append(disc)
            disc *= self.discount
        self.disc_vector = T.tensor(disc_list, dtype=T.float32)

        ### CONFIGURE MODELS ###
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

            # q and pi params
            'pi_dim': pi_dim,
            'q_dim': q_dim,
            'q_lr': alpha,

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
            'state_action_dim': self.network_input_dims,

            # misc
            'device': self.device,
            'steps_per_day': self.steps_per_day
        }

        self.Q1, self.Q2 = Q(self.latent_ode_params), Q(self.latent_ode_params)
        self.Q1_target, self.Q2_target = Q(self.latent_ode_params), Q(self.latent_ode_params)
        self.pi = DeterministicNetwork(output_dims=self.act_dim, input_dims=self.latent_dim,
                                       chkpt_path=self.latent_ode_params['models_dir'])
        self.model = LatentODE(self.latent_ode_params)
        self.model.to(self.device)
        self.optimiser = T.optim.Adamax(self.model.parameters(), lr=alpha)
        self.kl_cnt = 0
        self.kl_coef = 1

        ### CONFIGURE OPTIMIZER ###
        self.CEM = CEM(act_dim=self.act_dim, horizon=self.horizon, reward_estimator=self.reward_estimator,
                       popsize=self.popsize)
        self.cem_init_mean = T.zeros(size=(self.horizon, self.act_dim), dtype=T.float, requires_grad=False).to(
            self.device)
        self.cem_init_var = T.tile(T.tensor((1 - (-1)) ** 2 / 16, requires_grad=False),
                                   (self.horizon, self.act_dim)).to(self.device)
        self.mix_coeff = mix_coeff

        super(Agent, self).__init__(self.env, self.normaliser, self.memory, self.config, self.beta, self.theta,
                                    self.act_dim, self.energy_reward_key, self.temp_reward, self.lower_t, self.upper_t,
                                    self.n_steps, self.deltas, self.phi, self.include_grid, self.c02_reward_key,
                                    self.minutes_per_step, self.obs_space, self.cont_actions)

    def Q(self, z, a):
        x = T.cat([z, a], dim=-1)

        return self.Q1(x), self.Q2(x)

    def sample_pi(self, z, std=0.05):
        """
        Samples action from learned policy pi
        :param z: latent state (latent_dim)
        :param std: standard deviation for applying noise to sample
        """

        mu = self.pi(z)
        if std > 0:
            std = T.ones_like(mu) * std
            dist = TruncatedNormal(loc=mu, scale=std, a=-2, b=2)  # range [-2,2] to avoid discontinuity at [-1,1]
            act = dist.sample()
            # ensure actions in range [-1,1]
            act = T.where(act < self.act_norm_low, self.act_norm_low, act)
            act = T.where(act > self.act_norm_high, self.act_norm_high, act)
            return act

        return mu

    def traj_sampler(self, init_state, cem_act_seqs):
        """
        Performs trajectory sampling using latent odes
        :param init_state: array of more recent observation (state_dim,)
        :param act_seqs: numpy array of action sequences of shape (popsize, horizon, act_dim)
        :return trajs: array of trajectories of shape: (particles, N, horizon, state_dim)
        """
        cem_act_seqs = cem_act_seqs.clone().cpu().detach().numpy()
        cem_act_seqs = np.tile(cem_act_seqs, (self.particles, 1, 1, 1))  # [part, cem_actions, horizon, act_dim]
        state_tile = np.tile(init_state, (self.particles, self.popsize, 1))  # [particles, popsize, state_dim]
        window = np.tile(self.memory.window, (self.particles, self.popsize, 1))  # [part, pop, window_size*state_dim]
        hist = np.tile(self.memory.history,
                       (self.particles, self.popsize, 1, 1))  # [part, pop, hist_length, state_act_dim]

        # initialise trajectory and pi_action arrays
        trajs = np.zeros(shape=(self.particles, self.CEM.popsize, self.horizon, self.state_dim + self.time_dim))
        pi_act_seqs = np.zeros(shape=(self.particles, self.popsize * self.mix_coeff, self.horizon, self.act_dim))

        for i in range(self.horizon):
            # store traj
            trajs[:, :, i, :] = state_tile

            # format current state-action
            cem_actions = cem_act_seqs[:, :, i, :] # [particles, cem_actions, act_dim]

            # sample some actions from policy
            ### THIS ISNT RIGHT AND NEEDS TO BE FIXED; I CANT SAMPLE ONE PARTICLE ONLY ###
            z = self.model.get_z0(hist[1, -int(self.popsize * self.mix_coeff):, :, :]) # [pi_actions, latent_dim]
            z_tile = np.tile(z, self.particles, 1, 1) # [particles, pi_actions, latent_dim]
            pi_actions = self.pi.sample_pi(z_tile) # [particles, pi_actions, act_dim]

            # update pi_act memory to give back to CEM
            pi_act_seqs[:, :, i, :] = pi_actions

            # concat cem and policy actions
            actions = np.concatenate((cem_actions, pi_actions), axis=1) # [particles, popsize, act_dim]

            state_action = np.concatenate((actions, state_tile, window), axis=2)
            state_action = np.expand_dims(state_action, axis=2) # [part, popsize, 1, net_inp_dims]
            input = np.concatenate((hist, state_action), axis=2)  # [part, popsize, hist_length + 1, net_inp_dims]
            assert input.shape == (self.particles, self.popsize, self.hist_length + 1, self.network_input_dims)

            # predict
            model_input = T.tensor(input, dtype=T.float).to(self.device)
            pred_states = self.model.predict_next_state(history=model_input, train=False)
            assert pred_states.shape == (self.particles, self.popsize, self.state_dim)

            # add time
            pred_states = self.normaliser.update_time(state_tensor=pred_states, init_date=self.TS_init_date,
                                                      init_time=self.TS_init_time, TS_step=i)

            # update window with current state
            window[:, :, self.state_time_dim:] = window[:, :, :(self.window_size - 1) * self.state_time_dim]
            window[:, :, :self.state_time_dim] = state_tile

            # update history with current state-action
            hist[:, :, :-1, :] = hist[:, :, 1:, :]  # shift memory back one timestep
            hist[:, :, -1, :] = state_action.squeeze(axis=2)  # removes expanded dim from above

            state_tile = pred_states.cpu().detach().numpy()

        combined_acts = np.concatenate((cem_act_seqs, pi_act_seqs), axis=1) # [particles, popsize, horizon, act_dim]
        assert combined_acts.shape == (self.particles, self.popsize, self.horizon, self.act_dim)

        # get zs for terminal value function


        return trajs, combined_acts

    def act(self, observation, env, prev_action):
        """
        Selects action given current state either by random exploration (when n_step < exploration steps) or by
        sampling actions from CEM optimiser in trajectory sampler.
        :param observation: dict output of simulation describing current state of environment
        :param prev_action: array of action taken in environment at (t - 1), used to select new action near to it
        :param env: environment instance used to get current date and time
        """
        obs = self.normaliser.outputs(outputs=observation, env=env, for_memory=False)

        if self.n_steps <= self.exploration_steps:
            action_dict, action_norm = self.explore(prev_action)
            window = np.concatenate((action_norm, obs, self.memory.window), axis=0)  # create window
            window = np.expand_dims(window, axis=0) # [1, net_inp_dims]
            model_input = np.concatenate((self.memory.history, window),
                                         axis=0)  # create model input [hist_lenght+1, net_inp_dims]
            assert model_input.shape == (self.hist_length + 1, self.network_input_dims)

        else:
            # store date/time for TS
            min, hour, day, month = env.get_date()
            self.TS_init_time = (min, hour)
            self.TS_init_date = (day, month)

            actions = self.CEM.optimal_action(obs, self.cem_init_mean, self.cem_init_var)
            action = actions[0].cpu().detach().numpy() # take only first action
            window = np.concatenate((action, obs, self.memory.window), axis=0)  # create window
            window = np.expand_dims(window, axis=0)
            model_input = np.concatenate((window, self.memory.history),
                                         axis=0)  # create model input [hist_lenght+1, net_inp_dims]
            assert model_input.shape == (self.hist_length + 1, self.network_input_dims)

            action_dict = self.normaliser.revert_actions(action)

        self.memory.store_window(obs)
        self.memory.store_history(window)

        return action_dict, model_input

    def learn(self):
        '''
        :param trajs: batched array of input trajectories of shape (n_batches, batch_size, state_action_dim)
        :param traj_batch: array of shape (batch_size, traj_length, state_action_dim)
        :return:
        '''

        print('...updating model parameters...')
        # generate batches
        model_inp_array, obs_array, batches = self.memory.generate_batches()
        n_batches = len(batches)

        for epoch in range(self.epochs):
            if epoch % 5 == 0:
                print('learning epoch:', epoch)

            avg_loss = 0.0
            self.kl_coef = 1 - 0.99 ** self.kl_cnt
            self.kl_cnt += 1

            for batch in batches:
                input_batch = T.tensor(model_inp_array[batch], dtype=T.float).to(self.device)
                output_batch = T.tensor(obs_array[batch], dtype=T.float).to(self.device)

                pred_state_mean, pred_state_std, z_dists = self.model.predict_next_state(input_batch)
                loss = self.model.loss(pred_state_mean, output_batch, z_dists, self.kl_coef)
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()
                avg_loss += loss / n_batches

        self.memory.clear_memory()

    def estimate_value(self, init_state: np.array, cem_act_seqs: T.tensor):
        '''
        Takes popsize action sequences, runs each through a trajectory sampler to obtain P-many possible trajectories
        per sequence and then calculates expected reward of said sequence
        :param init_state: Tensor of initial normalised state of environment of shape (obs_dim,)
        :param act_seqs: Tensor of candidate action sequences of shape (popsize, horizon)
        :return rewards: Tensor of expected reward for each action sequence of shape (popsize,)
        '''

        particle_trajs, combined_actions = self.traj_sampler(init_state, cem_act_seqs)
        particle_trajs_revert = self.normaliser.model_predictions_to_tensor(particle_trajs)
        rewards = self.planning_reward(particle_trajs_revert) # [popsize,]

        # value


        return rewards, combined_actions

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

        # apply discounting to horizon
        disc_temp_rewards = T.mul(temp_rewards, self.disc_vector)
        temp_sum = T.sum(disc_temp_rewards, axis=[2, 3])  # sum across sensors and horizon
        temp_mean = T.mean(temp_sum, axis=0)  # expectation across particles

        if self.include_grid:
            c02_elements = particle_trajs[:, :, :, self.c02_idx]
            energy_elements_kwh = (energy_elements * (self.minutes_per_step / 60)) / 1000
            c02 = (c02_elements * energy_elements_kwh) * -self.phi

            # discount
            disc_c02 = T.mul(c02, self.disc_vector)

            c02_sum = T.sum(disc_c02, axis=2)
            c02_mean = T.mean(c02_sum, axis=0)
            exp_reward = c02_mean + temp_mean

        else:
            energy_sum = T.sum(energy_elements, axis=2) * -self.beta  # get cumulative energy use across each act seq
            energy_mean = T.mean(energy_sum, axis=0)  # take expectation across particles
            exp_reward = energy_mean + temp_mean

        # normalise rewards
        exp_reward = self.normaliser.rewards(exp_reward)

        ### NEED TO ADD IN TERMINAL VALUE FUNCTION CALCULATION


        return exp_reward

    def remember(self, observation, model_input):
        obs_norm = self.normaliser.outputs(observation, env=self.env, for_memory=True)
        self.memory.store_memory(model_input=model_input, observation=obs_norm)

    def save_models(self):
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

    def track_q_grad(self, enable=True):
        """Utility function that enables/disables gradient tracking of Q-networks"""
        for m in [self.Q1, self.Q2]:
            for param in m.parameters():
                param.requires_grad(enable)

    def plan_2(self, observation, env, prev_action):
        # normalise observation
        obs = self.normaliser.outputs(outputs=observation, env=env, for_memory=False)

        # exploration policy
        if self.n_steps <= self.exploration_steps:
            action_dict, action_norm = self.explore(prev_action)
            window = np.concatenate((action_norm, obs, self.memory.window), axis=0)  # create window
            window = np.expand_dims(window, axis=0)  # [1, net_inp_dims]
            model_input = np.concatenate((self.memory.history, window),
                                         axis=0)  # create model input [hist_lenght+1, net_inp_dims]
            assert model_input.shape == (self.hist_length + 1, self.network_input_dims)

        else:
            # store date/time for TS
            min, hour, day, month = env.get_date()
            self.TS_init_time = (min, hour)
            self.TS_init_date = (day, month)




            actions = self.CEM.optimal_action(obs, self.cem_init_mean, self.cem_init_var)








            action = actions[0].cpu().detach().numpy()  # take only first action
            window = np.concatenate((action, obs, self.memory.window), axis=0)  # create window
            window = np.expand_dims(window, axis=0)
            model_input = np.concatenate((window, self.memory.history),
                                         axis=0)  # create model input [hist_lenght+1, net_inp_dims]
            assert model_input.shape == (self.hist_length + 1, self.network_input_dims)

            action_dict = self.normaliser.revert_actions(action)

        self.memory.store_window(obs)
        self.memory.store_history(window)

        return action_dict, model_input



