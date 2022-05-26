import os
import numpy as np
import torch
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
                 batch_size=32, horizon=20, beta=1, theta=1000, phi=1, hist_length=3, latent_dim=200,
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
        self.network_input_dims = self.state_dim + self.time_dim + self.act_dim
        self.memory = ErodeMemory(agent=self.agent_name, batch_size=self.batch_size,
                                  hist_length=self.hist_length, obs_dim=self.state_dim + self.time_dim,
                                  net_inp_dims=self.network_input_dims)

        # create discount vector
        disc_list = []
        self.discount = discount
        disc = discount
        for i in range(self.horizon):
            disc_list.append(disc)
            disc *= self.discount
        self.disc_vector = T.tensor(disc_list, dtype=T.float32).unsqueeze(0) # [1, horizon]
        self.disc_tensor = T.tile(self.disc_vector, dims=(self.particles, self.popsize, 1))

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

    @torch.no_grad()
    def traj_sampler(self, init_state, stoch_acts):
        """
        Performs trajectory sampling using latent odes
        :param init_state: array of more recent observation (state_dim,)
        :param act_seqs: numpy array of action sequences of shape (popsize, horizon, act_dim)
        :return trajs: array of trajectories of shape: (particles, N, horizon, state_dim)
        """
        pi = self.mix_coeff > 0 # include pi actions
        stoch_acts = stoch_acts.clone().cpu().detach().numpy()
        stoch_acts = np.tile(stoch_acts, (self.particles, 1, 1, 1))  # [part, cem_actions, horizon, act_dim]
        state_tile = np.tile(init_state, (self.particles, self.popsize, 1))  # [particles, popsize, state_dim]
        hist = np.tile(self.memory.history, (self.particles, self.popsize, 1, 1))  # [part, pop, hist_length, state_act_dim]

        # initialise trajectory and pi_action arrays
        trajs = np.zeros(shape=(self.particles, self.popsize * (1 - self.mix_coeff), self.horizon, self.state_dim + self.time_dim))
        if pi:
            pi_acts = np.zeros(shape=(self.particles, self.popsize * self.mix_coeff, self.horizon, self.act_dim))

        for i in range(self.horizon):
            # store traj
            trajs[:, :, i, :] = state_tile

            # format current state-action
            stoch_actions = stoch_acts[:, :, i, :] # [particles, cem_actions, act_dim]

            if pi:
                # sample some actions from policy
                z = self.model.get_z0(hist[:, -int(self.popsize * self.mix_coeff):, :, :], plan=True) # [particles, pi_actions, latent_dim] -- each particle a different z0 sampled from dist from which pi selects action
                pi_actions = self.pi.sample_pi(z) # [particles, pi_actions, act_dim]

                # update pi_act memory to give back to CEM
                pi_acts[:, :, i, :] = pi_actions
                actions = np.concatenate((stoch_actions, pi_actions), axis=1)  # [particles, popsize, act_dim]

            else:
                actions = stoch_actions

            state_action = np.concatenate((actions, state_tile), axis=2)
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

            # update history with current state-action
            hist[:, :, :-1, :] = hist[:, :, 1:, :]  # shift memory back one timestep
            hist[:, :, -1, :] = state_action.squeeze(axis=2)  # removes expanded dim from above

            state_tile = pred_states.cpu().detach().numpy()

        combined_acts = np.concatenate((stoch_acts, pi_acts), axis=1) # [particles, popsize, horizon, act_dim]
        assert combined_acts.shape == (self.particles, self.popsize, self.horizon, self.act_dim)

        return trajs, combined_acts

    @torch.no_grad()
    def plan(self, observation, env, prev_action):
        """

        """
        obs = self.normaliser.outputs(outputs=observation, env=env, for_memory=False)

        # Exploration Policy
        if self.n_steps <= self.exploration_steps:
            action_dict, action_norm = self.explore(prev_action)
            state_action = np.concatenate((action_norm, obs), axis=0)  # create state/action
            state_action = np.expand_dims(state_action, axis=0) # [1, net_inp_dims]
            model_input = np.concatenate((self.memory.history, state_action),
                                         axis=0)  # create model input [hist_lenght+1, net_inp_dims]
            assert model_input.shape == (self.hist_length + 1, self.network_input_dims)

        # Policy
        else:
            # store date/time for TS
            min, hour, day, month = env.get_date()
            self.TS_init_time = (min, hour)
            self.TS_init_date = (day, month)

            # CEM
            print('...planning...')
            mean, var, t = self.cfg.init_mean, self.cfg.init_var, 0
            # cem optimisation loop
            while (t < self.cfg.max_iters) and (T.max(var) > self.cfg.epsilon):
                # sample stochastic actions
                stoch_samples = int(self.cfg.popsize * (1 - self.cfg.mix_coeff))
                dist = TruncatedNormal(loc=mean, scale=var, a=-2, b=2)  # range [-2,2] to avoid discontinuity at [-1,1]
                stoch_acts = dist.sample(sample_shape=[stoch_samples,])  # output popsize x horizon x action_dims matrix
                stoch_acts = T.where(stoch_acts < self.act_norm_low, self.act_norm_low, stoch_acts) # clip
                stoch_acts = T.where(stoch_acts > self.act_norm_high, self.act_norm_high, stoch_acts) # clip

                trajs, combined_acts = self.traj_sampler(obs, stoch_acts)

                exp_rewards = self.estimate_value(trajs)  # returns pi_actions appended to CEM actions
                elites = combined_acts[np.argsort(exp_rewards)][:int(self.cfg.elites * self.cfg.popsize)]

                new_mean = T.mean(elites, axis=0)
                new_var = T.var(elites, axis=0)

                mean = self.cfg.kappa * mean + (1 - self.cfg.kappa) * new_mean
                var = self.cfg.kappa * var + (1 - self.cfg.kappa) * new_var

                t += 1

            opt_actions = mean
            action = opt_actions[0].cpu().detach().numpy()  # take only first action

            # variables for memory
            state_action = np.concatenate((action, obs), axis=0)  # create state/action
            state_action = np.expand_dims(state_action, axis=0)
            model_input = np.concatenate((self.memory.history, state_action), axis=0)  # create model input [hist_lenght+1, net_inp_dims]
            assert model_input.shape == (self.hist_length + 1, self.network_input_dims)

            action_dict = self.normaliser.revert_actions(action)

        self.memory.store_history(state_action)

        return action_dict, model_input, obs

    def learn(self):
        '''
        :param trajs: batched array of input trajectories of shape (n_batches, batch_size, state_action_dim)
        :param traj_batch: array of shape (batch_size, traj_length, state_action_dim)
        :return:
        '''

        print('...updating model parameters...')
        # generate batches
        model_inp_array, obs_array, batches = self.memory.sample()
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

    def update_pi(self, zs):
        """
        Updates policy given a trajectory of latent states to horizon H (we likely want to batch this)
        :param zs: vector of latent states
        """
        self.pi.optimizer.zero_grad(set_to_none=True)
        self.track_q_grad(False)

        # loss is a weighted sum of q values
        pi_loss = 0
        for t, z in enumerate(zs):
            action = self.sample_pi(z)
            q = torch.min(*self.Q(z, action))
            pi_loss += -q.mean() * (self.cfg.rho ** t)

        pi_loss.backward()
        self.pi.optimizer.step()
        self.track_q_grad(True)

        return pi_loss.item()


    @torch.no_grad()
    def estimate_value(self, trajs, actions):
        """
        Takes array of trajectories and estimates their value. Doing so is a combination of calculating reward
        from predicted trajectories and added the terminal value at horizon H
        :param trajs: Tensor of trajectories (particles, popsize, horizon, state_dim)
        :param actions: Tensor of action sequences used to generate trajs of shape (particles, popsize, horizon, act_dim)
        :return exp_rewards: Tensor of expected rewards for each action sequence of shape (popsize,)
        """

        # only calculate terminal value if we are using actions sampled from policy
        if self.pi:
            # terminal value
            term_states = trajs[:, :, -(self.hist_length + 1):, :] # [particles, popsize, hist_length + 1, state_dim]
            term_actions = actions[:, :, -(self.hist_length + 1):, :] # [particles, popsize, hist_length + 1, act_dim]
            term_trajs = T.cat([term_actions, term_states], dim=3) # [particles, popsize, hist_length + 1, state_act_dim]
            term_vals = self.terminal_value(term_trajs) # [particles, popsize]
            assert term_vals.shape == (self.particles, self.popsize)

        # unnormalise
        trajs_revert = self.normaliser.model_predictions_to_tensor(trajs)

        # temps
        temp_elements = trajs_revert[:, :, :, self.temp_idx]
        temp_penalties = T.minimum((self.lower_t - temp_elements) ** 2, (self.upper_t - temp_elements) ** 2) * -self.theta
        temp_rewards = T.where((self.lower_t >= temp_elements) | (self.upper_t <= temp_elements), temp_penalties,
                               T.tensor([0.0], dtype=T.double))  # zero if in correct range, penalty otherwise
        temp = T.sum(temp_rewards, axis=[3])  # [particles, popsize, horizon]

        # c02
        energy_elements = trajs_revert[:, :, :, self.energy_idx]
        c02_elements = trajs_revert[:, :, :, self.c02_idx]
        energy_elements_kwh = (energy_elements * (self.minutes_per_step / 60)) / 1000
        c02 = (c02_elements * energy_elements_kwh) * -self.phi # [particles, popsize, horizon]

        # total
        total = c02 + temp
        total_disc = total * self.disc_tensor
        total_norm = self.normaliser.rewards(total_disc) # [particles, popsize, horizon]
        total_norm = T.sum(total_norm, dim=2) # [particles, popsize]
        rewards = total_norm + term_vals

        exp_rewards = T.mean(rewards, dim=0) # [popsize]

        return exp_rewards

    @torch.no_grad()
    def terminal_value(self, term_trajs):
        """
        Takes the final state of a trajectory (requiring hist_length prev states too), estimates z0, and calculates the
        Q-values.
        :param term_trajs: traj up to final state s_H of shape (popsize, hist_length + 1, state_act_dim)
        :return qs: tensor of q values of shape (popsize,)
        """
        z0s = self.model.get_z0(term_trajs, plan=True) # [any, hist_length + 1, latent_dim]
        acts = self.sample_pi(z0s, std=0) # [any, hist_length + 1, act_dim]
        qs = torch.min(*self.Q(z0s, acts)) * (self.discount ** (self.horizon + 1))

        return qs

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





