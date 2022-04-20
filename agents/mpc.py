import os
import numpy as np
import torch as T
from utils.utils import Normalize
import config.env_configs as env_configs
from components.memory import ModelBasedMemory
from components.networks import DeterministicNetwork
from .base import Base

class Agent(Base):
    def __init__(self, env, steps_per_day, env_name, models_dir, exploration_mins=180, alpha=0.0003, n_epochs=50,
                 batch_size=32, horizon=20, popsize=25, beta=1, theta=1000, phi=1, past_window_size=2, expl_del=0.05,
                 output_norm_range=[-1, 1], include_grid=True):

        if env_name not in ["MixedUseFanFCU-v0", "SeminarcenterThermostat-v0", "OfficesThermostat-v0",
                            "Apartments2Thermal-v0"]:
            raise ValueError("Invalid environment name, please select from: [\"MixedUseFanFCU-v0\",\
                                \"SeminarcenterThermostat-v0\", \"OfficesThermostat-v0\", \"Apartments2Thermal-v0\"]")
        ### AGENT PROPERTIES ###
        self.agent_name = 'mpc'
        self.model_free = False
        self.model_based = True
        self.model_path = os.path.join(models_dir, 'model.pth')

        ### CONFIGURE ENVIRONMENT ###
        self.env = env
        self.steps_per_day = steps_per_day
        self.n_steps = 0
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
        self.normaliser = Normalize(env=env, agent=self.agent_name, config=self.config,
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
        self.act_space = self.normaliser.act_space
        self.act_dim = len(self.act_space)
        self.obs_dim = len(self.obs_space)
        self.time_dim = 4  # number of time features added manually
        self.beta = beta
        self.theta = theta
        self.phi = phi
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.past_window_size = past_window_size
        self.horizon = horizon
        self.exploration_mins = exploration_mins
        self.output_norm_range = output_norm_range
        self.output_norm_low = T.tensor([np.min(self.output_norm_range)], dtype=T.float)
        self.output_norm_high = T.tensor([np.max(self.output_norm_range)], dtype=T.float)
        self.popsize = popsize
        lower = self.normaliser.action_lower_bound
        upper = self.normaliser.action_upper_bound
        self.expl_del = expl_del
        self.deltas = {}
        for key in lower.keys() & upper.keys():
            delta = (upper[key] - lower[key]) * self.expl_del
            self.deltas[key] = delta
        self.memory = ModelBasedMemory(agent=self.agent_name, batch_size=self.batch_size, past_window_size=self.past_window_size,
                             obs_dim=self.obs_dim + self.time_dim, popsize=self.popsize)
        self.include_grid = include_grid

        ### CONFIGURE MODEL ###
        self.network_input_dims = ((1 + self.past_window_size) * (self.obs_dim + self.time_dim)) + self.act_dim
        self.model = DeterministicNetwork(input_dims=self.network_input_dims, output_dims=self.obs_dim, alpha=alpha,
                                          chkpt_path=self.model_path)

        super().__init__(self.env, self.normaliser, self.memory, self.config, self.beta, self.theta, self.act_dim,
                         self.energy_reward_key, self.temp_reward, self.lower_t, self.upper_t, self.n_steps,
                         self.deltas, self.phi, self.include_grid, self.c02_reward_key, self.minutes_per_step,
                         self.obs_space, self.cont_actions)

    def choose_action(self, observation, prev_action, env):
        '''
        Selects action given current state either by random exploration (when n_step < exploration steps) or by
        random shooting of action sequences through model using trajectory sampler.
        :param observation: dict output of simulation describing current state of environment
        :param prev_action: array of action taken in environment at (t - 1), used to select new action near to it
        :param env: environment instance used to get current date and time
        :return action_dict: dictionary describing agent's current best estimate of the optimal action given state
        :return action_norm: array of action selected (shape: (act_dim,)), normalised in range [-1,1] for use
                            in model training
        '''
        obs = self.normaliser.outputs(observation, env, for_memory=False)  # normalise state

        if self.n_steps <= self.exploration_steps:
            action_dict, action_norm = self.explore(prev_action)
            model_input = np.concatenate((action_norm, obs, self.memory.previous), axis=0)

        else:
            # store date/time for TS
            min, hour, day, month = env.get_date()
            self.TS_init_time = (min, hour)
            self.TS_init_date = (day, month)

            self.memory.previous_sampled = np.tile(self.memory.previous, (self.popsize, 1))  # reset sampled memory
            actions_trajs = self.generate_action_trajectories()
            state_trajs = self.trajectory_sampler(init_state=obs, act_seqs=actions_trajs)
            state_trajs_revert = self.normaliser.model_predictions_to_tensor(state_trajs)
            rewards = self.planning_reward(state_trajs_revert)
            action = actions_trajs[np.argsort(rewards)][0, 0, :] # first action from optimal sequence

            action_dict = self.normaliser.revert_actions(action)
            model_input = np.concatenate((action, obs, self.memory.previous), axis=0)

        self.memory.store_previous(obs)

        return action_dict, model_input

    def generate_action_trajectories(self):
        '''
        Generates array of action trajectories that begin with random action and then propogate following a random walk
        :return actions: array of actions in range [-1,1] of shape (popsize, horizon, act_dim)
        '''
        actions = np.random.uniform(low=self.output_norm_low.numpy(), high=self.output_norm_high.numpy(),
                                    size=(self.popsize, self.horizon, self.act_dim))
        for i in range(self.horizon - 1):
            deltas = np.random.uniform(low=-self.expl_del, high=self.expl_del, size=(self.popsize, self.act_dim))
            action = np.maximum(self.output_norm_low.numpy(), np.minimum(actions[:, i, :] + deltas,
                                self.output_norm_high.numpy()))
            actions[:, i + 1, :] = action

        return actions

    def trajectory_sampler(self, init_state, act_seqs):
        '''
        Takes one action sequence from popsize action sequences and propogates P times through models to horizon H.
        :param init_state:
        :param act_seqs:

        :return trajs: array of trajectories of shape: (popsize, horizon, obs_dim + 4) (+4 for time features)
        '''

        state_tile = np.tile(init_state, (self.popsize, 1))
        trajs = np.zeros(shape=(self.popsize, self.horizon, self.obs_dim + self.time_dim))

        for i in range(self.horizon):
            action = act_seqs[:, i, :]
            trajs[:, i, :] = state_tile
            input = np.concatenate((action, state_tile, self.memory.previous_sampled), axis=-1)

            # store state in memory after input has been created
            self.memory.store_previous_samples(state_tile)

            # predict next state
            model_input = T.tensor(input, dtype=T.float, requires_grad=False).to(self.model.device)

            with T.no_grad():
                state_ = self.model.forward(model_input)

            state_ = self.normaliser.update_time(state_tensor=state_, init_date=self.TS_init_date,
                                                 init_time=self.TS_init_time, TS_step=i)

            state_tile = state_.cpu().detach().numpy()

        return trajs

    def planning_reward(self, trajs):
        '''
        Takes state trajectories and calculates expected reward for sequence
        :param trajs: Tensor of sampled trajectories of shape: (popsize, horizon, act_dim)
        :return exp_reward: Tensor of expected rewards for each action trajectory in popsize. Shape: (popsize,)
        '''
        energy_elements = trajs[:, :, self.energy_idx]
        temp_elements = trajs[:, :, self.temp_idx]

        temp_penalties = T.minimum((self.lower_t - temp_elements) ** 2,
                                   (self.upper_t - temp_elements) ** 2) * -self.theta
        temp_rewards = T.where((self.lower_t >= temp_elements) | (self.upper_t <= temp_elements), temp_penalties,
                               T.tensor([0.0], dtype=T.double))  # zero if in correct range, penalty otherwise
        temp_sum = T.sum(temp_rewards, axis=[1, 2])  # sum across sensors and horizon

        if self.include_grid:
            c02_elements = trajs[:, :, self.c02_idx] # gC02
            energy_elements_kwh = (energy_elements * (self.minutes_per_step / 60)) / 1000 # kWh
            c02 = (c02_elements * energy_elements_kwh) * -self.phi
            c02_sum = T.sum(c02, axis=1)
            exp_reward = c02_sum + temp_sum
        else:
            energy_sum = T.sum(energy_elements, axis=1) * -self.beta
            exp_reward = energy_sum + temp_sum

        return exp_reward

    def learn(self):
        '''

        :return losses: mean MSE across batches in final epoch
        '''
        # updates the parameters of the learned dynamical model

        global losses
        for i in range(self.n_epochs):
            if i % 10 == 0:
                print('learning epoch:', i)
            state_actions_arr, states_arr, batches = self.memory.generate_batches()

            batch_loss = []
            for batch in batches:
                state_actions = T.tensor(state_actions_arr[batch], dtype=T.float).to(self.model.device)
                states_pred = self.model.forward(state_actions)
                states_true = T.tensor(states_arr[batch], dtype=T.float).to(self.model.device)

                MSE = T.nn.MSELoss()
                loss = MSE(states_pred, states_true)
                self.model.optimizer.zero_grad()
                loss.backward()
                self.model.optimizer.step()
                batch_loss.append(loss.cpu().detach().numpy())

            if i == (self.n_epochs - 1):
                losses = np.mean(batch_loss)

        self.memory.clear_memory()

        return losses

    def save_models(self):
        print('... saving models ...')
        self.model.save_checkpoint()

    def load_models(self, path=None):
        print('... loading models ...')
        self.model.load_checkpoint(path)
