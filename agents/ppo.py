import os
import numpy as np
import torch as T
from utils.utils import Normalize
import config.env_configs as env_configs
from components.networks import DeterministicNetwork
from components.memory import ModelFreeMemory
import copy
import datetime


class Agent:
    def __init__(self, env, steps_per_day, env_name, models_dir, cov_fill_value=0.9, gamma=0.99, alpha=0.0003,
                 gae_lambda=0.95, policy_clip=0.2, beta=1, theta=1000, phi=1, batch_size=32, past_window_size=2,
                 n_epochs=10, exploration_mins=180, include_grid=True):

        if env_name not in ["MixedUseFanFCU-v0", "SeminarcenterThermostat-v0", "OfficesThermostat-v0",
                            "Apartments2Thermal-v0"]:
            raise ValueError("Invalid environment name, please select from: [\"MixedUseFanFCU-v0\",\
                                \"SeminarcenterThermostat-v0\", \"OfficesThermostat-v0\", \"Apartments2Thermal-v0\"]")

        ### CONFIGURE ENVIRONMENT-RELATED VARIABLES ###
        self.env = env
        self.env_name = env_name
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
        self.temp_reward = self.config.temp_reward
        self.energy_reward_key = self.config.energy_reward[0]
        self.c02_reward_key = self.config.c02_reward[0]
        self.lower_t = self.config.lower_temp_goal
        self.upper_t = self.config.upper_temp_goal

        ### CONFIGURE AGENT ###
        self.agent_name = 'ppo'
        self.model_free = True
        self.model_based = False
        self.actor_path = os.path.join(models_dir, 'actor.pth')
        self.critic_path = os.path.join(models_dir, 'critic.pth')
        self.n_steps = 0
        self.exploration_mins = exploration_mins
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.beta = beta
        self.theta = theta
        self.phi = phi
        self.alpha = alpha
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.past_window_size = past_window_size
        self.normaliser = Normalize(env, agent=self.agent_name, config=self.config,
                                    steps_per_day=steps_per_day, include_grid=include_grid)
        self.act_space = self.normaliser.act_space
        self.obs_space = self.normaliser.obs_space
        self.act_dim = len(self.act_space)
        self.obs_dim = len(self.obs_space)
        self.time_dim = 4
        self.network_input_dims = ((1 + self.past_window_size) * (self.obs_dim + self.time_dim))
        self.actor = DeterministicNetwork(output_dims=self.act_dim, input_dims=self.network_input_dims, alpha=alpha,
                                            chkpt_path=self.actor_path)
        self.critic = DeterministicNetwork(output_dims=1, input_dims=self.network_input_dims, alpha=alpha,
                                            chkpt_path=self.critic_path)
        self.memory = ModelFreeMemory(batch_size=self.batch_size, obs_dim=self.obs_dim + self.time_dim,
                                past_window_size=self.past_window_size)
        self.cov_var = T.full(size=(self.act_dim,), fill_value=cov_fill_value)
        self.cov_mat = T.diag(self.cov_var).to(self.actor.device)
        self.include_grid = include_grid

    def choose_action(self, observation, env):
        '''
        Selects action given current state either from policy distribution (when n_step < exploration steps) or
        deterministically.
        :param observation: dict output of simulation describing current state of environment
        :param env: environment instance used to get current date and time
        :return action_dict: dictionary describing agent's current best estimate of the optimal action given state
        :return action_norm: array of action selected (shape: (act_dim,)), normalised in range [-1,1] for use
                            in model training
        '''
        # normalise/clean observation and convert to tensor
        obs = self.normaliser.outputs(observation, env, for_memory=False)
        input = np.concatenate((obs, self.memory.previous), axis=0)
        input_tensor = T.tensor(input, dtype=T.float).to(self.actor.device)

        mean = self.actor.forward(input_tensor)  # mean from multivariate normal dist
        value = self.critic.forward(input_tensor)
        dist = T.distributions.MultivariateNormal(mean, self.cov_mat)  # create normal dist

        if self.n_steps <= self.exploration_steps:
            action = dist.sample() # sample from distribution for exploration
            log_prob = dist.log_prob(action)
        else:
            action = mean # deterministically select mean of policy
            log_prob = dist.log_prob(action)

        # detach from computational graphs
        action = action.cpu().detach().numpy()
        log_prob = log_prob.cpu().detach()
        value = value.cpu().detach()

        # store memories
        self.memory.store_previous(obs)
        if self.n_steps > self.past_window_size:
            self.memory.store_memory(input, action, log_prob, value)

        # convert action back to dictionary
        action_dict = self.normaliser.revert_actions(action)

        # log data for analysis
        logger = np.concatenate((action, input), axis=0)

        return action_dict, logger, obs

    def calculate_reward(self, state_dict):
        '''
        Calculates reward from dictionary output of environment. Reward depends on whether grid C02 is included in
        observation space.
        :param state_dict: Dictionary defining the state of variables in an observation
        :return reward: Scalar reward
        '''

        temp_reward = 0
        for t in self.temp_reward:
            temp = state_dict[t]
            if (self.lower_t <= temp) and (temp <= self.upper_t):
                pass
            else:
                temp_reward -= self.theta * min((self.lower_t - temp) ** 2, (self.upper_t - temp) ** 2)

        if self.include_grid:
            c02 = state_dict[self.c02_reward_key] # gCO2/kWh
            energy_kwh = (state_dict[self.energy_reward_key] * (self.minutes_per_step / 60)) / 1000 # kWh
            c02_reward = -(self.phi * c02 * energy_kwh) # gC02
            reward = c02_reward + temp_reward
        else:
            energy_reward = -(self.beta * state_dict[self.energy_reward_key])
            reward = energy_reward + temp_reward

        return reward

    def learn(self):
        '''
        Updates parameters of actor and critic networks
        '''
        for i in range(self.n_epochs):
            if i % 10 == 0:
                print('learning epoch:', i)
            model_input_arr, action_arr, old_probs_arr, vals_arr, reward_arr, batches = self.memory.generate_batches()
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t

            # normalize advantages to decrease variance
            advantage = (advantage - np.mean(advantage)) / (np.std(advantage) + 1e-10)
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)

            for batch in batches:
                # get old actions, values and log probs
                states = T.tensor(model_input_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
                old_actions = T.tensor(action_arr[batch]).to(self.actor.device)

                # calculate new actions, values and log probs given updates actors/critics
                new_means = self.actor.forward(states)
                new_dist = T.distributions.MultivariateNormal(new_means, self.cov_mat)
                new_probs = new_dist.log_prob(old_actions)
                new_values = self.critic.forward(states)
                critic_value = T.squeeze(new_values)

                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[
                    batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                # get mse
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                # update network parameters
                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

    def add_c02(self, observation_):
        '''
        Takes observation dictionary and adds C02 if include_grid == True
        :param observation_: dictionary of state of environment
        :return observation_:
        '''

        # do not update observation if it already contains C02 data
        if 'Grid_CO2' in self.obs_space:
            return observation_

        obs = copy.deepcopy(observation_)
        min, hour, day, month = self.env.get_date()

        dt = datetime.datetime(self.config.c02_year, month, day, hour, min)
        # index c02 data from dataframe using datetime of simulation
        c02 = \
         self.config.c02_data[self.config.c02_carbon_col][self.config.c02_data[self.config.c02_dt_col] == dt].values[0]
        obs['c02'] = c02

        return obs

    def store_reward(self, reward: float):
        '''
        Takes reward obtained from environment and stores in memory
        :param reward: scalar quantifying value of action of shape(1,)
        '''
        self.memory.rewards.append(reward)

    def save_models(self):
        '''
        Saves parameters of each model in ensemble to directory
        '''
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        '''
        Loads parameters of pre-trained models from directory
        '''
        print('... loading models ...')
        self.actor.load_checkpoint(self.actor_path)
        self.critic.load_checkpoint(self.critic_path)
