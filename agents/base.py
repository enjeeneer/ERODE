import copy
import datetime
import numpy as np


class Base:
    '''
    General parent class that defines common model-based agent methods
    '''

    def __init__(self, env, normaliser, memory, config, beta, theta, act_dim, energy_reward_key,
                 temp_reward, lower_t, upper_t, n_steps, deltas, phi, include_grid, c02_reward_key,
                 minutes_per_step, obs_space, cont_actions
                 ):
        self.env, self.normaliser, self.memory, self.config, self.beta = env, normaliser, memory, config, beta
        self.energy_reward_key, self.temp_reward, self.theta = energy_reward_key, temp_reward, theta
        self.lower_t, self.upper_t, self.n_steps, self.deltas, self.act_dim = lower_t, upper_t, n_steps, deltas, act_dim
        self.phi, self.include_grid, self.c02_reward_key = phi, include_grid, c02_reward_key
        self.minutes_per_step, self.obs_space, self.cont_actions = minutes_per_step, obs_space, cont_actions

    def remember(self, observation, model_input):
        '''
        Takes one observation dictionary from the environment and stores in agent memory as normalised array
        :param observation: dict
        :param model_input: tensor of input to model of shape (network_input_dims,)
        '''
        obs_norm = self.normaliser.outputs(observation, env=self.env, for_memory=True)
        self.memory.store_memory(state_action=model_input, observation=obs_norm)

    def calculate_reward(self, state_dict):
        '''
        Calculates reward from dictionary output of environment
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
            c02 = state_dict[self.c02_reward_key]  # gCO2/kWh
            energy_kwh = (state_dict[self.energy_reward_key] * (self.minutes_per_step / 60)) / 1000  # kWh
            c02_reward = -(self.phi * c02 * energy_kwh)  # gC02
            reward = c02_reward + temp_reward
        else:
            energy_reward = -(self.beta * state_dict[self.energy_reward_key])
            reward = energy_reward + temp_reward

        return reward

    def explore(self, prev_action):
        '''

        :param prev_action: dict of previous actions used in simulation (unnormalised)
        :return: action_dict: new dict of actions close to previous
        :return: action_norm: normalised version of above
        '''
        if self.n_steps == 0:
            # select random action at first timestep
            action_norm = np.random.uniform(low=-1, high=1, size=self.act_dim)
            new_action_dict = self.normaliser.revert_actions(action_norm)

        else:
            # select next action close to previous to avoid hardware damage
            old_action_dict = prev_action.copy()
            new_action_dict = {}
            for key, _ in old_action_dict.items():
                if key in self.cont_actions:
                    delta_cont = np.random.uniform(-self.deltas[key], self.deltas[key])
                    candidate_action = old_action_dict[key][0] + delta_cont
                    # equation 3.3 in FYR
                    action = np.maximum(self.normaliser.action_lower_bound[key],
                                        (np.minimum(self.normaliser.action_upper_bound[key], candidate_action))
                                        )
                    new_action_dict[key] = [action]  # list is correct format for simulator
                else:
                    new_action_dict[key] = [np.random.choice(2)] # discrete actions either 0 or 1

            working_dict = copy.deepcopy(new_action_dict)
            action_norm = self.normaliser.norm_actions(working_dict)

        return new_action_dict, action_norm

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
