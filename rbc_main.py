import pickle
import os
import energym
from energym.examples.Controller import LabController
from energym.examples.Controller import SimpleController
from energym.examples.Controller import MixedUseController
import config.env_configs as env_configs
import copy
import datetime

envs = {
    # 'MixedUseFanFCU-v0': 'GRC_A_Athens',
    'SeminarcenterThermostat-v0': 'DNK_MJ_Horsens1',
    'OfficesThermostat-v0': 'GRC_A_Athens',
    'Apartments2Thermal-v0': 'ESP_CT_Barcelona'
}

# setup environment
for key, value in envs.items():
    weather = value
    env = energym.make(key, weather=weather, simulation_days=365)
    simulation_days = 365
    inputs = env.get_inputs_names()
    # config
    if key == "MixedUseFanFCU-v0":
        config = env_configs.MixedUse()
        controller = MixedUseController(control_list=inputs, lower_tol=0.3, upper_tol=0.8, nighttime_setback=True,
                                        nighttime_start=17, nighttime_end=6, nighttime_temp=22)
        minutes_per_step = 15

    elif key == "SeminarcenterThermostat-v0":
        config = env_configs.SeminarcenterThermal()
        controller = SimpleController(control_list=inputs, lower_tol=0.3, upper_tol=0.8,
                                      env="SeminarcenterThermostat-v0", nighttime_setback=True, nighttime_start=17,
                                      nighttime_end=6,nighttime_temp=22)
        minutes_per_step = 10

    elif key == 'OfficesThermostat-v0':
        config = env_configs.Offices()
        controller = SimpleController(control_list=inputs, lower_tol=0.3, upper_tol=0.8, env='OfficesThermostat-v0',
                                      nighttime_setback=True, nighttime_start=17, nighttime_end=6, nighttime_temp=22)
        minutes_per_step = 15

    elif key == 'Apartments2Thermal-v0':
        config = env_configs.Apartments2Thermal()
        controller = LabController(control_list=inputs, lower_tol=0.3, upper_tol=0.8, nighttime_setback=True,
                                   nighttime_start=17, nighttime_end=6, nighttime_temp=22)
        minutes_per_step = 3

    steps_per_day = int((60 * 24) / minutes_per_step)  # one step every 15 minutes, skipping first timestep
    sim_steps = steps_per_day * simulation_days
    include_grid = True

    beta = 1
    theta = 1000
    phi = 1
    lower_t = config.lower_temp_goal
    upper_t = config.upper_temp_goal
    temp_reward_keys = config.temp_reward
    co2_reward_key = config.c02_reward[0]
    energy_reward_key = config.energy_reward[0]


    # design reward function
    def calculate_reward(observation_):
        '''
        Calculates reward from dictionary output of environment
        :param state_dict: Dictionary defining the state of variables in an observation
        :return reward: Scalar reward
        '''

        temp_reward = 0
        for t in temp_reward_keys:
            temp = observation_[t]
            if (lower_t <= temp) and (temp <= upper_t):
                pass
            else:
                temp_reward -= theta * min((lower_t - temp) ** 2, (upper_t - temp) ** 2)

        if include_grid:
            co2 = observation_[co2_reward_key]  # gCO2/kWh
            energy_kwh = (observation_[energy_reward_key] * (minutes_per_step / 60)) / 1000  # kWh
            co2_reward = -(phi * co2 * energy_kwh)  # gC02
            reward = co2_reward + temp_reward
        else:
            energy_reward = -(beta * observation_[energy_reward_key])
            reward = energy_reward + temp_reward

        return reward


    def add_c02(observation_):
        '''
        Takes observation dictionary and adds C02 if include_grid == True
        :param observation_: dictionary of state of environment
        :return observation_:
        '''

        # do not update observation if it already contains C02 data
        if 'Grid_CO2' in observation_:
            return observation_

        obs = copy.deepcopy(observation_)
        min, hour, day, month = env.get_date()

        dt = datetime.datetime(config.c02_year, month, day, hour, min)
        # index c02 data from dataframe using datetime of simulation
        c02 = config.c02_data[config.c02_carbon_col][config.c02_data[config.c02_dt_col] == dt].values[0]
        obs['c02'] = c02

        return obs


    # main control loop
    output_list = []
    reward_list = []
    control_list = []
    outputs = env.get_output()
    if include_grid:
        outputs = add_c02(outputs)
    hour = 0
    month = 1
    score = 0
    n_steps = 0

    for _ in range(sim_steps):
        control = controller.get_control(outputs, 22, month, hour)
        control = {p: control[p] for p in control if p in inputs}
        outputs = env.step(control)
        if include_grid:
            outputs = add_c02(outputs)
        reward = calculate_reward(outputs)
        score += reward
        _, hour, _, month = env.get_date()
        n_steps += 1

        # append results
        control_list += [{p: control[p][0] for p in control}]
        output_list.append(outputs)
        if n_steps % steps_per_day == 0:  # end of day
            reward_list.append(score)
            _, _, day, month = env.get_date()
            print('date:', day, '/', month, '--',
                  'today\'s score %.1f' % score)
            score = 0

    env.close()

    # save results
    results_dir = os.path.join(os.getcwd(), 'experiments', key, 'rbc')
    output_name = 'rbc_outputs_511'
    action_name = 'rbc_actions_511'
    score_name = 'rbc_scores_511'
    output_path = os.path.join(results_dir, output_name)
    action_path = os.path.join(results_dir, action_name)
    score_path = os.path.join(results_dir, score_name)

    with open(output_path, 'wb') as f:
        pickle.dump(output_list, f)

    with open(action_path, 'wb') as f:
        pickle.dump(control_list, f)

    with open(score_path, 'wb') as f:
        pickle.dump(reward_list, f)
