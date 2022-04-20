import energym
import numpy as np
from agents.ppo import Agent as PPO
import pickle
from tqdm import tqdm
import os
import wandb
import argparse

envs = {
    'MixedUseFanFCU-v0': 'GRC_A_Athens',
    'OfficesThermostat-v0': 'GRC_A_Athens',
    'SeminarcenterThermostat-v0': 'DNK_MJ_Horsens1',
    # 'Apartments2Thermal-v0': 'ESP_CT_Barcelona',
}

envs_timesteps = {
    'OfficesThermostat-v0':  15,
    'SeminarcenterThermostat-v0': 10,
    # 'Apartments2Thermal-v0': 3,
    'MixedUseFanFCU-v0': 15,
}

### COMMAND LINE ARGS ###
# parser = argparse.ArgumentParser()
# parser.add_argument('--batch_size', default=32, type=int)
# parser.add_argument('--learning_rate', default=0.003, type=float)
# parser.add_argument('--horizon', default=20, type=int)
# parser.add_argument('--popsize', default=25, type=int)
# parser.add_argument('--epochs', default=25, type=int)
# parser.add_argument('--exploration_mins', default=25, type=int)
# args = parser.parse_args()

com_period = [
    720,
    1440,
    2880,
    5760
]

if __name__ == '__main__':
    # energym setup
    for key, value in envs.items():
        # for mins in com_period:

        years = 15
        wandb_config = dict(
            # exploration_mins=mins,
            env=key,
            years=years,
            episode_length='monthly',
            n_epochs=25,
            batch_size=64
        )

        ## WANDB SETUP ###
        wandb.init(
            project='pearl',
            entity="enjeeneer",
            config=wandb_config,
            tags=['ppo-oracle'],
            allow_val_change=True
        )
        wandb.config.update(wandb_config, allow_val_change=True)

        print('### WANDB CONFIG ###')
        print(wandb.config)

        weather = value
        env_name = key
        simulation_days = 365

        minutes_per_step = envs_timesteps[key]
        N = int((60 * 24) / minutes_per_step)  # make model updates at end of each day
        steps_per_day = int((60 * 24) / minutes_per_step)
        sim_steps = steps_per_day * simulation_days
        steps_per_month = steps_per_day * 31

        # for exp in alpha:
        # setup experiment logging
        results_dir = os.path.join(os.getcwd(), 'experiments', key, 'ppo', 'icml', 'oracle')
        models_dir = os.path.join(os.getcwd(), 'tmp', key, 'ppo', 'oracle')

        for year in range(years):
            output_name = 'outputs.pickle'
            action_name = 'actions.pickle'
            daily_score_name = 'daily_scores.pickle'
            score_name = 'scores.pickle'

            output_path = os.path.join(results_dir, output_name)
            action_path = os.path.join(results_dir, action_name)
            daily_score_path = os.path.join(results_dir, daily_score_name)
            score_path = os.path.join(results_dir, score_name)

            env = energym.make(env_name, weather=weather, simulation_days=simulation_days)
            agent = PPO(env=env,
                         steps_per_day=steps_per_day,
                         env_name=env_name,
                        batch_size=wandb_config['batch_size'],
                        n_epochs=wandb_config['n_epochs'],
                         models_dir=models_dir,
                         include_grid=True)

            if year > 0:
                agent.load_models()


            ### MAIN MODEL-BASED SCRIPT ###
            if agent.model_based:
                print('### RUNNING MODEL-BASED SCRIPT ###')
                outputs = []
                actions = []
                learn_iters = 0
                daily_scores = []
                step_scores = []
                avg_scores = []

                emissions = []

                prev_action = agent.normaliser.revert_actions(
                    np.random.uniform(low=-1, high=1,
                                      size=agent.act_dim))  # dummy variable for action selection at first timestep

                observation = env.get_output()
                if agent.include_grid:
                    observation = agent.add_c02(observation)
                score = 0
                for i in tqdm(range(sim_steps)):
                    action_dict, model_input = agent.choose_action(observation, prev_action, env)
                    observation_ = env.step(action_dict)
                    if agent.include_grid:
                        observation_ = agent.add_c02(observation_)
                    reward = agent.calculate_reward(observation_)
                    score += reward
                    agent.n_steps += 1
                    if agent.n_steps > agent.past_window_size:  # skip first two state-action pairs
                        agent.remember(observation_, model_input)
                    outputs.append(observation_)
                    actions.append(action_dict)
                    step_scores.append(reward)
                    emissions.append(
                        observation_[agent.energy_reward_key] * (envs_timesteps[key]/60) / 1000
                        * (observation_[agent.c02_reward_key] / 1000)
                    )

                    min, hour, day, month = env.get_date()

                    # exploration phase update
                    if (agent.n_steps < steps_per_day) and (agent.n_steps % agent.batch_size == 0):
                        model_loss = agent.learn()
                        wandb.log({'model_loss': model_loss})
                        learn_iters += 1
                        agent.save_models()

                    # save models at end of commissioning period
                    if agent.n_steps == (wandb_config['exploration_mins'] / minutes_per_step):
                        agent.save_models()

                    # normal update
                    if agent.n_steps % agent.steps_per_day == 0:
                        model_loss = agent.learn()
                        wandb.log({'model_loss': model_loss})
                        learn_iters += 1

                        daily_scores.append(score)
                        avg_score = np.mean(daily_scores[-3:])
                        avg_scores.append(avg_score)
                        _, _, day, month = env.get_date()


                        print('date:', day, '/', month, '--',
                              'today\'s score %.1f' % score, 'avg score %.1f' % avg_score,
                              'learning steps', learn_iters)

                        wandb.log({'mean_reward': avg_score})

                        score = 0

                    # log in wandb
                    # wandb.log({'zone_temp': observation_['Z02_T']})
                    # wandb.log({'C02': observation_['c02']})
                    wandb.log({'cum_emissions': sum(emissions)})
                    # wandb.log({'temp_setpoint': action_dict['Z02_T_Thermostat_sp'][0]})
                    # if env_name == 'MixedUseFanFCU-v0':
                    #     wandb.log({'flowrate setpoint': action_dict['Bd_Fl_AHU1_sp'][0]})

                    observation = observation_
                    prev_action = action_dict

                with open(output_path, 'wb') as f:
                    pickle.dump(outputs, f)

                with open(action_path, 'wb') as f:
                    pickle.dump(actions, f)

                with open(daily_score_path, 'wb') as f:
                    pickle.dump(daily_scores, f)

                with open(score_path, 'wb') as f:
                    pickle.dump(step_scores, f)
                env.close()

            ### MAIN MODEL-FREE SCRIPT ###
            elif agent.model_free:
                print('### RUNNING MODEL-FREE SCRIPT ###')
                outputs = []
                actions = []
                learn_iters = 0
                mon_scores = []
                step_scores = []
                avg_scores = []

                observation = env.get_output()
                if agent.include_grid:
                    observation = agent.add_c02(observation)
                score = 0
                for i in tqdm(range(sim_steps)):
                    action_dict, inp, out = agent.choose_action(observation, env)
                    observation_ = env.step(action_dict)
                    if agent.include_grid:
                        observation_ = agent.add_c02(observation_)
                    reward = agent.calculate_reward(observation_)
                    score += reward
                    if agent.n_steps > agent.past_window_size:
                        agent.store_reward(reward)
                    agent.n_steps += 1

                    outputs.append(observation_)
                    actions.append(action_dict)
                    step_scores.append(reward)
                    min, hour, day, month = env.get_date()

                    observation = observation_

                    # model updates when memory length = batch size
                    # if agent.n_steps % agent.batch_size == 0:
                    #     agent.learn()
                    #     learn_iters += 1

                    if agent.n_steps % (steps_per_month) == 0:
                        agent.learn()
                        agent.save_models()
                        mon_scores.append(score)
                        avg_score = np.mean(mon_scores[-3:])
                        avg_scores.append(avg_score)
                        _, _, day, month = env.get_date()

                        # with open(output_path, 'wb') as f:
                        #     pickle.dump(outputs, f)
                        #
                        # with open(action_path, 'wb') as f:
                        #     pickle.dump(actions, f)

                        #
                        # with open(daily_score_path, 'wb') as f:
                        #     pickle.dump(daily_scores, f)
                        #
                        # with open(score_path, 'wb') as f:
                        #     pickle.dump(step_scores, f)
                        #
                        print('date:', day, '/', month, '--',
                              'today\'s score %.1f' % score, 'avg score %.1f' % avg_score,
                              'learning steps', learn_iters)

                        wandb.log({'monthly_score': score})

                        score = 0

                wandb.log({'annual_score': np.sum(step_scores)})
                env.close()
