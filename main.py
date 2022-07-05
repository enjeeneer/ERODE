import energym
import numpy as np
from agents.erode import Agent as ERODE
from tqdm import tqdm
import os
import wandb
import argparse

envs = {
    'SeminarcenterThermostat-v0': 'DNK_MJ_Horsens1',
    'OfficesThermostat-v0': 'GRC_A_Athens',
    'MixedUseFanFCU-v0': 'GRC_A_Athens',
}

envs_timesteps = {
    'OfficesThermostat-v0': 15,
    'SeminarcenterThermostat-v0': 10,
    'MixedUseFanFCU-v0': 15,
}

### COMMAND LINE ARGS ###
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--learning_rate', default=0.003, type=float)
parser.add_argument('--horizon', default=20, type=int)
parser.add_argument('--popsize', default=25, type=int)
parser.add_argument('--epochs', default=25, type=int)
parser.add_argument('--exploration_mins', default=25, type=int)
args = parser.parse_args()

if __name__ == '__main__':
    # energym setup
    for key, value in envs.items():

        # setup logging dirs
        id = np.random.randint(low=0, high=1000)
        models_path = os.path.join(os.getcwd(), 'tmp', key, 'erode', str(id))
        os.mkdir(models_path, mode=0o666)

        years = 1
        wandb_config = dict(
            exploration_mins=540,
            env=key,
            id=id,
            alpha=0.003268,
            particles=30,
            latent_dim=200,
            GRU_dim=100,
            f_ode_dim=100,
            hist_length=1,
            hist_ode_dim=250,
            n_epochs=10,
            batch_size=32
        )

        ## WANDB SETUP ###
        wandb.init(
            project='erode',
            entity="enjeeneer",
            config=wandb_config,
            tags=['erode-testing'],
        )
        wandb.config.update(wandb_config)

        weather = value
        env_name = key
        simulation_days = 365

        minutes_per_step = envs_timesteps[key]
        N = int((60 * 24) / minutes_per_step)  # make model updates at end of each day
        steps_per_day = int((60 * 24) / minutes_per_step)
        sim_steps = steps_per_day * simulation_days
        steps_per_month = steps_per_day * 31

        for year in range(years):
            env = energym.make(env_name, weather=weather, simulation_days=simulation_days)
            agent = ERODE(env=env,
                          steps_per_day=steps_per_day,
                          env_name=env_name,
                          models_dir=models_path,
                          alpha=wandb_config['alpha'],
                          hist_length=wandb_config['hist_length'],
                          particles=wandb_config['particles'],
                          latent_dim=wandb_config['latent_dim'],
                          GRU_dim=wandb_config['GRU_dim'],
                          f_ode_dim=wandb_config['f_ode_dim'],
                          hist_ode_dim=wandb_config['hist_ode_dim'],
                          n_epochs=wandb_config['n_epochs'],
                          batch_size=wandb_config['batch_size']
                          )


            print('### RUNNING MODEL-BASED SCRIPT ###')
            learn_iters = 0
            daily_scores = []
            emissions = []
            temps = []

            prev_action = agent.normaliser.revert_actions(
                np.random.uniform(low=-1, high=1,
                                  size=agent.act_dim))  # dummy variable for action selection at first timestep

            observation = env.get_output()
            if agent.include_grid:
                observation = agent.add_c02(observation)
            score = 0
            for i in tqdm(range(sim_steps)):
                action_dict, model_input, obs = agent.plan(observation, env, prev_action)
                obs_next = env.step(action_dict)
                if agent.include_grid:
                    observation_ = agent.add_c02(obs_next)
                reward = agent.calculate_reward(obs_next)
                score += reward
                agent.n_steps += 1
                if agent.n_steps > agent.hist_length:
                    agent.memory.store(model_input=model_input,
                                       obs=obs,
                                       obs_next=obs_next,
                                       reward=reward
                                       )
                emissions.append(
                    obs_next[agent.energy_reward_key] * (envs_timesteps[key] / 60) / 1000
                    * (obs_next[agent.c02_reward_key] / 1000)
                )
                temps.append(obs_next['Z02_T'])

                min, hour, day, month = env.get_date()

                # exploration phase update
                if (agent.n_steps < steps_per_day) and (agent.n_steps
                                                        % (agent.batch_size + agent.hist_length) == 0):
                    agent.learn()
                    learn_iters += 1

                # normal update
                if agent.n_steps % agent.steps_per_day == 0:
                    agent.learn()
                    agent.save_models()
                    learn_iters += 1

                    daily_scores.append(score)
                    avg_score = np.mean(daily_scores[-3:])
                    _, _, day, month = env.get_date()

                    print('date:', day, '/', month, '--',
                          'today\'s score %.1f' % score, 'avg score %.1f' % avg_score,
                          'learning steps', learn_iters)

                    wandb.log({'mean_reward': avg_score})
                    wandb.log({'av_zone_temp': np.mean(temps[-steps_per_day:])})
                    wandb.log({'cum_emissions': sum(emissions)})

                    score = 0

                observation = observation_
                prev_action = action_dict

            env.close()
