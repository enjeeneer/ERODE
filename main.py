import energym
import torch
import numpy as np
from agents.erode import Agent as ERODE
from tqdm import tqdm
from configs.parser import parse_cfg
import os
import wandb
import argparse

envs = ['MixedUseFanFCU-v0', 'SeminarcenterThermostat-v0', 'OfficesThermostat-v0']

if __name__ == '__main__':
    # energym setup
    for env in envs:

        agent_cfg_path = 'configs/erode.yaml'
        env_cfg_path = 'configs/envs.yaml'
        cfg = parse_cfg(agent_cfg_path, env_cfg_path, env_name=env)

        # setup logging dirs
        id = np.random.randint(low=0, high=1000)
        models_path = os.path.join(os.getcwd(), 'tmp', env, 'erode', str(id))
        os.mkdir(models_path, mode=0o666)

        years = 1

        ## WANDB SETUP ###
        wandb.init(
            project='erode',
            entity="enjeeneer",
            config=dict(cfg),
            tags=['erode-testing'],
        )
        wandb.config.update(dict(cfg))

        N = int((60 * 24) / cfg.mins_per_step)  # make model updates at end of each day
        steps_per_day = int((60 * 24) / cfg.mins_per_step)
        sim_steps = steps_per_day * cfg.simulation_days

        cfg.steps_per_day = steps_per_day
        cfg.include_grid = True
        cfg.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


        for year in range(years):
            env = energym.make(cfg.env_name, weather=cfg.weather, simulation_days=cfg.days)
            agent = ERODE(cfg=cfg, env=env)


            print('### RUNNING MODEL-BASED SCRIPT ###')
            learn_iters = 0
            daily_scores = []
            emissions = []
            temps = []

            prev_action = agent.normaliser.revert_actions(
                np.random.uniform(low=-1, high=1,
                                  size=agent.act_dim))  # dummy variable for action selection at first timestep

            observation = env.get_output()
            if agent.cfg.include_grid:
                observation = agent.add_c02(observation)
            score = 0
            for i in tqdm(range(sim_steps)):
                action_dict, model_input, obs = agent.plan(observation, env, prev_action)
                obs_next = env.step(action_dict)
                if agent.cfg.include_grid:
                    observation_ = agent.add_c02(obs_next)
                reward = agent.calculate_reward(obs_next)
                score += reward
                agent.n_steps += 1
                if agent.n_steps > agent.cfg.hist_length:
                    agent.memory.store(model_input=model_input,
                                       obs=obs,
                                       obs_next=obs_next,
                                       reward=reward
                                       )
                emissions.append(
                    obs_next[agent.cfg.energy_reward_key] * (cfg.mins_per_step / 60) / 1000
                    * (obs_next[agent.cfg.c02_reward_key] / 1000)
                )
                temps.append(obs_next['Z02_T'])

                min, hour, day, month = env.get_date()

                # exploration phase update
                if (agent.n_steps < steps_per_day) and (agent.n_steps
                                                        % (agent.cfg.batch_size + agent.cfg.hist_length) == 0):
                    agent.learn()
                    learn_iters += 1

                # normal update
                if agent.n_steps % cfg.steps_per_day == 0:
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
