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
        sim_steps = steps_per_day * cfg.days

        cfg.steps_per_day = steps_per_day
        cfg.include_grid = True
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


        for year in range(years):
            env = energym.make(cfg.env_name, weather=cfg.weather, simulation_days=cfg.days)
            agent = ERODE(cfg=cfg, env=env, device=device)

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
                action_dict, action, model_input, obs = agent.plan(observation, env, prev_action)
                obs_next = env.step(action_dict)
                if agent.cfg.include_grid:
                    obs_next = agent.add_c02(obs_next)
                reward, cO2_reward, temp_reward = agent.one_step_reward(obs_next, main_loop=True, env=env)
                score += reward
                agent.n_steps += 1
                if agent.n_steps > agent.cfg.hist_length:
                    agent.memory.store(model_input=model_input,
                                       action=action,
                                       obs=obs,
                                       reward=reward
                                       )
                emissions.append(
                    obs_next[agent.cfg.energy_reward] * (cfg.mins_per_step / 60) / 1000
                    * (obs_next[agent.cfg.c02_reward] / 1000)
                )
                temps.append(obs_next['Z02_T'])

                min, hour, day, month = env.get_date()

                wandb.log({'train/reward': reward,
                           'train/cO2_reward': cO2_reward,
                           'train/temp_reward': temp_reward,
                            'train/zone2-sp': action_dict['Z02_T_Thermostat_sp'][0],
                            'train/hp-T1-sp': action_dict['Bd_T_AHU1_sp'][0],
                            'train/hp-fr1-sp': action_dict['Bd_Fl_AHU1_sp'][0],
                            'train/hp-T2-sp': action_dict['Bd_T_AHU2_sp'][0],
                            'train/hp-fr2-sp': action_dict['Bd_Fl_AHU2_sp'][0]})

                # exploration phase update
                if (agent.n_steps < steps_per_day) and (agent.n_steps
                                                        % (agent.cfg.batch_size + agent.cfg.hist_length) == 0):
                    model_loss, policy_loss, value_loss = agent.learn()
                    learn_iters += 1

                # normal update
                if agent.n_steps % cfg.steps_per_day == 0:
                    model_loss, policy_loss, value_loss = agent.learn()
                    # agent.save_models()
                    learn_iters += 1

                    daily_scores.append(score)
                    avg_score = np.mean(daily_scores[-3:])
                    _, _, day, month = env.get_date()

                    print('date:', day, '/', month, '--',
                          'today\'s score %.1f' % score, 'avg score %.1f' % avg_score,
                          'learning steps', learn_iters)

                    wandb.log({'train/mean_zone_temp': np.mean(temps[-cfg.steps_per_day:]),
                                'train/emissions': sum(emissions),
                                'train/reward:': score,
                                'train/model_loss': model_loss,
                               'train/policy_loss': policy_loss,
                               'train/value_loss': value_loss,
                           })

                    score = 0

                observation = obs_next
                prev_action = action_dict

            env.close()
