import os
import torch
import math
import torch.nn.functional as F
import numpy as np
from components.networks import LatentODE, MLP, Q
from components.memory import ErodeMemory
from utils.utils import Normalize
from utils.torch_truncnorm import TruncatedNormal
from .base import Base


class Agent(Base):
    def __init__(self, cfg, env, device):

        # GENERAL PARAMS
        self.cfg = cfg
        self.env = env
        self.device = device
        self.normaliser = Normalize(self.env, cfg=self.cfg, device=device)
        self.obs_dim = len(self.normaliser.obs_space)
        self.act_dim = len(self.normaliser.act_space)
        self.obs_act_dim = self.obs_dim + self.cfg.time_dim + self.act_dim
        self.n_steps = 0
        self.model_path = os.path.join(cfg.models_dir, 'model.pth')
        self.exploration_steps = self.cfg.exploration_mins / self.cfg.mins_per_step

        # COMPONENTS
        self.memory = ErodeMemory(cfg=self.cfg, obs_dim=self.obs_dim,
                                  act_dim=self.act_dim, net_inp_dims=self.obs_act_dim)
        self.Q1, self.Q2 = Q(cfg, self.act_dim, device=device), Q(cfg, self.act_dim, device=device)
        self.Q1_target, self.Q2_target = Q(cfg, self.act_dim, device=device), Q(cfg, self.act_dim, device=device)
        self.pi = MLP(output_dims=self.act_dim, input_dims=cfg.latent_dim, chkpt_path=self.cfg.models_dir, device=device)
        self.model = LatentODE(cfg, obs_act_dim=self.obs_act_dim, obs_dim=self.obs_dim, device=device)
        self.model.to(device)
        self.optimiser = torch.optim.Adamax(self.model.parameters(), lr=cfg.alpha)
        self.kl_cnt = 0
        self.kl_coef = 1
        self.cem_init_mean = torch.zeros(size=(cfg.horizon, self.act_dim), dtype=torch.float, requires_grad=False).to(
            self.device)
        self.cem_init_var = torch.tile(torch.tensor(cfg.init_var, requires_grad=False),
                                   (cfg.horizon, self.act_dim)).to(self.device)

        # REWARD PARAMS
        temp_idx = []
        for temp in self.cfg.temp_reward:
            idx = self.normaliser.obs_space.index(temp)
            temp_idx.append(idx)
        self.temp_idx = temp_idx
        self.energy_idx = self.normaliser.obs_space.index(self.cfg.energy_reward)
        self.c02_idx = self.normaliser.obs_space.index(self.cfg.c02_reward)

        # EXPLORATION PARAMS
        lower = self.normaliser.action_lower_bound
        upper = self.normaliser.action_upper_bound
        self.expl_deltas = {}
        for key in lower.keys() & upper.keys():
            delta = (upper[key] - lower[key]) * self.cfg.expl_del
            self.expl_deltas[key] = delta

        # DISCOUNT PARAMS
        gamma_list = []
        disc = self.cfg.gamma
        for i in range(self.cfg.horizon):
            gamma_list.append(disc)
            disc *= self.cfg.gamma
        self.disc_vector = torch.tensor(gamma_list, dtype=torch.float32).unsqueeze(0)  # [1, horizon]
        self.disc_tensor = torch.tile(self.disc_vector, dims=(self.cfg.particles, self.cfg.popsize, 1))

        super(Agent, self).__init__(self.env, self.normaliser, self.memory, self.cfg, self.act_dim, self.normaliser.obs_space,
                                    self.n_steps, self.expl_deltas)

    @torch.no_grad()
    def traj_sampler(self, init_state, stoch_acts):
        """
        Performs trajectory sampling using latent odes
        :param init_state: array of more recent observation (state_dim,)
        :param act_seqs: numpy array of action sequences of shape (popsize, horizon, act_dim)
        :return trajs: array of trajectories of shape: (particles, N, horizon, state_dim)
        """
        pi = self.cfg.mix_coeff > 0  # include pi actions
        stoch_acts = stoch_acts.clone().cpu().detach().numpy()
        stoch_acts = np.tile(stoch_acts, (self.cfg.particles, 1, 1, 1))  # [part, cem_actions, horizon, act_dim]
        state_tile = np.tile(init_state, (self.cfg.particles, self.cfg.popsize, 1))  # [particles, popsize, state_dim]
        hist = np.tile(self.memory.history,
                       (self.cfg.particles, self.cfg.popsize, 1, 1))  # [part, pop, hist_length, state_act_dim]

        # initialise trajectory and pi_action arrays
        trajs = np.zeros(
            shape=(self.cfg.particles, self.cfg.popsize, self.cfg.horizon, self.obs_dim + self.cfg.time_dim))

        if pi:
            pi_acts = np.zeros(shape=(self.cfg.particles,
                                      math.ceil(self.cfg.popsize * self.cfg.mix_coeff),
                                      self.cfg.horizon,
                                      self.act_dim))
            assert pi_acts.shape[1] + stoch_acts.shape[1] == self.cfg.popsize

        for i in range(self.cfg.horizon):
            # store traj
            trajs[:, :, i, :] = state_tile

            # format current state-action
            stoch_actions = stoch_acts[:, :, i, :]  # [particles, cem_actions, act_dim]

            if pi:
                # sample some actions from policy
                histories = torch.tensor(hist[:, -math.ceil(self.cfg.popsize * self.cfg.mix_coeff):, :, :], dtype=torch.float).to(self.device)
                z = self.model.get_z0(histories, plan=True)  # [particles, pi_actions, latent_dim] -- each particle a different z0 sampled from dist from which pi selects action
                pi_actions = self.sample_pi(z)  # [particles, pi_actions, act_dim]

                # update pi_act memory to give back to CEM
                pi_acts[:, :, i, :] = pi_actions.cpu().detach().numpy()
                actions = np.concatenate((stoch_actions, pi_actions.cpu().detach().numpy()), axis=1)  # [particles, popsize, act_dim]

            else:
                actions = stoch_actions

            state_action = np.concatenate((actions, state_tile), axis=2)
            state_action = np.expand_dims(state_action, axis=2)  # [part, popsize, 1, net_inp_dims]
            input = np.concatenate((hist, state_action), axis=2)  # [part, popsize, hist_length + 1, net_inp_dims]
            assert input.shape == (self.cfg.particles, self.cfg.popsize, self.cfg.hist_length + 1,
                                   self.obs_act_dim)

            # predict
            model_input = torch.tensor(input, dtype=torch.float).to(self.device)
            pred_states = self.model.predict_next_state(history=model_input, train=False)
            assert pred_states.shape == (self.cfg.particles, self.cfg.popsize, self.obs_dim)

            # add time
            pred_states = self.normaliser.update_time(state_tensor=pred_states, init_date=self.TS_init_date,
                                                      init_time=self.TS_init_time, TS_step=i)

            # update history with current state-action
            hist[:, :, :-1, :] = hist[:, :, 1:, :]  # shift memory back one timestep
            hist[:, :, -1, :] = state_action.squeeze(axis=2)  # removes expanded dim from above

            state_tile = pred_states.cpu().detach().numpy()

        combined_acts = np.concatenate((stoch_acts, pi_acts), axis=1)  # [particles, popsize, horizon, act_dim]
        assert combined_acts.shape == (self.cfg.particles, self.cfg.popsize, self.cfg.horizon, self.act_dim)

        return trajs, combined_acts

    @torch.no_grad()
    def plan(self, observation, env, prev_action):

        obs = self.normaliser.outputs(outputs=observation, env=env, for_memory=False)
        obs_mem = self.normaliser.outputs(outputs=observation, env=env, for_memory=True)

        # Exploration Policy
        if self.n_steps <= self.exploration_steps:
            action_dict, action = self.explore(prev_action)
            state_action = np.concatenate((action, obs), axis=0)  # create state/action
            state_action = np.expand_dims(state_action, axis=0)  # [1, net_inp_dims]
            model_input = np.concatenate((self.memory.history, state_action),
                                         axis=0)  # create model input [hist_lenght+1, net_inp_dims]
            assert model_input.shape == (self.cfg.hist_length + 1, self.obs_act_dim)

        # Policy
        else:
            # store date/time for TS
            min, hour, day, month = env.get_date()
            self.TS_init_time = (min, hour)
            self.TS_init_date = (day, month)

            # CEM
            print('...planning...')
            mean, var, t = self.cem_init_mean, self.cem_init_var, 0
            # cem optimisation loop
            while (t < self.cfg.max_iters) and (torch.max(var) > self.cfg.epsilon):
                # sample stochastic actions
                stoch_samples = math.floor(self.cfg.popsize * (1 - self.cfg.mix_coeff))
                dist = TruncatedNormal(loc=mean, scale=var, a=-2, b=2)  # range [-2,2] to avoid discontinuity at [-1,1]
                stoch_acts = dist.sample(sample_shape=[stoch_samples,]).float()
                stoch_acts = torch.where(stoch_acts < torch.tensor([-1.0], device=self.device),
                                         torch.tensor([-1.0], device=self.device),
                                         stoch_acts)  # clip
                stoch_acts = torch.where(stoch_acts > torch.tensor([1.0], device=self.device),
                                         torch.tensor([1.0], device=self.device),
                                         stoch_acts)  # clip

                trajs, combined_acts = self.traj_sampler(obs, stoch_acts)

                exp_rewards = self.estimate_value(trajs, combined_acts).cpu().detach().numpy()  # returns pi_actions appended to CEM actions
                combined_acts = combined_acts[0, :, :, :] # particles are identical so take first particle to reduce dim
                elites = combined_acts[np.argsort(exp_rewards)][:int(self.cfg.elites * self.cfg.popsize)]

                elites = torch.tensor(elites).to(self.device)
                new_mean = torch.mean(elites, axis=0)
                new_var = torch.var(elites, axis=0)

                mean = self.cfg.kappa * mean + (1 - self.cfg.kappa) * new_mean
                var = self.cfg.kappa * var + (1 - self.cfg.kappa) * new_var

                t += 1

            opt_actions = mean
            action = opt_actions[0, :].cpu().detach().numpy()  # take only first action

            # variables for memory
            state_action = np.concatenate((action, obs), axis=0)  # create state/action
            state_action = np.expand_dims(state_action, axis=0)
            model_input = np.concatenate((self.memory.history, state_action),
                                         axis=0)  # create model input [hist_lenght+1, net_inp_dims]
            assert model_input.shape == (self.cfg.hist_length + 1, self.obs_act_dim)

            action_dict = self.normaliser.revert_actions(action)

        self.memory.store_history(state_action)

        return action_dict, action, model_input, obs_mem

    def learn(self):
        '''
        :param trajs: batched array of input trajectories of shape (n_batches, batch_size, state_action_dim)
        :param traj_batch: array of shape (batch_size, traj_length, state_action_dim)
        :return:
        '''

        print('...updating model parameters...')
        # generate batches
        inp_model, obs_model, inp_trajs, act_trajs, obs_trajs, reward_trajs = self.memory.sample()

        for epoch in range(self.cfg.epochs):
            if epoch % 5 == 0:
                print('learning epoch:', epoch)

            # model training
            self.kl_coef = 1 - 0.99 ** self.kl_cnt
            self.kl_cnt += 1
            for i in range(inp_model.shape[0]):
                input_batch = torch.tensor(inp_model[i, :, :, :], dtype=torch.float).to(self.device)
                obs_batch = torch.tensor(obs_model[i, :, :], dtype=torch.float).to(self.device)

                pred_state_mean, pred_state_std, z_dists = self.model.predict_next_state(input_batch, train=True)
                model_loss = self.model.loss(pred_state_mean, obs_batch, z_dists, self.kl_coef)
                self.optimiser.zero_grad()
                model_loss.backward()
                self.optimiser.step()

            # policy and value training
            with torch.no_grad():
                zs = torch.zeros(size=(inp_trajs.shape[0], self.cfg.horizon, self.cfg.latent_dim), device=self.device).float()
                zs_ = torch.zeros(size=(obs_trajs.shape[0], self.cfg.horizon, self.cfg.latent_dim), device=self.device).float()
                for i in range(self.cfg.horizon):
                    z, _ = self.model.get_z0(torch.tensor(inp_trajs[:, i, :, :], device=self.device), train=True)  # [traj_batches, horizon, 1]
                    z_, _ = self.model.get_z0(torch.tensor(obs_trajs[:, i, :, :], device=self.device), train=True) # [traj_batches, horizon, 1]
                    zs[:, i, :] = z
                    zs_[:, i, :] = z_

            print('zs:', zs.shape)
            pi_loss = self.update_pi(zs)
            value_loss = self.update_q(zs, zs_, act_trajs, reward_trajs)

        # update target networks
        if self.cfg.update_freq % self.n_steps == 0:
            self.update_target_net(self.Q1, self.Q1_target, tau=self.cfg.tau)
            self.update_target_net(self.Q2, self.Q2_target, tau=self.cfg.tau)

        self.memory.clear_memory()

        return model_loss, pi_loss, value_loss

    def update_pi(self, zs):
        """
        Updates policy given a trajectory of latent states to horizon H. It minimises the negative q-value i.e
        it maximises the value of the trajectory.
        :param zs: vector of latent states of shape (batch_size, horizon, latent_dim)
        """
        self.pi.optimizer.zero_grad(set_to_none=True)
        self.track_q_grad(False)

        # loss is a weighted sum of q values
        pi_loss = 0
        for t in range(self.cfg.horizon):
            action = self.sample_pi(zs[:, t, :])
            q = torch.min(*self.Q(zs[:, t, :], action))
            pi_loss += -q.mean() * (self.cfg.rho ** t) # minimise negative Q i.e. maximise value

        pi_loss.backward()
        self.pi.optimizer.step()
        self.track_q_grad(True)

        return pi_loss

    def update_q(self, zs, zs_, act_trajs, reward_trajs):
        """
        Updates value networks given batched trajectories of latent states
        """
        self.Q1.optimizer.zero_grad()
        self.Q2.optimizer.zero_grad()
        value_loss = 0
        for t in range(self.cfg.horizon):
            Q1, Q2 = self.Q(zs[:, t, :], act_trajs[:, t, :])
            z_, reward = zs_[:, t, :], reward_trajs[:, t, :]
            td_target = self.td_target(z_, reward)

            # losses
            rho = (self.cfg.rho ** t)
            value_loss += rho * (F.mse_loss(Q1, td_target) + F.mse_loss(Q2, td_target))

        value_loss.backward()
        self.Q1.optimizer.step()
        self.Q2.optimizer.step()

    def td_target(self, z_, reward):
        """
        Computes from a reward and the observation at the following timestep.
        """
        a_ = self.pi(z_, self.cfg.min_std)
        td_target = reward + self.cfg.gamama * torch.min(*self.Q(z_, a_, target=True))

        return td_target

    def update_target_net(self, model, model_target, tau):
        """
        Update slow-moving average of online network (target network) at rate tau.
        """
        with torch.no_grad():
            for p, p_target in zip(model.parameters(), model_target.parameters()):
                p_target.data.lerp_(p.data, tau)

    def Q(self, z, a, target=False):
        """
        Computes value of a state-action using both Q functions
        """
        x = torch.cat([z, a], dim=-1)

        if target:
            return self.Q1_target(x), self.Q2_target(x)

        else:
            return self.Q1(x), self.Q2(x)

    def sample_pi(self, z, std=0.05):
        """
        Samples action from learned policy pi
        :param z: latent state (latent_dim)
        :param std: standard deviation for applying noise to sample
        """
        print(z.shape)
        mu = self.pi(z)
        if std > 0:
            std = torch.ones_like(mu) * std
            dist = TruncatedNormal(loc=mu, scale=std, a=-2, b=2)  # range [-2,2] to avoid discontinuity at [-1,1]
            act = dist.sample()
            # ensure actions in range [-1,1]
            act = torch.where(act < torch.tensor([-1.0], device=self.device),
                              torch.tensor([-1.0], device=self.device), act)
            act = torch.where(act > torch.tensor([1.0], device=self.device),
                              torch.tensor([1.0], device=self.device), act)

            return act

        return mu

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
            term_states = torch.tensor(trajs[:, :, -(self.cfg.hist_length + 1):, :]).to(self.device)  # [particles, popsize, hist_length + 1, state_dim]
            term_actions = torch.tensor(actions[:, :, -(self.cfg.hist_length + 1):, :]).to(self.device)  # [particles, popsize, hist_length + 1, act_dim]
            term_trajs = torch.cat([term_actions, term_states],
                               dim=3).to(self.device)  # [particles, popsize, hist_length + 1, state_act_dim]
            term_vals = self.terminal_value(term_trajs)  # [particles, popsize]
            assert term_vals.shape == (self.cfg.particles, self.cfg.popsize, 1)

        # unnormalise
        trajs_revert = self.normaliser.model_predictions_to_tensor(trajs)

        # temps
        temp_elements = trajs_revert[:, :, :, self.temp_idx]
        temp_penalties = torch.minimum((self.cfg.low_temp_goal - temp_elements) ** 2,
                                   (self.cfg.high_temp_goal - temp_elements) ** 2) * -self.cfg.theta
        temp_rewards = torch.where((self.cfg.low_temp_goal >= temp_elements) | (self.cfg.high_temp_goal <= temp_elements), temp_penalties,
                               torch.tensor([0.0], dtype=torch.double))  # zero if in correct range, penalty otherwise
        temp = torch.sum(temp_rewards, axis=[3])  # [particles, popsize, horizon]

        # c02
        energy_elements = trajs_revert[:, :, :, self.energy_idx]
        c02_elements = trajs_revert[:, :, :, self.c02_idx]
        energy_elements_kwh = (energy_elements * (self.cfg.mins_per_step / 60)) / 1000
        c02 = (c02_elements * energy_elements_kwh) # [particles, popsize, horizon]

        # total
        total = c02 + temp
        total_disc = total * self.disc_tensor
        rewards = torch.sum(total_disc, dim=2).to(self.device)  # [particles, popsize]
        if self.pi:
            rewards = rewards + torch.squeeze(term_vals).to(self.device)

        exp_rewards = torch.mean(rewards, dim=0)  # [popsize]

        return exp_rewards

    @torch.no_grad()
    def terminal_value(self, term_trajs):
        """
        Takes the final state of a trajectory (requiring hist_length prev states too), estimates z0, and calculates the
        Q-values.
        :param term_trajs: traj up to final state s_H of shape (popsize, hist_length + 1, state_act_dim)
        :return qs: tensor of q values of shape (popsize,)
        """
        z0s = self.model.get_z0(term_trajs, plan=True)  # [any, hist_length + 1, latent_dim]
        acts = self.sample_pi(z0s, std=0)  # [any, hist_length + 1, act_dim]
        qs = torch.min(*self.Q(z0s, acts)) * (self.cfg.gamma ** (self.cfg.horizon + 1))

        return qs

    def save_models(self):
        '''
        Saves parameters of each model in ensemble to directory
        '''
        print('... saving models ...')
        torch.save(self.model.state_dict(), self.model_path)

    def load_models(self):
        '''
        Loads parameters of pre-trained models from directory
        '''
        print('... loading models ...')
        self.model.load_state_dict(torch.load(self.model_path))

    def track_q_grad(self, enable=True):
        """Utility function that enables/disables gradient tracking of Q-networks"""
        for m in [self.Q1, self.Q2]:
            for param in m.parameters():
                param.requires_grad = enable
