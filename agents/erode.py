import os
import numpy as np
import torch as T
from components.networks import LatentODE
from utils.data_utils import DataManipulator
# from utils.utils import Normalize
# import config.env_configs as env_configs
# from .base import Base

class Agent:
    def __init__(self, param):
        super(Agent, self).__init__()
        self.model = LatentODE(param)
        self.device = param['device']
        self.epochs = param['n_epochs']
        self.act_dim = param['act_dim']
        self.state_dim = param['state_dim']
        self.state_act_dim = param['state_action_dim']
        self.batch_size = param['batch_size']
        self.horizon = param['horizon']
        self.hist_length = param['hist_length']
        self.optimiser = T.optim.Adamax(self.model.parameters(), lr=param['lr'])
        self.data_helper = DataManipulator(param)
        self.kl_cnt = 0
        self.kl_coef = 1
        self.model_path = os.path.join(param['models_dir'], 'latent_model.pth')
        self.particles = param['particles']
        self.steps_per_day = param['steps_per_day']
        self.day = int(0)

    def plan(self, init_history, act_seqs):
        """
        :param init_state:
        :param init_history: array of history of shape (N, history_length + 1,  state_act_dim)
        :param act_seqs: numpy array of action sequences of shape (N, horizon, act_dim)
        :return trajs: array of trajectories of shape: (particles, N, horizon, state_dim)
        """

        particle_act_seqs = np.tile(act_seqs, (self.particles, 1, 1, 1)) # [part, N, horizon, act_dim]
        memory = np.tile(init_history, (self.particles, 1, 1, 1))  # [particles, N, hist_length + 1, state_act_dim]
        n_act_seqs = act_seqs.shape[0]
        trajs = np.zeros(
            shape=(self.particles, n_act_seqs, self.horizon + 1, self.state_dim)
        )
        trajs[:, :, 0, :] = memory[:, :, -1, self.act_dim: self.state_act_dim] # current state is last in memory

        for i in range(self.horizon):
            model_input = T.tensor(memory, dtype=T.float).to(self.device)
            pred_states = self.model.predict_next_state(history=model_input, train=False)
            assert pred_states.shape == (self.particles, n_act_seqs, self.state_dim)

            # update memory
            memory[:, :, :-1, :] = memory[:, :, 1:, :] # shift memory back one timestep
            memory[:, :, -1, :self.act_dim] = particle_act_seqs[:, :, i, :] # update action mem (act seqs starts at next state)
            memory[:, :, -1, self.act_dim:self.state_act_dim] = pred_states.cpu().detach().numpy() # update state mem

            # update trajectories
            trajs[:, :, i + 1, :] = pred_states.cpu().detach().numpy()

        return trajs

    def learn(self, trajs):
        '''
        :param trajs: batched array of input trajectories of shape (n_batches, batch_size, state_action_dim)
        :param traj_batch: array of shape (batch_size, traj_length, state_action_dim)
        :return:
        '''

        # generate batches
        day_trajs = trajs[(self.day * self.steps_per_day):(self.day + 1) * self.steps_per_day, :, :]
        train_batches = self.data_helper.generate_batches(trajs=day_trajs, batch_size=self.batch_size)
        train_torch = T.from_numpy(train_batches).to(self.device).to(T.float)
        train_input = train_torch[:, :, :-1, :self.state_act_dim]
        train_output = train_torch[:, :, -1, self.act_dim:self.state_act_dim]
        n_batches = train_input.shape[0]

        for epoch in range(self.epochs):
            avg_loss = 0.0
            self.kl_coef = 1 - 0.99 ** self.kl_cnt
            self.kl_cnt += 1
            for input_batch, output_batch in zip(train_input, train_output):
                pred_state_mean, pred_state_std, z_dists = self.model.predict_next_state(input_batch)
                loss = self.model.loss(pred_state_mean, output_batch, z_dists, self.kl_coef)

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()
                avg_loss += loss / n_batches

            #print('Epoch{0} | loss = {1:.5f}'.format(epoch, avg_loss))

        #self.save_model()

    def save_model(self):
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