import numpy as np

class ErodeMemory:
    def __init__(self, batch_size, hist_length, horizon, traj_batches, obs_dim, act_dim, net_inp_dims):
        self.model_inputs = []
        self.actions = []
        self.obs_next = []
        self.rewards = []
        self.batch_size = batch_size
        self.horizon = horizon
        self.traj_batches = traj_batches
        self.hist_length = hist_length
        self.obs_dim = obs_dim # obs_dim + time_dim
        self.act_dim = act_dim
        self.net_inp_dims = net_inp_dims
        self.history = np.zeros(shape=(hist_length, self.net_inp_dims))

    def sample(self):
        """
        Generates batches of training data for dynamical model and value/policy functions
        :return inp_model: array of model inputs of shape (model_batches, batch_size, model_inp_dim)
        :return obs_model: array of observations from model inputs (model_batches, batch_size, obs_dim)
        :return inp_trajs: array of input trajectories of shape (traj_batches, horizon, model_inp_dim)
        :return obs_trajs: array of observation trajectories of shape (traj_batches, horizon, obs_dim)
        :return reward_trajs: array of reward trajectories of shape (traj_batches, horizon, 1)
        """
        datapoints = len(self.model_inputs)

        # samples trajectories
        inp_trajs = np.zeros(shape=(self.traj_batches, self.horizon, self.net_inp_dims))
        act_trajs = np.zeros(shape=(self.traj_batches, self.horizon, self.act_dim))
        obs_trajs = np.zeros(shape=(self.traj_batches, self.horizon, self.obs_dim))
        reward_trajs = np.zeros(shape=(self.traj_batches, self.horizon, 1))

        traj_batch_starts = np.random.choice(datapoints - self.horizon, size=self.traj_batches)
        for i, idx in enumerate(traj_batch_starts):
            traj_idxs = np.arange(idx, idx+self.horizon)

            # index into memory
            inps = self.model_inputs[traj_idxs]
            acts = self.actions[traj_idxs]
            obs = self.obs_next[traj_idxs]
            rewards = self.rewards[traj_idxs]

            # add to sample
            inp_trajs[i, :, :] = inps
            act_trajs[i, :, :] = acts
            obs_trajs[i, :, :] = obs
            reward_trajs[i, :, :] = rewards

        # samples input/output pairs in batches for model
        model_batches = round(datapoints / self.batch_size)
        inp_model = np.zeros(shape=(model_batches, self.batch_size, self.net_inp_dims))
        obs_model = np.zeros(shape=(model_batches, self.batch_size, self.obs_dim))

        model_batch_starts = np.arange(0, datapoints, self.batch_size)
        idxs = np.random.choice(datapoints, size=model_batches*self.batch_size, replace=True)
        batches = [idxs[i:i + self.batch_size] for i in model_batch_starts]

        for i, batch in batches:
            # index into memory
            inps = self.model_inputs[batch]
            outs = self.obs[batch]

            # add to sample
            inp_model[i, :, :] = inps
            obs_model[i, :, :] = outs

        return inp_model, obs_model, inp_trajs, act_trajs, obs_trajs, reward_trajs

    def store(self, model_input, obs, obs_next, reward):
        '''
        Stores model_input and correlating observation in memory
        :param state_action: normalised array of state_actions of shape (act_dim+obs_dim,)
        :param observation: normalised array of observations of shape (observation,)
        '''
        self.model_inputs.append(model_input)
        self.obs.append(obs)
        self.obs_next.append(obs_next)
        self.rewards.append(reward)

    def store_history(self, state_action):
        '''
        Stores state action in previous memory
        :param state_action: normalised array of state_actions of shape (act_dim+obs_dim+time_dim,)
        '''
        self.history[:self.hist_length-1] = self.history[1:]
        self.history[-1] = state_action

    def clear_memory(self):
        '''
        Clears working memory after each learning procedure.
        :return:
        '''
        self.model_inputs = []
        self.obs = []
        self.obs_next = []
        self.rewards = []

class ModelBasedMemory:
    def __init__(self, agent, batch_size, window_size, obs_dim, act_dim, particles=20, popsize=50):
        self.state_actions = []
        self.observations = []
        self.batch_size = batch_size
        self.window_size = window_size
        self.obs_dim = obs_dim # obs_dim + time_dim
        self.agent = agent
        self.state_act_dim = obs_dim + act_dim

        if self.agent == 'pets':
            self.previous = np.zeros(shape=(obs_dim * self.window_size,))
            self.previous_sampled = np.zeros(shape=(particles, popsize, obs_dim * self.window_size))

        if self.agent == 'mpc':
            self.previous = np.zeros(shape=(obs_dim * self.window_size,))
            self.previous_sampled = np.zeros(shape=(popsize, self.obs_dim * self.window_size))

    def generate_batches(self):
        '''
        Generates batches of training data for dynamical model from previously executed state actions and observations
        :return: array of all stored state actions of shape (datapoints, act_dim+obs_dim)
        :return: array of all stored observations of shape (datapoints, obs_dim)
        :return: array of batch indices of shape (datapoints/batch_size, batch_size)
        '''
        datapoints = len(self.state_actions)
        batch_start = np.arange(0, datapoints, self.batch_size)
        indices = np.random.choice(datapoints, size=datapoints, replace=True)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.state_actions), np.array(self.observations), batches

    def store_memory(self, state_action, observation):
        '''
        Stores state action and observation in memory
        :param state_action: normalised array of state_actions of shape (act_dim+obs_dim,)
        :param observation: normalised array of observations of shape (observation,)
        '''
        self.state_actions.append(state_action)
        self.observations.append(observation)

    def store_state_action(self, state_action):
        '''
        Stores state action in previous memory
        :param state_action: normalised array of state_actions of shape (act_dim+obs_dim+time_dim,)
        '''
        self.previous[:self.window_size-1] = self.previous[1:]
        self.previous[-1] = state_action

    def store_previous(self, state_tensor):
        '''
        Takes current state and stores in working memory for use in future action selection
        :param state_tensor:
        '''
        self.window[self.obs_dim:] = self.window[:(self.window_size - 1) * self.obs_dim]
        self.window[:self.obs_dim] = state_tensor

    def store_previous_samples(self, state_matrix):
        '''
        Stores states sampled using trajectory sampler (TS) in working memory for sampler propogation
        :param state_matrix: Tensor of states sampled using TS of shape (particles, popsize, obs_dim*hist_length)
        :return:
        '''
        if self.agent == 'pets':
            self.previous_sampled[:, :, :self.obs_dim] = self.previous_sampled[:, :, self.obs_dim:]
            self.previous_sampled[:, :, self.obs_dim:] = state_matrix

        if self.agent == 'mpc':
            self.previous_sampled[:, :self.obs_dim] = self.previous_sampled[:, self.obs_dim:]
            self.previous_sampled[:, self.obs_dim:] = state_matrix

    def clear_memory(self):
        '''
        Clears working memory after each learning procedure.
        :return:
        '''
        self.state_actions = []
        self.observations = []

class ModelFreeMemory:
    def __init__(self, batch_size, window_size, obs_dim):
        self.model_inputs = [] # states concat with previous hist_length * states
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.batch_size = batch_size
        self.obs_dim = obs_dim
        self.previous = np.zeros(shape=(obs_dim * window_size,))

    def generate_batches(self):
        '''
        Generates batches of training data for agent policy learning
        :return state array: array of previously seen states of shape (batch_size, obs_dim)
        :return action array: array of previously taken actions of shape (batch_size,act_dim)
        :return probs array: array of previous logs probs of actions given policy distribution of shape (batch_size,)
        :return vals array: array of previous critic values of shape (batch_size,)
        :return rewards array: array of previous rewards of shape (batch_size,)
        '''

        n_states = len(self.model_inputs)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.model_inputs), \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.vals, dtype=np.float16), \
               np.array(self.rewards), \
               batches

    def store_memory(self, model_input, action, prob, value):
        '''
        Stores state, action, log_prob of action, critic value of action in memory
        :return stored state: normalised state of shape ((obs_dim + time_dim) * hist_length,)
        :return stored action: normalised array of actions of shape (act_dim,)
        :return log_prob: log_prob of shape (1,)
        :return value: critic value of state (1,)
        '''
        self.model_inputs.append(model_input)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(value)

    def store_reward(self, reward):
        '''
        Stores reward
        :return stored reward: reward of shape (1,)
        '''
        self.rewards.append(reward)

    def clear_memory(self):
        '''
        Clears memory after batch_size-many timesteps
        '''
        self.model_inputs = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []

    def store_previous(self, state):
        '''
        Takes current state and stores in working memory for use in future action selection
        :param state:
        '''
        self.previous[:self.obs_dim] = self.previous[self.obs_dim:]
        self.previous[self.obs_dim:] = state

