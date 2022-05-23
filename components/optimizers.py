import numpy as np
import torch as T
from utils.torch_truncnorm import TruncatedNormal


class CEM():
    def __init__(self, act_dim, horizon, reward_estimator, max_iters=5,
                 popsize=400, elites=0.1, epsilon=0.001, alpha=0.1):
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.act_dim, self.horizon, self.max_iters = act_dim, horizon, max_iters
        self.reward_estimator = reward_estimator
        self.popsize, self.epsilon, self.alpha = popsize, epsilon, alpha
        self.num_elites = int(self.popsize * elites) # top 10%
        self.act_norm_low = T.tensor([-1], dtype=T.float, requires_grad=False).to(self.device)
        self.act_norm_high = T.tensor([1], dtype=T.float, requires_grad=False).to(self.device)

        if self.num_elites > self.popsize:
            raise ValueError("Number of elites must not be greater than the population size.")

    def optimal_action(self, state, init_mean, init_var):
        print('...planning...')
        mean, var, t = init_mean, init_var, 0

        # cem optimisation loop
        while (t < self.max_iters) and (T.max(var) > self.epsilon):

            dist = TruncatedNormal(loc=mean, scale=var, a=-2, b=2) # range [-2,2] to avoid discontinuity at [-1,1]
            act_seqs = dist.sample(sample_shape=[self.popsize,]) # output popsize x horizon x action_dims matrix

            # ensure actions in range [-1,1]
            act_seqs = T.where(act_seqs < self.act_norm_low, self.act_norm_low, act_seqs)
            act_seqs = T.where(act_seqs > self.act_norm_high, self.act_norm_high, act_seqs)

            exp_rewards = self.reward_estimator(state, act_seqs)
            elites = act_seqs[np.argsort(exp_rewards)][:self.num_elites]

            new_mean = T.mean(elites, axis=0)
            new_var = T.var(elites, axis=0)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var

            t += 1

        optimal_action_sequence = mean # select first action from optimal action sequence

        return optimal_action_sequence


