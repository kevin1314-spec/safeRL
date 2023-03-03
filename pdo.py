import torch
from safepo.algos.policy_gradient import PG
from safepo.algos.lagrangian_base import Lagrangian
import safepo.common.mpi_tools as mpi_tools

class PDO(PG,Lagrangian):
    def __init__( self, algo='pdo', cost_limit=25., **kwargs):
        PG.__init__(self, 
            algo=algo, 
            use_cost_value_function=True,
            use_kl_early_stopping=True, 
            use_standardized_reward=True, 
            use_standardized_cost=True, 
            use_standardized_obs=True,
            use_reward_scaling=False,
            **kwargs)

        Lagrangian.__init__(self, 
            cost_limit=cost_limit,
            lagrangian_multiplier_init=0.001, 
            lambda_lr=0.035, 
            lambda_optimizer='Adam')


    def compute_loss_pi(self, data: dict):
        dist, _log_p = self.ac.pi(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])
        
        lamda_multiply = self.lambda_range_projection(self.lagrangian_multiplier).item()
        advantage = data['adv'] - lamda_multiply * data['cost_adv']
        loss_pi = -(ratio * advantage).mean()
        loss_pi -= self.entropy_coef * dist.entropy().mean()

        approx_kl = (0.5 * (dist.mean - data['act']) ** 2 / dist.stddev**2).mean().item()
        entropy = dist.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=entropy, ratio=ratio.mean().item())

        return loss_pi, pi_info

    def update(self):
        raw_data = self.buf.get()
        data = self.pre_process_data(raw_data)
        ep_costs = self.logger.get_stats('EpCosts')[0]
        self.update_lagrange_multiplier(ep_costs)
        self.update_policy_net(data=data)
        self.update_value_net(data=data)
        self.update_cost_net(data=data)
        self.update_running_statistics(raw_data)

    def algorithm_specific_logs(self):
        super().algorithm_specific_logs()
        self.logger.log_tabular('LagrangeMultiplier', self.lagrangian_multiplier.item())
