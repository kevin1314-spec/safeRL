import torch
from safepo.algos.policy_gradient_copy import PG_copy
import safepo.common.mpi_tools as mpi_tools

class CRPO(PG_copy):
    def __init__( self, algo='crpo', cost_limit = [25.,1.] ,constraint_num = 2, eta = 0.5, clip = 0.2, **kwargs ):
        PG.__init__(self, 
            algo=algo, 
            use_cost_value_function=True,
            use_kl_early_stopping=True, 
            use_standardized_reward=True, 
            use_standardized_cost=True, 
            use_standardized_obs=True,
            use_reward_scaling=False,
            cost_limit = cost_limit,
            **kwargs)

        self.constraint_num = constraint_num
        self.eta = eta
        self.clip = clip

    def compute_loss_pi(self, data: dict):
        dist, _log_p = self.ac.pi(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])
        ratio_clip = torch.clamp(ratio, 1-self.clip, 1+self.clip)
        
        #分别创建cost列表和不满足约束条件的cost列表
        j = 0
        list = ['cost_adv_1','cost_adv_2']
        cost_not_satisfied_list = []
        
        #从cost里计算满足约束条件的cost总数
        for i in range(self.constraint_num):
            cost_num = list[i]
            if data[cost_num] <= self.cost_limit[i] + self.eta :
                j += 1
            else:
                cost_not_satisfied_list.append(data[cost_num])

        #判断是否所有约束条件都满足，若是，则policy只对reward的梯度方向进行更新
        if j == self.constraint_num :
           advantage = data['adv']
        #从不满足约束条件的cost里随机取一个，policy向着该cost的相反梯度方向更新
        else:
           cost_num = torch.randint(0, self.constraint_num - j)
           advantage = - cost_not_satisfied_list[cost_num]

        loss_pi = -(torch.min(ratio * advantage, ratio_clip * advantage)).mean()
        loss_pi -= self.entropy_coef * dist.entropy().mean()

        approx_kl = (0.5 * (dist.mean - data['act']) ** 2 / dist.stddev**2).mean().item()
        entropy = dist.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=entropy, ratio=ratio.mean().item())

        return loss_pi, pi_info

    def update(self):
        raw_data = self.buf.get()
        data = self.pre_process_data(raw_data)
        self.update_policy_net(data=data)
        self.update_value_net(data=data)

        self.update_cost_net_1(data=data)
        self.update_cost_net_2(data=data)

        self.update_running_statistics(raw_data)

    def algorithm_specific_logs(self):
        super().algorithm_specific_logs()
        self.logger.log_tabular('LagrangeMultiplier', self.lagrangian_multiplier.item())


