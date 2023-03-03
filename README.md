# safeRL
primal-dual 的变形 cpo 思想
![image](https://user-images.githubusercontent.com/60537667/222699710-cd841ba6-e419-4aaa-b7c4-023d1a65372e.png)
在约束条件后比 pdo 多了一个对策略的更新幅度的约束（trust-region),policy参数的距离度量采用 KL 散度，然后用转化为拉格朗日问题的对偶问题后，解该对偶问题，然后用解出的λ和 v 按公式对 policy 更新。对目标函数和约束函数进行近似处理（用泰勒公式进行一阶展开）
RCPO 的局限
① 当已经满足约束条件时，对拉格朗日乘子的优化依然会进行，即不断更新趋
于 0 以便摆脱约束，这会占用不必要的计算成本使得训练速度较慢
② 仅仅依赖拉格朗日乘子在 reward 和 cost 之间协调，因为每回合λ只更新一
次，这样导致该专注于提高 reward 和该专注于满足约束的切换间较迟钝
③ 总是超过约束的边界，或者过度低于约束边界，这导致较低的 reward 和不稳
定的约束条件的满足，因为最理想的情况应该是在约束边界的上限稳定下来然后
探索，这样可以增加探索空间。
④ 对拉格朗日乘子的学习率敏感
⑤ 很难满足多约束的情况
⑥ 在差别较大的任务上，如小车和机械手，lambda 的初始化差别较大，这就需
要额外的不断调参，否则λ初始化不同，对性能的影响较大。
RCPO code-level 细节
① 对策略网络的参数θ更新时使用 gamma 函数
② 对拉格朗日乘子λ进行投影裁剪，使其非负且小于最大值
CRPO 的局限性
我认为 CRPO 仅仅适用于约束条件较容易满足的任务，但是不适用于约束条件很
难满足的任务。比如机器人行走任务中限制腿的活动范围，这本身是到优化后期
才能逐步满足的,如果使用 CRPO，那它就将一直纠结于降低 cost，而完全不管提
升 reward，其实 reward 才是任务的主线。
CRPO 在多约束上的处理
1.将cost和reward完全独立开来，这样就不存在拉格朗日和解对偶问题，policy
只对 reward 和 cost 中的其中一个梯度方向更新，这样有助于在满足限制的条件
下不再在优化约束上浪费时间
2.当所有 cost 都满足约束条件时，policy 才向着 reward 的方向更新
3.当有多个 cost 不满足限制时，在其中随机取一个 cost，把 policy 向着该 cost
的相反梯度方向更新，直到把 policy 优化到所有 cost 都满足约束条件，再继续
进行 reward 方向的优化
4.对所有 cost 的限制加上一个常量η，有助于“软化”约束，论文中实验表明，
crpo 在η的较大的取值区间内有很强的适应鲁棒性
