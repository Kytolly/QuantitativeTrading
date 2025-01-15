from typing import Any, Dict, Optional, Type, Union

import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance


class A2C(OnPolicyAlgorithm):
    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]],  # 要使用的策略，可以是字符串或ActorCriticPolicy类的类型
            env: Union[GymEnv, str],  # 环境，可以是Gym环境或环境的名称字符串
            learning_rate: Union[float, Schedule] = 7e-4,  # 学习率，可以是浮点数或调度(Schedule)对象
            n_steps: int = 5,  # 每次更新策略前收集的步数
            gamma: float = 0.99,  # 折扣因子，影响未来奖励的权重
            gae_lambda: float = 1.0,  # GAE(generalized advantage estimation)的lambda参数
            ent_coef: float = 0.0,  # 熵系数，用于加权熵损失，以促进探索
            vf_coef: float = 0.5,  # 值函数损失的系数
            max_grad_norm: float = 0.5,  # 梯度裁剪的最大范数
            rms_prop_eps: float = 1e-5,  # RMSProp优化器的epsilon参数，避免除零错误
            use_rms_prop: bool = True,  # 是否使用RMSProp优化器
            use_sde: bool = False,  # 是否在策略中使用状态依赖的探索(State-Dependent Exploration)
            sde_sample_freq: int = -1,  # 状态依赖探索的采样频率
            normalize_advantage: bool = False,  # 是否标准化优势估计值
            tensorboard_log: Optional[str] = None,  # TensorBoard日志目录
            create_eval_env: bool = False,  # 是否创建一个额外的评估环境
            policy_kwargs: Optional[Dict[str, Any]] = None,  # 传递给策略构造函数的额外参数
            verbose: int = 0,  # 打印详细信息的级别
            seed: Optional[int] = None,  # 随机种子
            device: Union[th.device, str] = "auto",  # 计算设备，'auto'、'cpu'或'cuda'
            _init_setup_model: bool = True  # 是否在初始化时设置模型
    ):
        # 调用父类构造函数进行初始化
        super(A2C, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )
        self.normalize_advantage = normalize_advantage  # 存储是否需要标准化优势
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            # 默认使用RMSProp优化器，如果未在策略参数中指定
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)

        if _init_setup_model:
            self._setup_model()  # 设置模型，建立计算图

    def train(self) -> None:
        """进行一步模型训练，使用从环境中采集的数据批量更新策略和值函数。"""
        self.policy.set_training_mode(True)  # 将策略模型设置为训练模式

        self._update_learning_rate(self.policy.optimizer)  # 更新优化器的学习率

        for rollout_data in self.rollout_buffer.get(batch_size=None):  # 从缓冲区获取数据
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.long().flatten()  # 对离散动作进行必要的处理

            values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations,
                                                                     actions)  # 评估动作，获取值函数、对数概率和熵
            values = values.flatten()

            advantages = rollout_data.advantages  # 从数据中获取优势估计
            if self.normalize_advantage:
                # 如果启用了优势标准化
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            policy_loss = -(advantages * log_prob).mean()  # 计算策略损失

            value_loss = F.mse_loss(rollout_data.returns, values)  # 计算值函数损失

            if entropy is None:
                entropy_loss = -th.mean(-log_prob)  # 当熵未直接提供时，使用对数概率估计熵损失
            else:
                entropy_loss = -th.mean(entropy)

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss  # 总损失包括策略损失、熵损失和值函数损失

            self.policy.optimizer.zero_grad()  # 清零梯度
            loss.backward()  # 反向传播计算梯度
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)  # 应用梯度裁剪
            self.policy.optimizer.step()  # 更新模型参数

        # 记录训练过程中的统计信息
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())