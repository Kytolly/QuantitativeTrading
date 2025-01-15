import numpy as np
from kytolly_policy import KytollyPolicy
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
class Kytolly(OnPolicyAlgorithm):
    def __init__(self,
                 policy: Union[str, Type[KytollyPolicy]],  # 要使用的策略KytollyPolicy,或者用一个字符串表示
                 env: Union[GymEnv, str],  # 环境，可以是Gym环境或环境的名称字符串
                 tensorboard_log,  # TensorBoard日志目录，在config.py已经设置好了
                 verbose,  # 训练值为0，表示打印详细信息的级别
                 policy_kwargs,  # 传递给Policy构造函数的额外参数
                 # 以上 train.py中要用到的参数，重点关注
                 #################################################
                 # 以下参数缺省，到时候再调整
                 learning_rate: Union[float, Schedule] = 7e-4,  # 学习率
                 n_steps: int = 5,  # 每次更新策略前收集的步数
                 gamma: float = 0.99,  # 折扣因子，影响未来奖励的权重，收敛
                 gae_lambda: float = 1.0,  # GAE(generalized advantage estimation)的lambda参数
                 ent_coef: float = 0.0,  # 熵系数，用于加权**熵损失**，以促进探索
                 vf_coef: float = 0.5,  # 值函数损失的系数
                 max_grad_norm: float = 0.5,  # 梯度裁剪的最大范数
                 use_sde: bool = False,  # 是否在策略中使用状态依赖的探索(State-Dependent Exploration)
                 sde_sample_freq: int = -1,  # 状态依赖探索的采样频率
                 ):
        super(Kytolly, self).__init__(
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

    def predict(  # 运行trade.py才会用到的预测函数
            self,
            observation: np.ndarray,  # 当前的观察
            state: Optional[np.ndarray] = None,  # 在序列决策问题中，一般不为None，代表网络隐藏状态
            mask: Optional[np.ndarray] = None,  # 指示何时重置网络状态
            deterministic: bool = False,  # 动作输出方式，True表示输出一个当前概率最高的动作，False表示随机抽取动作，有利于探索
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self.policy.predict(observation, state, mask, deterministic)
    def train(self) -> None:
        pass

    def learn(
            self,
    ) -> None:
        return None


if '__name__' == '__main__':
    pass