from tool import policy_fn, MYPPO_PARAMS, get_data, get_env
from models import DRL_Agent

policy_dict = {'MlpPolicy': policy_fn}


class MyPPO(object):
    def __init__(
            self,
            policy,  # 'MlpPolicy' 多层感知机
            env,
            tensorboard_log,  # 日志位置 f"tensorboard_log/my_ppo"
            verbose,  # 日志显示，0:不输出日志信息 1:带进度条的输出日志信息
            policy_kwargs,  # None
            clip_param=0.2,  # 削波参数 epsilon
            entcoeff=0.01,  # 熵损失权重
            optim_epochs=4,  # 优化器的纪元数,向前和向后传播中所有批次的单次训练迭代
            optim_stepsize=0.001,  # 优化器的步长
            optim_batchsize=64,  # 优化器的批量大小
            gamma=0.99,  # 折扣因子
            lamda=0.95,  # 优势估计
            adam_epsilon=1e-05,  # Adam 优化器的 epsilon 值
            schedule='linear'  # 学习率更新的调度程序类型
    ):
        self.policy = policy
        self.env = env
        pass

    def learn(
            self,
            total_timesteps,
            eval_env,
            eval_freq,
            log_interval,
            tb_log_name,
            n_eval_episodes=1
    ):
        ob_space = self.env.observation_space  # agent观察到的状态空间
        ac_space = self.env.action_space  # agent的动作空间，正买入，负卖出，0观望
        newpi = policy_dict[self.policy]('newpi', ob_space, ac_space)  # 构建新策略的感知机网络
        oldpi = policy_dict[self.policy]('oldpi', ob_space, ac_space)  # 旧策略的网络
        print(newpi)
        print(oldpi)
        return newpi

    def predict(
            self,
            test_obs
    ):
        pass


if __name__ == '__main__':
    train_data, trade_data = get_data()
    env_train, env_trade = get_env(train_data, trade_data)
    agent = DRL_Agent(env=env_train)
    model = agent.get_model(
        'myppo',
        model_kwargs=MYPPO_PARAMS,
        verbose=0
    )
    model.learn(
        total_timesteps=200000,
        eval_env=env_trade,
        eval_freq=500,
        log_interval=1,
        tb_log_name='env_cashpenalty_highlr',
        n_eval_episodes=1
    )
