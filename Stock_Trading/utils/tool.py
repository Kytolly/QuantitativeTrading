import os

import pandas as pd
import mlp_policy
from env import StockLearningEnv
import config as cfg

MYPPO_PARAMS = {
    'clip_param': 0.2,
    'entcoeff': 0.01,
}


def get_data():
    train_data = pd.read_csv('../learn/data_file/train.csv')
    trade_data = pd.read_csv('../learn/data_file/trade.csv')
    print("数据读取成功!")
    return train_data, trade_data


def get_env(train_data, trade_data):
    """分别返回训练环境和交易环境"""
    e_train_gym = StockLearningEnv(df=train_data,
                                   random_start=True,
                                   **cfg.ENV_PARAMS)
    env_train, _ = e_train_gym.get_sb_env()

    e_trade_gym = StockLearningEnv(df=trade_data,
                                   random_start=False,
                                   **cfg.ENV_PARAMS)
    env_trade, _ = e_trade_gym.get_sb_env()
    return env_train, env_trade


def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space)
