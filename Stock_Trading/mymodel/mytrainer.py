import pandas as pd
from utils.env import StockLearningEnv
from utils import config
from utils.models import DRL_Agent

class Mytrainer(object):
    def __init__(self, model_name='kytolly', total_timesteps=500000):
        self.model_name = model_name
        self.total_timesteps = total_timesteps

    def train(self):
        train_data, trade_data = self.get_data()
        env_train, env_trade = self.get_env(train_data, trade_data)
        agent = DRL_Agent(env=env_train)
        model = agent.get_model(self.model_name,
                                model_kwargs=config.__dict__["{}_PARAMS".format(self.model_name.upper())],
                                verbose=0)
        model.learn(total_timesteps=self.total_timesteps,
                    eval_env=env_trade,
                    eval_freq=500,
                    log_interval=1,
                    tb_log_name='env_cashpenalty_highlr',
                    n_eval_episodes=1)

    def get_env(self, train_data, trade_data):
        e_train_gym = StockLearningEnv(df=train_data,
                                       random_start=True,
                                       **config.ENV_PARAMS)
        env_train, _ = e_train_gym.get_sb_env()

        e_trade_gym = StockLearningEnv(df=trade_data,
                                       random_start=False,
                                       **config.ENV_PARAMS)
        env_trade, _ = e_trade_gym.get_sb_env()

        return env_train, env_trade

    def get_data(self):
        train_data = pd.read_csv('train.csv')
        trade_data = pd.read_csv('trade.csv')
        return train_data, trade_data


if __name__ == '__main__':
    T = Mytrainer()
    T.train()
