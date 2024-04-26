nohup python -u ./trader.py -m 'ddpg' >./trade_file/ddpg.log 2>&1 &
nohup python -u ./trader.py -m 'sac' >./trade_file/sac.log 2>&1 &
nohup python -u ./trader.py -m 'a2c' >./trade_file/a2c.log 2>&1 &
nohup python -u ./trader.py -m 'ppo' >./trade_file/ppo.log 2>&1 &
nohup python -u ./trader.py -m 'td3' >./trade_file/td3.log 2>&1 &