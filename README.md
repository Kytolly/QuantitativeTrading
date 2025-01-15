# 课程设计-强化学习量化交易策略

# Course Project - Reinforcement Learning for Quantitative Trading Strategy

## Project Overview

This project implements a deep reinforcement learning model for stock trading based on the PPO (Proximal Policy Optimization) algorithm. The model learns to make trading decisions (buy, sell, hold) by analyzing historical market data to maximize long-term returns.

## Environment Description 

The trading environment models transactions of SSE 50 Index constituent stocks using a continuous action space. The state space includes features like current holdings, remaining capital, etc. For detailed environment specifications, please refer to Stock_Trading/README.md.

## Performance Metrics

The model is evaluated on 6 key metrics compared against baseline models:


## Results

Our PPO-based model achieves comparable performance to the baseline models across all evaluation metrics, consistently outperforming the SSE 50 Index benchmark. The results demonstrate the effectiveness of using reinforcement learning for quantitative trading strategies.





