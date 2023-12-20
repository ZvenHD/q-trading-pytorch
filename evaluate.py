import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from agent.agent import Agent
from functions import *

stock_name = '^HSI_2018'
window_size = 10

agent = Agent(window_size, True)
data = getStockDataVec(stock_name) # 將收盤價格形成一向量
l = len(data) - 1 
batch_size = 32
episode_count = 1000

closes = []
buys = []
sells = []

for e in range(episode_count):
	closes = []
	buys = []
	sells = []
	state = getState(data, 0, window_size + 1) # 狀態向量
	total_profit = 0
	agent.inventory = []
	capital = 100000
	for t in range(l):
		#action = agent.act(state)
		action = np.random.randint(0, 3) # action 採取隨機三動作
		closes.append(data[t]) 

		# sit
		next_state = getState(data, t + 1, window_size + 1) # 下一次狀態的向量
		reward = 0

		if action == 1: # buy
			if capital > data[t]: # 資產大於當前價格
				agent.inventory.append(data[t]) # 代理庫存+1
				buys.append(data[t]) # 買進紀錄
				sells.append(None)
				capital -= data[t] # 資產扣存
			else:
				buys.append(None)
				sells.append(None)

		elif action == 2: # sell 
			if len(agent.inventory) > 0: # 可以賣的
				bought_price = agent.inventory.pop(0) # 
				reward = max(data[t] - bought_price, 0)
				total_profit += data[t] - bought_price
				buys.append(None)
				sells.append(data[t])
				capital += data[t]
			else:
				buys.append(None)
				sells.append(None)
		elif action == 0: # 維持
			buys.append(None) #
			sells.append(None)

		done = True if t == l - 1 else False # 迴圈結束
		agent.memory.push(state, action, next_state, reward) # 
		state = next_state

		if done:
			print("--------------------------------")
			print(stock_name + " Total Profit: " + formatPrice(total_profit))
			print("--------------------------------")
