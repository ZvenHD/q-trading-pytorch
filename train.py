from agent.agent import Agent
from agent.memory import Transition, ReplayMemory
from functions import *
import sys
import torch
# sys 為檔案塊 如同在 terminal 端 打 train.py 就會執行 但是 如果 train.py stock window episodes 去執行 那就是執行 train 這個程式碼 並且輸入參數依序為 stock windo episodes 
if len(sys.argv) != 4:
	print("Usage: python train.py [stock] [window] [episodes]")
	exit()

stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

# 初始化 代理人
agent = Agent(window_size)
# 獲取資料
data = getStockDataVec(stock_name)
l = len(data) - 1

for e in range(episode_count + 1):
	print("Episode " + str(e) + "/" + str(episode_count))
	state = getState(data, 0, window_size + 1)

	total_profit = 0
	agent.inventory = []

	for t in range(l):
		action = agent.act(state)

		# sit
		next_state = getState(data, t + 1, window_size + 1)
		reward = 0

		if action == 1: # buy
			agent.inventory.append(data[t])
			# print("Buy: " + formatPrice(data[t]))

		elif action == 2 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop(0)
			reward = max(data[t] - bought_price, 0)
			total_profit += data[t] - bought_price
			# print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

		done = True if t == l - 1 else False
		agent.memory.push(state, action, next_state, reward)
		state = next_state

		if done:
			print("--------------------------------")
			print("Total Profit: " + formatPrice(total_profit))
			print("--------------------------------")
		# 最後的最後 優化
		agent.optimize()

	if e % 10 == 0:
		agent.target_net.load_state_dict(agent.policy_net.state_dict())
		torch.save(agent.policy_net, "models/policy_model")
		torch.save(agent.target_net, "models/target_model")
