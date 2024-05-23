import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import torch.nn.functional as F

class Reward_Regression_MLP(nn.Module): 
    def __init__(self, len_share_obs):
        super().__init__()
        """
        regression 함수
        """

        hidden1 = int(len_share_obs*0.4)
        #dropout_prob = 0.2

        self.fc1 = nn.Linear(len_share_obs, hidden1)
        self.gn1 = nn.GroupNorm(1, hidden1)

        self.fc2 = nn.Linear(hidden1, 1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(p = dropout_prob)
    
    def forward(self, share_obs):
        activation1 = self.leaky_relu(self.gn1(self.fc1(share_obs)))
        pred_reward = self.fc2(activation1)
        return pred_reward


    
class Reward_Function:
    def __init__(self, share_obs, device, num_agent, episode_length, use_shaping_weight, shaping_weight_type):
        """
        극도로 라벨이 편중된 데이터로 회귀 모델을 학습시킬 때 현재 방법보다 개선시킬 필요가 있는가? 
        
        """ 
        self.num_agent = num_agent

        self.use_shaping_weight = use_shaping_weight
        self.shaping_weight_type = shaping_weight_type

        self.episode_loss:float = 0.0
        self.device = device

        self.episode_length = episode_length

        self.model = Reward_Regression_MLP(len_share_obs = len(share_obs) + 1).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr = 1e-8, weight_decay= 1e-4)
        #self.criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()
        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lambda epoch: 0.99 ** epoch)

        # early stopping 구현 필요 가능성

    def shaping_weight_function(self, episode, episodes):
        """
        4/slope_weight + inflection_point = 1에 근사되기 시작하는 지점
        inflection_point - 4/slope_weight > 0 이어야 시작점이 0에 아름답게 근사됨 
        마지막 layer에서 activation을 어떻게 조작해야 할 지 모르겠음, 그냥 가중치만 사용하면 음수가 너무 자주 나오게 됨. 

        환경에서 제공하는 가중치의 종류에 어떤 것이 있는지 정확하게 파악할 필요가 있음. 
        """
        point_one = int(0.04*episodes) 
        inflection_point = int(point_one*0.8)
        slope_weight = 4 / (point_one - inflection_point)
        zero_point = episodes*0.4
        inflection_point_2 = zero_point * 0.9
        slope_weight_2 = 4 / (zero_point - inflection_point_2)

        if self.shaping_weight_type == "sigmoid":
                shaping_weight = F.sigmoid(torch.tensor(slope_weight*(episode - inflection_point)))       
        
        if self.shaping_weight_type == "sigmoid_flipsigmoid":
            if episode < inflection_point_2-16/slope_weight_2:
                shaping_weight = F.sigmoid(torch.tensor(slope_weight*(episode - inflection_point)))
            else:
                shaping_weight = F.sigmoid(torch.tensor(-slope_weight_2*(episode - inflection_point_2)))

        return shaping_weight

    def reward_shaping(self, episode, episodes, step, share_obs, rewards):
        self.model.train()
        step_share_obs = torch.cat(tensors=(torch.tensor([[step/self.episode_length]]), share_obs.unsqueeze(0)), dim = 1).to(self.device)


        reward = torch.tensor(rewards).reshape(-1).to(torch.float).to(self.device)

        if self.use_shaping_weight:
            shaping_weight = self.shaping_weight_function(episode = episode, episodes = episodes)
            
        
        train_pred_reward = self.model(share_obs = step_share_obs)

        loss = self.criterion(train_pred_reward[0], reward[0])
        loss.backward()

        self.model.eval()
        # if rewards[0][0][0] == 0.0:
        #     rewards = [[[-0.01*shaping_weight.item()] for _ in range(self.num_agent)]] 

        with torch.no_grad():
            eval_pred_reward = self.model(share_obs=step_share_obs)
            if self.use_shaping_weight:
                rewards = rewards - ((eval_pred_reward[0] - reward[0])*shaping_weight).cpu().detach().numpy()
            else:
                rewards = rewards - (eval_pred_reward[0] - reward[0]).cpu().detach().numpy()
        self.optimizer.step()
        self.episode_loss += loss
        if self.use_shaping_weight:
            return rewards, shaping_weight
        else: 
            return rewards

    def update(self):
        #self.scheduler.step()
        self.episode_loss:float = 0.0


