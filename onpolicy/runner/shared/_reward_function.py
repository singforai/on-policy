import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import torch.nn.functional as F


class K_Means_Clustering:
    def __init__(self, num_clusters, device, len_share_obs):
        self.num_clusters = num_clusters
        self.device = device
        self.len_share_obs = len_share_obs
        self.centroids = [None for _ in range(self.num_clusters)]


    def init_centroids(self, share_obs_set):
        episode_length, n_features = share_obs_set.shape 

        first_centroids_idx = torch.randint(0, episode_length, (1,))
        self.centroids[0] = share_obs_set[first_centroids_idx]

        for idx in range(1, self.num_clusters):
            distances = []
            for point in share_obs_set:
                distances.append(np.sum((point - self.centroids[idx-1])**2, axis=0))
            
            while True:
                next_centroid_idx = np.random.choice(len(share_obs_set), p=distances/np.sum(distances))
                if not any(np.array_equal(share_obs_set[next_centroid_idx], centroid) for centroid in self.centroids[:idx]):
                    self.centroids[idx] = share_obs_set[next_centroid_idx]
                    break

        self.centroids_max_rewards = [float('-inf') for _ in range(self.num_clusters)]
    
    def decision_clusters(self, share_obs_set, rewards_set):
        self.clusters = [[] for _ in range(self.num_clusters)]
        for point_idx, point in enumerate(share_obs_set):
            closest_centroid_idx = np.argmin(np.sqrt(np.sum((point-self.centroids)**2, axis=1)))
            self.clusters[closest_centroid_idx].append(point_idx)

            if rewards_set[point_idx] > self.centroids_max_rewards[closest_centroid_idx]:
                self.centroids_max_rewards[closest_centroid_idx] = rewards_set[point_idx]

    def cal_new_centroids(self, share_obs_set):
        for idx, cluster in enumerate(self.clusters):
            if len(cluster) == 0:
                pass
            else:
                self.centroids[idx] = np.mean(share_obs_set[cluster], axis=0)
    
    def training(self, episode, share_obs_set, rewards_set):
        
        if episode == 0:
            self.init_centroids(share_obs_set)
        
        self.decision_clusters(
            share_obs_set = share_obs_set,
            rewards_set = rewards_set,
        )

        self.previous_centroids = self.centroids

        self.cal_new_centroids(share_obs_set = share_obs_set)



    def change_rewards(self, rewards, expected_max_rewards):
        if rewards[0][0][0] == expected_max_rewards:
            return rewards
        else:
            return rewards - expected_max_rewards

    def predict_cluster(self, share_obs, rewards):
        distances = []
        for centroid in self.centroids:
            distances.append(np.sum((share_obs - centroid)**2))
        closest_centroid_idx = torch.argmin(torch.tensor(distances))

        expected_max_rewards = self.centroids_max_rewards[closest_centroid_idx]

        shaped_rewards = self.change_rewards(
            rewards = rewards, 
            expected_max_rewards = expected_max_rewards
        )

        return shaped_rewards


class Reward_Function:

    def __init__(self, num_clusters, device, share_obs):
        self.num_clusters = num_clusters
        self.device = device
        self.len_share_obs = len(share_obs)

        self.clustering = K_Means_Clustering(
            num_clusters = self.num_clusters, 
            device = self.device, 
            len_share_obs = self.len_share_obs
        )

        

        


    



# class Reward_Function:
#     def __init__(self, share_obs, device, num_agent, episode_length, use_shaping_weight, shaping_weight_type):
#         """
#         극도로 라벨이 편중된 데이터로 회귀 모델을 학습시킬 때 현재 방법보다 개선시킬 필요가 있는가? 
        
#         """ 
#         self.num_agent = num_agent

#         self.episode_loss:float = 0.0
#         self.device = device

#         self.episode_length = episode_length

#         self.model = K_Means_Clustering(len_share_obs = len(share_obs) + 1).to(self.device)

#         self.optimizer = optim.Adam(self.model.parameters(), lr = 1e-8, weight_decay= 1e-4)
#         #self.criterion = nn.MSELoss()
#         self.criterion = nn.L1Loss()
#         self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lambda epoch: 0.99 ** epoch)

#         # early stopping 구현 필요 가능성

#     def shaping_weight_function(self, episode, episodes):
#         """
#         4/slope_weight + inflection_point = 1에 근사되기 시작하는 지점
#         inflection_point - 4/slope_weight > 0 이어야 시작점이 0에 아름답게 근사됨 
#         마지막 layer에서 activation을 어떻게 조작해야 할 지 모르겠음, 그냥 가중치만 사용하면 음수가 너무 자주 나오게 됨. 

#         환경에서 제공하는 가중치의 종류에 어떤 것이 있는지 정확하게 파악할 필요가 있음. 
#         """
#         point_one = int(0.04*episodes) 
#         inflection_point = int(point_one*0.8)
#         slope_weight = 4 / (point_one - inflection_point)
#         zero_point = episodes*0.4
#         inflection_point_2 = zero_point * 0.9
#         slope_weight_2 = 4 / (zero_point - inflection_point_2)

#         if self.shaping_weight_type == "sigmoid":
#                 shaping_weight = F.sigmoid(torch.tensor(slope_weight*(episode - inflection_point)))       
        
#         if self.shaping_weight_type == "sigmoid_flipsigmoid":
#             if episode < inflection_point_2-16/slope_weight_2:
#                 shaping_weight = F.sigmoid(torch.tensor(slope_weight*(episode - inflection_point)))
#             else:
#                 shaping_weight = F.sigmoid(torch.tensor(-slope_weight_2*(episode - inflection_point_2)))

#         return shaping_weight

#     def reward_shaping(self, episode, episodes, step, share_obs, rewards):
#         self.model.train()
#         step_share_obs = torch.cat(tensors=(torch.tensor([[step/self.episode_length]]), share_obs.unsqueeze(0)), dim = 1).to(self.device)


#         reward = torch.tensor(rewards).reshape(-1).to(torch.float).to(self.device)

#         if self.use_shaping_weight:
#             shaping_weight = self.shaping_weight_function(episode = episode, episodes = episodes)
            
        
#         train_pred_reward = self.model(share_obs = step_share_obs)

#         loss = self.criterion(train_pred_reward[0], reward[0])
#         loss.backward()

#         self.model.eval()
#         # if rewards[0][0][0] == 0.0:
#         #     rewards = [[[-0.01*shaping_weight.item()] for _ in range(self.num_agent)]] 

#         with torch.no_grad():
#             eval_pred_reward = self.model(share_obs=step_share_obs)
#             if self.use_shaping_weight:
#                 rewards = rewards - ((eval_pred_reward[0] - reward[0])*shaping_weight).cpu().detach().numpy()
#             else:
#                 rewards = rewards - (eval_pred_reward[0] - reward[0]).cpu().detach().numpy()
#         self.optimizer.step()
#         self.episode_loss += loss
#         if self.use_shaping_weight:
#             return rewards, shaping_weight
#         else: 
#             return rewards

#     def update(self):
#         #self.scheduler.step()
#         self.episode_loss:float = 0.0


