import copy
import wandb
import torch

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

class K_Means_plus2_Clustering:
    def __init__(self, device, use_wandb, num_agents, num_clusters, clutering_max_iter, memory_size, decay_rate):
        
        self.device = device
        
        self.use_wandb: bool = use_wandb
        self.num_agents = num_agents

        self.num_clusters: int = num_clusters
        self.memory_size: int = memory_size
        self.clutering_max_iter: int = clutering_max_iter
        self.nth_clustering: int = 0

        self.decay_rate: float = decay_rate
        #self.standard_factor: float = 0.5

        self.feature_memory = []
        self.reward_memory = []

        self.centroids = [None for _ in range(self.num_clusters)]

        self.centroids_max_R = np.array([float("-inf") for _ in range(self.num_clusters)])

    
    def init_centroids(self):
        centorid0_idx = torch.randint(0, self.memory_size, (1,))
        self.centroids[0] = self.feature_memory[centorid0_idx]

        for idx in range(1, self.num_clusters):
            distances = []
            for point in self.feature_memory:
                distances.append(torch.sum(input = ((point - self.centroids[idx-1])**2), dim = 0, keepdim=False))
            while True:
                next_centroid_idx = np.random.choice(len(self.feature_memory), p = distances/np.sum(distances))
                if not any(torch.equal(self.feature_memory[next_centroid_idx],centroid) for centroid in self.centroids[:idx]):
                    self.centroids[idx] = self.feature_memory[next_centroid_idx]
                    break

    def decision_clusters(self):
        self.clusters = [[] for _ in range(self.num_clusters)]
        for point_idx, point in enumerate(self.feature_memory): 
            closest_centroid_idx = np.argmin(np.sum((np.expand_dims(point, axis=0)-np.array(self.centroids))**2, axis=1))
            self.clusters[closest_centroid_idx].append(point_idx)

            if self.reward_memory[point_idx] > self.centroids_max_R[closest_centroid_idx]:
                self.centroids_max_R[closest_centroid_idx] = self.reward_memory[point_idx]
            

    def cal_new_centroids(self):
        for idx, cluster in enumerate(self.clusters):
            if len(cluster) == 0:
                pass
            else:
                self.centroids[idx] = np.mean(np.array(self.feature_memory)[cluster], axis=0)

    def predict_cluster(self, input_feature, rewards):
        distances = []
        for centroid in self.centroids:
            distances.append(torch.sum((input_feature - torch.tensor(centroid))**2))
        closest_centroid_idx = torch.argmin(torch.tensor(distances))
        print(closest_centroid_idx)

        if rewards[0][0][0] == self.centroids_max_R[closest_centroid_idx]:
            pass
        else:
            rewards = rewards - self.centroids_max_R[closest_centroid_idx]

        return rewards
    

    def checking(self, total_num_steps, rewards):
        if self.nth_clustering >= 10:
            shaped_rewards = self.predict_cluster(
                input_feature = self.feature_memory[-1],
                rewards = rewards
            )
        else:
            shaped_rewards = [[[self.reward_memory[-1]] for _ in range(self.num_agents)]]

        if len(self.feature_memory) >= self.memory_size:
            if self.nth_clustering == 0:
                self.init_centroids()

            for _ in range(self.clutering_max_iter):

                self.decision_clusters()
                self.previous_centroids = copy.deepcopy(self.centroids)
                self.cal_new_centroids()
                
                diff = np.sqrt(np.sum((np.array(self.previous_centroids) - np.array(self.centroids))**2))
                if diff < 0.1:
                    break

            # self.change_centroid_minmax_R(nth_clustering = self.nth_clustering)
            # self.cal_reward_shaping()
            
            self.feature_memory.clear()
            self.reward_memory.clear()
            self.nth_clustering += 1
            if self.use_wandb:
                wandb.log({"average_shape_value": np.mean(np.array(self.centroids_max_R))}, step=total_num_steps)
        
        return np.array(shaped_rewards)


