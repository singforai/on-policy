import torch

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


class K_Means_Clustering:
    def __init__(self, episode_length, num_clusters, device, len_share_obs):
        self.episode_length = episode_length
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
    
    def transform_share_obs(self, share_obs_set):
        episode_length, n_features = share_obs_set.shape
        step_share_obs_set = np.zeros((episode_length, n_features + 1))

        for idx, share_obs in enumerate(share_obs_set):
            step_share_obs_set[idx] = np.append(share_obs, idx/self.episode_length)
        
        return step_share_obs_set
    
    def distance_weight(self, point, closest_centroid_idx):
    
        dist_weight = np.exp(-1/self.num_clusters*np.sqrt(np.sum((self.centroids[closest_centroid_idx] - point)**2)))
        return dist_weight

    def decision_clusters(self, share_obs_set, rewards_set):
        self.clusters = [[] for _ in range(self.num_clusters)]
        for point_idx, point in enumerate(share_obs_set):
            closest_centroid_idx = np.argmin(np.sqrt(np.sum((point-self.centroids)**2, axis=1)))
            self.clusters[closest_centroid_idx].append(point_idx)
            
            # dist_weight = self.distance_weight(
            #     point = point,
            #     closest_centroid_idx = closest_centroid_idx,
            # )

            if rewards_set[point_idx] > self.centroids_max_rewards[closest_centroid_idx]:
                self.centroids_max_rewards[closest_centroid_idx] = rewards_set[point_idx]

    def cal_new_centroids(self, share_obs_set):
        for idx, cluster in enumerate(self.clusters):
            if len(cluster) == 0:
                pass
            else:
                self.centroids[idx] = np.mean(share_obs_set[cluster], axis=0)
    
    def training(self, episode, share_obs_set, rewards_set):

        #share_obs_set = self.transform_share_obs(share_obs_set = share_obs_set)
        
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
    
    def visualize_clusters(self, episode, share_obs_set):
        pca = PCA(n_components=2)
        pca_share_obs = pca.fit_transform(share_obs_set)
        pca_centroids = pca.fit_transform(self.centroids)

        colors = []
        for _ in range(self.num_clusters):
            color = "#{:06x}".format(np.random.randint(0, 0xFFFFFF))
            colors.append(color)
    
        for idx, cluster in enumerate(self.clusters):
            if len(cluster) > 0:
                points = pca_share_obs[cluster]
                plt.scatter(points[:, 0], points[:, 1], s = 100, color = colors[idx], label=f'Cluster-{idx}')
        
            plt.scatter(pca_centroids[idx, 0], pca_centroids[idx, 1], s = 200 ,color = colors[idx], marker='X', label=f'Centroids-{idx}')
            
        plt.title(f'K-Means Clusters with PCA - {episode}')
        plt.legend(scatterpoints=1, markerscale=2, fontsize=16)
        plt.show()



class Reward_Function:

    def __init__(self, num_clusters, device, episode_length, share_obs):
        self.num_clusters = num_clusters
        self.device = device
        self.len_share_obs = len(share_obs)

        self.clustering = K_Means_Clustering(
            episode_length = episode_length,
            num_clusters = self.num_clusters, 
            device = self.device, 
            len_share_obs = self.len_share_obs
            
        )