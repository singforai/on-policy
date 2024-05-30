import random
import torch
import wandb
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

class AutoEncoder(nn.Module):
    def __init__(self, num_clusters, len_share_obs):
        super(AutoEncoder, self).__init__()

        self.data = []

        self.input_dim = len_share_obs
        self.hidden1_dim = int(len_share_obs / 2)
        self.hidden2_dim = int(self.hidden1_dim / 2)
        self.latent_dim = int(self.hidden2_dim / 2)
        self.dropout_prob = 0.1

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden1_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden1_dim, self.hidden2_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden2_dim, self.latent_dim),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden2_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden2_dim, self.hidden1_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden1_dim, self.input_dim),
            nn.LeakyReLU(),
        )
    def forward(self, x):
        latent_vector = self.encoder(x)
        pred_sh_obs = self.decoder(latent_vector)
        return pred_sh_obs


class K_Means_Clustering:
    def __init__(self, cluster_update_interval, num_clusters, device, len_share_obs, use_wandb):
        self.cluster_update_interval = cluster_update_interval
        self.num_clusters = num_clusters
        self.device = device
        self.len_share_obs = len_share_obs
        self.centroids = [None for _ in range(self.num_clusters)]
        self.use_wandb = use_wandb
        self.decay_rate = 0.999
        self.num_clusterings = [0 for _ in range(self.num_clusters)]
        self.centroid_rate = 0.9

    def init_centroids(self, share_obs_set):
        num_data, n_features = share_obs_set.shape 

        first_centroids_idx = torch.randint(0, num_data, (1,))
        self.centroids[0] = share_obs_set[first_centroids_idx].reshape(-1)

        for idx in range(1, self.num_clusters):
            distances = []
            for point in share_obs_set:
                distances.append(np.sum((point - self.centroids[idx-1])**2))

            while True:
                next_centroid_idx = np.random.choice(len(share_obs_set), p=distances/np.sum(distances))
                if not any(np.array_equal(share_obs_set[next_centroid_idx], centroid) for centroid in self.centroids[:idx]):
                    self.centroids[idx] = share_obs_set[next_centroid_idx]
                    break

        #self.new_centroids_rewards = [0 for _ in range(self.num_clusters)]
        self.new_centroids_rewards = [float("-inf") for _ in range(self.num_clusters)]
    
    def bias_correction(self, reward, closest_centroid_idx):
        self.num_clusterings[closest_centroid_idx] += 1

        self.new_centroids_rewards[closest_centroid_idx] = self.decay_rate*self.new_centroids_rewards[closest_centroid_idx] + (1 - self.decay_rate)*reward
        
        
        
    def decision_clusters(self, share_obs_set, rewards_set):
        self.clusters = [[] for _ in range(self.num_clusters)]
        for point_idx, point in enumerate(share_obs_set):
            closest_centroid_idx = np.argmin(np.sqrt(np.sum((point-self.centroids)**2, axis=1)))
            self.clusters[closest_centroid_idx].append(point_idx)
            
            # self.bias_correction(
            #     reward = rewards_set[point_idx],
            #     closest_centroid_idx = closest_centroid_idx
            # )

            if rewards_set[point_idx] > self.new_centroids_rewards[closest_centroid_idx]:
                self.new_centroids_rewards[closest_centroid_idx] = rewards_set[point_idx]

    def cal_new_centroids(self, share_obs_set):
        for idx, cluster in enumerate(self.clusters):
            if len(cluster) == 0:
                pass
            else:
                self.centroids[idx] = (1-self.centroid_rate)*(np.mean(share_obs_set[cluster], axis=0)) + self.centroid_rate*self.centroids[idx]

    def manifold(self, encoder, share_obs_set):
        out = encoder(torch.tensor(share_obs_set).to(self.device))
        return np.array(out.detach().cpu())
    
    def training(self, episode, share_obs_set, rewards_set, total_num_steps, encoder):

        share_obs_set = self.manifold(encoder, share_obs_set)
        
        if episode == self.cluster_update_interval:
            self.init_centroids(share_obs_set)
        
        self.decision_clusters(
            share_obs_set = share_obs_set,
            rewards_set = rewards_set,
        )

        self.previous_centroids = self.centroids

        self.cal_new_centroids(share_obs_set = share_obs_set)

        if self.use_wandb:
            wandb.log({"cluster_avg_value": np.mean(np.array(self.new_centroids_rewards))}, step=total_num_steps)
    

    def change_rewards(self, rewards, expected_rewards):

        if rewards[0][0][0] >= expected_rewards:
            return rewards 
        else:
            return rewards - expected_rewards

    def predict_cluster(self, share_obs, rewards):
        distances = []
        for centroid in self.centroids:
            distances.append(np.sum((share_obs - centroid)**2))
        closest_centroid_idx = torch.argmin(torch.tensor(distances))

        expected_rewards = self.new_centroids_rewards[closest_centroid_idx]

        shaped_rewards = self.change_rewards(
            rewards = rewards, 
            expected_rewards = expected_rewards
        )
        

        return shaped_rewards

class Reward_Function:

    def __init__(self, num_clusters, device, cluster_update_interval, share_obs, use_wandb):
        self.num_clusters = num_clusters
        self.device = device
        self.len_share_obs = len(share_obs)


        self.clustering = K_Means_Clustering(
            cluster_update_interval = cluster_update_interval,
            num_clusters = self.num_clusters, 
            device = self.device, 
            len_share_obs = self.len_share_obs,
            use_wandb = use_wandb
            
        )

class TrainingEncoder:
    def __init__(self, num_clusters, device, share_obs):

        self.autoencoder = AutoEncoder(
            num_clusters = num_clusters,
            len_share_obs = len(share_obs),
        ).to(device)

    def train_encoder(self, device, seed):
        random.shuffle(self.autoencoder.data)
        data = self.autoencoder.data
        training_data = data[:int(0.9*len(data))]
        validation_data = data[int(0.9*len(data)):]

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.01)

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 0.99 ** epoch)

        self.batch_size = 32
        best_val_error = float("inf")
        overfitting_stack = 0

        self.train_epoch = 10000

        for epoch in tqdm(range(self.train_epoch), desc="Train Encoder..."):
            self.autoencoder.train()
            for idx in range(0, len(training_data), self.batch_size): 
                end_idx = min(idx + self.batch_size, len(training_data))
                mini_batch = torch.tensor(np.array(training_data[idx : end_idx])).to(device)
                output_batch = self.autoencoder(mini_batch)
                loss = criterion(output_batch, mini_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                self.autoencoder.eval()
                val_loss = 0
                for idx in range(0, len(validation_data), self.batch_size):
                    end_idx = min(idx + self.batch_size, len(validation_data))
                    mini_batch = torch.tensor(np.array(validation_data[idx: end_idx]), dtype=torch.float32).to(device)
                    output_batch = self.autoencoder(mini_batch)
                    loss = criterion(output_batch, mini_batch)
                    val_loss += loss.item()
                if best_val_error > val_loss:
                    best_val_error = val_loss
                    best_autoencoder = self.autoencoder.state_dict()
                    overfitting_stack = 0
                else:
                    overfitting_stack += 1

                if overfitting_stack > 200 or epoch == (self.train_epoch-1):
                    import os.path as osp
                    torch.save(best_autoencoder, osp.join(osp.dirname(__file__),f'auto_encoder/best_autoencoder_{seed}.pth'))
                    print("[ encoder 생성 완료 ]")
                    break
            scheduler.step()
        