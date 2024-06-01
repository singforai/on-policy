import torch
import random 

import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm 

class AutoEncoder(nn.Module):
    def __init__(self, len_share_obs):
        super(AutoEncoder, self).__init__()

        self.data = []

        self.input_dim = len_share_obs
        self.hidden1_dim = len_share_obs // 2
        self.hidden2_dim = self.hidden1_dim // 2
        self.latent_dim = self.hidden2_dim // 2
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
    def __init__(self, num_clusters, device, len_share_obs, use_autoencoder, use_pre_sampling, cluster_update_interval):
        self.device = device

        self.num_clusters: int = num_clusters
        self.len_share_obs: int = len_share_obs
        self.cluster_update_interval: int  = cluster_update_interval

        self.use_pre_sampling: bool = use_pre_sampling
        self.use_autoencoder: bool = use_autoencoder


        self.centroids = [None for _ in range(self.num_clusters)]

        self.shareobs_storage = []
        self.rewards_storage = []

    def init_centroids(self):
        self.shareobs_storage = np.array(self.shareobs_storage)
        num_step_data, _ = self.shareobs_storage.shape

        first_centroids_idx = torch.randint(0, num_step_data, (1,))
        self.centroids[0] = self.shareobs_storage[first_centroids_idx]

        for idx in range(1, self.num_clusters):
            distances = []
            for point in self.shareobs_storage:
                distances.append(np.sum((point - self.centroids[idx-1])**2, axis=0))
            
            while True:
                next_centroid_idx = np.random.choice(len(self.shareobs_storage), p=distances/np.sum(distances))
                if not any(np.array_equal(self.shareobs_storage[next_centroid_idx], centroid) for centroid in self.centroids[:idx]):
                    self.centroids[idx] = self.shareobs_storage[next_centroid_idx]
                    break

        self.centroids_max_rewards = [float('-inf') for _ in range(self.num_clusters)]

        self.training()
    
    def decision_clusters(self):
        self.clusters = [[] for _ in range(self.num_clusters)]
        for point_idx, point in enumerate(self.shareobs_storage):
            closest_centroid_idx = np.argmin(np.sqrt(np.sum((point-self.centroids)**2, axis=1)))
            self.clusters[closest_centroid_idx].append(point_idx)

            if self.rewards_storage[point_idx] > self.centroids_max_rewards[closest_centroid_idx]:
                self.centroids_max_rewards[closest_centroid_idx] = self.rewards_storage[point_idx]

    def cal_new_centroids(self):
        for idx, cluster in enumerate(self.clusters):
            if len(cluster) == 0:
                pass
            else:
                self.centroids[idx] = np.mean(self.shareobs_storage[cluster], axis=0)
    
    def training(self):
        self.shareobs_storage = np.array(self.shareobs_storage)

        self.decision_clusters()
        self.previous_centroids = self.centroids
        self.cal_new_centroids()

        self.shareobs_storage = []
        self.rewards_storage = []

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

        if len(self.shareobs_storage) % self.cluster_update_interval == 0 and len(self.shareobs_storage) != 0:
            self.training()
        return shaped_rewards


class Reward_Function:

    def __init__(self, num_clusters, device, share_obs, use_autoencoder, use_pre_sampling, seed, cluster_update_interval):
        self.device = device
        self.backup_data = None

        self.seed: int = seed
        self.num_clusters: int = num_clusters
        self.len_share_obs: int = len(share_obs)

        self.use_autoencoder: bool = use_autoencoder
        self.use_pre_sampling: bool = use_pre_sampling

        self.clustering = K_Means_Clustering(
            num_clusters = self.num_clusters, 
            device = self.device, 
            len_share_obs = self.len_share_obs,
            use_autoencoder = self.use_autoencoder,
            use_pre_sampling = self.use_pre_sampling,
            cluster_update_interval = cluster_update_interval
        )

        if self.use_autoencoder:
            if self.use_pre_sampling:
                self.AutoEncoder = AutoEncoder(
                    len_share_obs = self.len_share_obs
                ).to(self.device)
            else: 
                raise ValueError("Auto encoder를 사용하기 위해서는 use_pre_sampling이 True가 되어야 합니다.")

    def train_ae(self, train_epoch, batch_size):

        self.train_epoch: int = train_epoch
        self.batch_size: int = batch_size

        self.backup_data = self.AutoEncoder.data

        random.shuffle(self.AutoEncoder.data)
        data = self.AutoEncoder.data
        training_data = data[:int(0.9*len(data))]
        validation_data = data[int(0.9*len(data)):]

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.AutoEncoder.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        
        best_val_error = float("inf")
        overfitting_stack = 0

        for epoch in (pbar:=tqdm(range(self.train_epoch), desc="AE_Train_Loss")):
            self.AutoEncoder.train()
            train_loss: float = 0.0
            for idx in range(0, len(training_data), self.batch_size): 
                end_idx = min(idx + self.batch_size, len(training_data))
                mini_batch = torch.tensor(np.array(training_data[idx : end_idx])).to(self.device)
                output_batch = self.AutoEncoder(mini_batch)
                loss = criterion(output_batch, mini_batch)
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

            with torch.no_grad():
                self.AutoEncoder.eval()
                val_loss = 0
                for idx in range(0, len(validation_data), self.batch_size):
                    end_idx = min(idx + self.batch_size, len(validation_data))
                    mini_batch = torch.tensor(np.array(validation_data[idx: end_idx]), dtype=torch.float32).to(self.device)
                    output_batch = self.AutoEncoder(mini_batch)
                    loss = criterion(output_batch, mini_batch)
                    val_loss += loss.item()
                if best_val_error > val_loss:
                    best_val_error = val_loss
                    best_autoencoder = self.AutoEncoder.state_dict()
                    overfitting_stack = 0
                else:
                    overfitting_stack += 1

                if overfitting_stack > 200 or epoch == (self.train_epoch-1):
                    import os.path as osp
                    torch.save(best_autoencoder, osp.join(osp.dirname(__file__),f'auto_encoder/best_autoencoder_{self.seed}.pth'))
                    print("[ encoder 생성 완료 ]")
                    break
            scheduler.step(train_loss)
            pbar.desc = f"Autoencoder val loss: {best_val_error}"#