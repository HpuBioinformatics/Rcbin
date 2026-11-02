import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import random
import argparse
from tqdm import tqdm

class MultiViewDataset(Dataset):
    def __init__(self, data_path):
        self.v0_data = np.load(os.path.join(data_path, 'view_0.npy'))
        self.v1_data = np.load(os.path.join(data_path, 'view_1.npy'))
        self.n_samples = self.v0_data.shape[0]
        
    def __len__(self): 
        return self.n_samples
    
    def __getitem__(self, idx):
        v0 = torch.nan_to_num(torch.from_numpy(self.v0_data[idx]).float(), nan=0.0)
        v1 = torch.nan_to_num(torch.from_numpy(self.v1_data[idx]).float(), nan=0.0)
        v0_clean = torch.clamp(v0, -50.0, 50.0)
        v1_clean = torch.clamp(v1, -50.0, 50.0)
        
        v0_a = v0_clean * (torch.rand_like(v0_clean) > 0.10).float()
        v0_b = v0_clean * (torch.rand_like(v0_clean) > 0.15).float()
        v0_c = v0_clean * (torch.rand_like(v0_clean) > 0.20).float()
        
        v1_a = v1_clean * torch.exp(torch.randn_like(v1_clean) * 0.05)
        v1_b = v1_clean * torch.exp(torch.randn_like(v1_clean) * 0.10)
        v1_c = v1_clean * torch.exp(torch.randn_like(v1_clean) * 0.15)
        
        return [v0_clean, v1_clean], [v0_a, v0_b, v0_c, v1_a, v1_b, v1_c], idx

class IndexedSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        global_idx = self.indices[idx]
        v_clean, v_aug, _ = self.dataset[global_idx]
        return v_clean, v_aug, global_idx

class ViewSpecificEncoder(nn.Module):
    def __init__(self, input_dim, feat_dim=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512, feat_dim)
        )
        
    def forward(self, x): 
        return self.enc(x)

class CrossViewAttention(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = dim ** 0.5
        
    def forward(self, target, context):
        attn = F.softmax(torch.matmul(self.q(target), self.k(context).transpose(-2, -1)) / self.scale, dim=-1)
        return target + torch.matmul(attn, self.v(context))

class QNetwork(nn.Module):
    def __init__(self, state_dim=3, action_dim=9):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, action_dim))
        
    def forward(self, state): 
        return self.net(state)

class MemoryEnhancedRefinement(nn.Module):
    def __init__(self, n_samples, feat_dim, n_clusters, beta=0.9):
        super().__init__()
        self.beta = beta
        self.memory_bank = torch.zeros(n_samples, feat_dim)
        self.historical_probs = torch.ones(n_samples, n_clusters) / n_clusters
        self.initialized = torch.zeros(n_samples)
        
    @torch.no_grad()
    def update_memory(self, features, indices):
        feats_cpu = features.detach().cpu().float()
        indices_cpu = indices.cpu()
        mask_uninit = (self.initialized[indices_cpu] == 0)
        mask_init = (self.initialized[indices_cpu] == 1)
        idx_uninit = indices_cpu[mask_uninit]
        idx_init = indices_cpu[mask_init]
        if len(idx_uninit) > 0:
            self.memory_bank[idx_uninit] = feats_cpu[mask_uninit]
            self.initialized[idx_uninit] = 1
        if len(idx_init) > 0:
            self.memory_bank[idx_init] = self.beta * self.memory_bank[idx_init] + (1 - self.beta) * feats_cpu[mask_init]

    def identify_uncertain_samples(self, prob_dist, eta_e, eta_c):
        entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-10), dim=1)
        sorted_probs, _ = torch.sort(prob_dist, descending=True)
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
        return (entropy > eta_e) | (margin < eta_c), entropy, margin

    def refine_assignments(self, curr_feat, curr_prob, mask, psi=0.7, tau=0.5, max_mem_size=32768):
        if mask.sum() == 0: 
            return curr_prob
        valid_idx = torch.where(self.initialized == 1)[0]
        if len(valid_idx) < 100: 
            return curr_prob
        if len(valid_idx) > max_mem_size:
            valid_idx = valid_idx[torch.randperm(len(valid_idx))[:max_mem_size]]
        
        mem_subset = F.normalize(self.memory_bank[valid_idx].to(curr_feat.device), dim=1)
        hist_subset = self.historical_probs[valid_idx].to(curr_feat.device)
        
        sim = torch.mm(F.normalize(curr_feat[mask], dim=1), mem_subset.t())
        weights = F.softmax(sim / tau, dim=1)
        refined = curr_prob.clone()
        refined[mask] = psi * curr_prob[mask] + (1 - psi) * torch.matmul(weights, hist_subset)
        return refined
        
    @torch.no_grad()
    def update_historical(self, probs, indices):
        self.historical_probs[indices.cpu()] = probs.detach().cpu().float()

class RcbinModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.n_clusters = config['n_clusters']
        
        self.enc_kmer = ViewSpecificEncoder(config['input_dims'][0])
        self.enc_cov = ViewSpecificEncoder(config['input_dims'][1])
        
        self.attn_kmer = CrossViewAttention(128)
        self.attn_cov = CrossViewAttention(128)
        
        self.proj_kmer = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128))
        self.proj_cov = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128))
        
        self.prob_mapper = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, self.n_clusters), nn.Softmax(dim=-1))
        
        self.dec_kmer = nn.Sequential(nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, config['input_dims'][0]))
        self.dec_cov = nn.Sequential(nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, config['input_dims'][1]))
        
        self.refiner = MemoryEnhancedRefinement(config['n_samples'], 256, self.n_clusters)
        self.q_net = QNetwork(state_dim=3).to(device)
        self.target_q = QNetwork(state_dim=3).to(device)
        self.target_q.load_state_dict(self.q_net.state_dict())
        
        self.action_space = [
            [0.6, 1.5, 0.3, 0.4], [0.7, 2.0, 0.5, 0.5], [0.8, 2.5, 0.7, 0.6], 
            [0.65, 1.8, 0.4, 0.45], [0.75, 2.2, 0.6, 0.55], [0.85, 2.8, 0.8, 0.7], 
            [0.55, 1.2, 0.2, 0.3], [0.7, 1.6, 0.45, 0.45], [0.8, 2.4, 0.55, 0.65]
        ]

    def forward(self, views_aug):
        z_k_a, z_k_b, z_k_c = self.enc_kmer(views_aug[0]), self.enc_kmer(views_aug[1]), self.enc_kmer(views_aug[2])
        z_c_a, z_c_b, z_c_c = self.enc_cov(views_aug[3]), self.enc_cov(views_aug[4]), self.enc_cov(views_aug[5])
        
        z_k_mean = (z_k_a + z_k_b + z_k_c) / 3.0
        z_c_mean = (z_c_a + z_c_b + z_c_c) / 3.0
        
        h_k = self.attn_kmer(z_k_mean, z_c_mean)
        h_c = self.attn_cov(z_c_mean, z_k_mean)
        
        z_concat = torch.cat([h_k, h_c], dim=1) 
        probs = self.prob_mapper(z_concat)
        recons = [self.dec_kmer(z_concat), self.dec_cov(z_concat)]
        
        projs = [
            self.proj_kmer(z_k_a), self.proj_kmer(z_k_b), self.proj_kmer(z_k_c), 
            self.proj_cov(z_c_a), self.proj_cov(z_c_b), self.proj_cov(z_c_c)
        ]
        return z_concat, probs, recons, projs

    def info_nce_loss(self, proj_list, tau=0.03):
        loss, pair_count = 0, 0
        n = proj_list[0].size(0)
        labels = torch.arange(n, device=self.device)
        proj_norm = [F.normalize(p, dim=1) for p in proj_list]
        for i in range(len(proj_list)):
            for j in range(i + 1, len(proj_list)):
                sim_ij = torch.mm(proj_norm[i], proj_norm[j].t()) / tau
                sim_ji = torch.mm(proj_norm[j], proj_norm[i].t()) / tau
                loss += (F.cross_entropy(sim_ij, labels) + F.cross_entropy(sim_ji, labels)) / 2
                pair_count += 1
        return loss / pair_count

    def compute_clustering_state(self, features, probs):
        labels = probs.argmax(1)
        K = probs.size(1)
        counts = torch.bincount(labels, minlength=K).float()
        valid_mask = counts > 0
        sum_feats = torch.zeros(K, features.size(1), device=self.device)
        sum_feats.index_add_(0, labels, features)
        centroids = torch.zeros_like(sum_feats)
        centroids[valid_mask] = sum_feats[valid_mask] / counts[valid_mask].unsqueeze(1)
        
        intra_dist = torch.norm(features - centroids[labels], dim=1).mean()
        
        valid_centroids = centroids[valid_mask]
        if len(valid_centroids) > 1:
            idx = torch.randperm(len(valid_centroids))[:500]
            dist_mat = torch.cdist(valid_centroids[idx], valid_centroids[idx])
            inter_dist = dist_mat.sum() / (len(idx) * (len(idx)-1))
        else: 
            inter_dist = torch.tensor(0.0, device=self.device)
            
        progress = valid_mask.float().mean()
        return torch.stack([intra_dist, inter_dist, progress])

    def refine_clustering(self, z_concat, probs, indices, epsilon=0.1):
        self.refiner.update_memory(z_concat, indices)
        state = self.compute_clustering_state(z_concat, probs)
        
        if np.random.random() < epsilon: 
            action_idx = np.random.randint(9)
        else:
            with torch.no_grad(): 
                action_idx = self.q_net(state.unsqueeze(0)).argmax().item()
                
        psi, tau, eta_c, eta_e = self.action_space[action_idx]
        uncertain_mask, _, _ = self.refiner.identify_uncertain_samples(probs, eta_e=eta_e, eta_c=eta_c)
        refined_probs = self.refiner.refine_assignments(z_concat, probs, uncertain_mask, psi=psi, tau=tau)
        self.refiner.update_historical(refined_probs, indices)
        next_state = self.compute_clustering_state(z_concat, refined_probs)
        
        reward = next_state[1] - 1.0 * next_state[0] + 1.0 * (~uncertain_mask).float().mean()
        return state, action_idx, reward, next_state, refined_probs

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.opt = optim.AdamW(model.parameters(), lr=1e-4)
        self.q_opt = optim.Adam(model.q_net.parameters(), lr=1e-3)
        self.config, self.device = config, device
        self.replay_buffer = []

    def validate(self):
        self.model.eval()
        v_loss = 0
        with torch.no_grad():
            for v_clean, v_aug, _ in self.val_loader:
                v_clean, v_aug = [v.to(self.device) for v in v_clean], [v.to(self.device) for v in v_aug]
                _, _, recons, projs = self.model(v_aug)
                nce = self.model.info_nce_loss(projs, tau=0.03)
                rec = F.mse_loss(recons[0], v_clean[0]) + F.mse_loss(recons[1], v_clean[1])
                v_loss += (nce + 0.5 * rec).item()
        return v_loss / len(self.val_loader)

    def run(self):
        best_loss, strikes = float('inf'), 0
        for epoch in range(1, self.config['epochs'] + 1):
            self.model.train()
            train_loss = 0
            epsilon = max(0.05, 0.5 * (1 - epoch / self.config['epochs']))
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            
            for v_clean, v_aug, idx in pbar:
                v_clean, v_aug = [v.to(self.device) for v in v_clean], [v.to(self.device) for v in v_aug]
                self.opt.zero_grad()
                
                z_concat, probs, recons, projs = self.model(v_aug)
                with torch.no_grad():
                    state, act_idx, reward, next_state, refined_probs = self.model.refine_clustering(z_concat, probs, idx, epsilon)
                
                self.replay_buffer.append((state, act_idx, reward, next_state))
                if len(self.replay_buffer) > 2000: self.replay_buffer.pop(0)
                
                loss_kl = F.kl_div((probs + 1e-10).log(), refined_probs, reduction='batchmean')
                loss_nce = self.model.info_nce_loss(projs, tau=0.03)
                loss_rec = F.mse_loss(recons[0], v_clean[0]) + F.mse_loss(recons[1], v_clean[1])
                
                main_loss = loss_nce + 0.5 * loss_rec + 0.5 * loss_kl
                main_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.opt.step()
                
                if len(self.replay_buffer) >= 64:
                    self.q_opt.zero_grad()
                    batch = random.sample(self.replay_buffer, 64)
                    s = torch.stack([b[0] for b in batch]).to(self.device)
                    a = torch.tensor([b[1] for b in batch]).to(self.device)
                    r = torch.tensor([b[2] for b in batch]).to(self.device).float()
                    s_next = torch.stack([b[3] for b in batch]).to(self.device)
                    curr_q = self.model.q_net(s).gather(1, a.unsqueeze(1)).squeeze()
                    with torch.no_grad(): 
                        target_q = r + 0.99 * self.model.target_q(s_next).max(1)[0]
                    rl_loss = F.smooth_l1_loss(curr_q, target_q)
                    rl_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.q_net.parameters(), max_norm=1.0)
                    self.q_opt.step()
                    
                train_loss += main_loss.item()
                pbar.set_postfix({'Loss': f'{main_loss.item():.3f}'})

            if epoch % 5 == 0: 
                self.model.target_q.load_state_dict(self.model.q_net.state_dict())
            
            val_loss = self.validate()
            print(f"Epoch {epoch} | Train: {train_loss/len(self.train_loader):.4f} | Val: {val_loss:.4f}")
            
            if val_loss < best_loss - 1e-4:
                best_loss = val_loss
                strikes = 0
                torch.save(self.model.state_dict(), self.config['save_path'])
                print("Best model updated.")
            else:
                strikes += 1
                if strikes >= self.config['patience']: 
                    print("Early stopping triggered.")
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--n_clusters', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dims = [np.load(os.path.join(args.data_path, f'view_{v}.npy')).shape[1] for v in range(2)]
    
    ds = MultiViewDataset(args.data_path)
    train_sz = int(0.9 * len(ds))
    
    indices = torch.randperm(len(ds)).tolist()
    train_indices = indices[:train_sz]
    val_indices = indices[train_sz:]
    
    train_ds = IndexedSubset(ds, train_indices)
    val_ds = IndexedSubset(ds, val_indices)
    
    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=4)
    
    config = {
        'n_clusters': args.n_clusters, 
        'input_dims': dims, 
        'n_samples': len(ds), 
        'epochs': args.epochs, 
        'patience': args.patience,
        'save_path': args.save_path
    }
    
    model = RcbinModel(config, device).to(device)
    Trainer(model, train_loader, val_loader, config, device).run()
