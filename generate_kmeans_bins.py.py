import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from Bio import SeqIO
import numpy as np
import argparse
from tqdm import tqdm

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

class FeatureExtractor(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.enc_kmer = ViewSpecificEncoder(dims[0])
        self.enc_cov = ViewSpecificEncoder(dims[1])
        self.attn_kmer = CrossViewAttention(128)
        self.attn_cov = CrossViewAttention(128)
        
    def forward(self, views):
        z_k, z_c = self.enc_kmer(views[0]), self.enc_cov(views[1])
        h_k = self.attn_kmer(z_k, z_c)
        h_c = self.attn_cov(z_c, z_k)
        return torch.cat([h_k, h_c], dim=1) 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--fasta_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--n_clusters', type=int, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Parsing data for K-Means (K={args.n_clusters})...")
    dims = [np.load(os.path.join(args.data_path, f'view_{v}.npy'), mmap_mode='r').shape[1] for v in range(2)]
    n_samples = np.load(os.path.join(args.data_path, 'view_0.npy'), mmap_mode='r').shape[0]
    
    with open(os.path.join(args.data_path, "contig_names.txt"), "r") as f: 
        namelist = [line.strip() for line in f]
    
    global_recs = {}
    fasta_lens = {}
    for rec in SeqIO.parse(args.fasta_path, "fasta"):
        global_recs[rec.id] = rec
        fasta_lens[rec.id] = len(rec.seq)

    lengths = np.array([fasta_lens.get(name, 0) for name in namelist])

    print("Extracting 256D features...")
    model = FeatureExtractor(dims).to(DEVICE)
    model.load_state_dict({k: v for k, v in torch.load(args.model_path, map_location=DEVICE).items() if 'enc_' in k or 'attn_' in k}, strict=False)
    model.eval()

    v0_mmap = np.load(os.path.join(args.data_path, "view_0.npy"))
    v1_mmap = np.load(os.path.join(args.data_path, "view_1.npy"))
    embs = []
    with torch.no_grad():
        for i in tqdm(range(0, n_samples, 4096), desc="Extracting"):
            v0 = torch.clamp(torch.nan_to_num(torch.from_numpy(v0_mmap[i:i+4096]).float()), -50., 50.).to(DEVICE)
            v1 = torch.clamp(torch.nan_to_num(torch.from_numpy(v1_mmap[i:i+4096]).float()), -50., 50.).to(DEVICE)
            embs.append(model([v0, v1]).cpu().numpy())
            
    embeddings = np.concatenate(embs, axis=0)
    embeddings /= (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

    print("Running length-weighted K-Means...")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings, sample_weight=lengths)

    print("Writing independent candidate bins...")
    bins = {i: [] for i in range(args.n_clusters)}
    for idx, label in enumerate(labels):
        bins[label].append(namelist[idx])

    for f in os.listdir(args.output_dir):
        if f.endswith(".fa"): 
            os.remove(os.path.join(args.output_dir, f))
        
    out_count = 0
    for b_id, contig_ids in bins.items():
        recs = [global_recs[c_id] for c_id in contig_ids if c_id in global_recs]
        if sum(len(r.seq) for r in recs) >= 200000:
            with open(os.path.join(args.output_dir, f"km_{args.n_clusters}_{b_id}.fa"), "w") as f:
                SeqIO.write(recs, f, "fasta")
            out_count += 1
            
    print(f"Generated {out_count} K-Means candidate bins.")

if __name__ == "__main__":
    main()
