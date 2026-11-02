import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import hnswlib
import scipy.sparse as sp
import igraph as ig
import leidenalg
from sklearn.metrics.pairwise import cosine_similarity
from Bio import SeqIO
import numpy as np
from collections import defaultdict
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

def check_scg_conflict(nodes, markers_dict, namelist):
    seen_markers = set()
    for n in nodes:
        node_markers = markers_dict.get(namelist[n], [])
        for mk in node_markers:
            if mk in seen_markers:
                return True
            seen_markers.add(mk)
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--fasta_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--marker_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--min_bin_size', type=int, default=200000)
    parser.add_argument('--max_bin_size', type=int, default=5000000)
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    dims = [np.load(os.path.join(args.data_path, f'view_{v}.npy'), mmap_mode='r').shape[1] for v in range(2)]
    n_samples = np.load(os.path.join(args.data_path, 'view_0.npy'), mmap_mode='r').shape[0]
    
    with open(os.path.join(args.data_path, "contig_names.txt"), "r") as f: 
        namelist = [line.strip() for line in f]
    
    fasta_lens = {r.id: len(r.seq) for r in SeqIO.parse(args.fasta_path, "fasta")}
    contig_lens = np.array([fasta_lens.get(name, 0) for name in namelist])

    print("Extracting features...")
    model = FeatureExtractor(dims).to(DEVICE)
    state_dict = torch.load(args.model_path, map_location=DEVICE)
    model.load_state_dict({k: v for k, v in state_dict.items() if 'enc_' in k or 'attn_' in k}, strict=False)
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

    print("Parsing SCG markers...")
    markers_dict = defaultdict(list)
    with open(args.marker_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2: 
                markers_dict[parts[0]].append(parts[1])

    print("Building HNSW graph...")
    index = hnswlib.Index(space='cosine', dim=embeddings.shape[1])
    index.init_index(max_elements=n_samples, ef_construction=200, M=16)
    index.set_num_threads(16)
    index.add_items(embeddings)

    indices, distances = index.knn_query(embeddings, k=15)
    rows, cols, weights_list = [], [], []
    for i in tqdm(range(n_samples), desc="Building Graph"):
        for j_idx, dist in zip(indices[i], distances[i]):
            if i == j_idx: 
                continue
            sim = 1.0 - dist
            if sim >= 0.90:
                rows.append(i)
                cols.append(j_idx)
                weights_list.append(sim)

    adj = sp.coo_matrix((weights_list, (rows, cols)), shape=(n_samples, n_samples))
    adj_mnn = adj.minimum(adj.T).tocoo()
    global_sources, global_targets, global_weights = adj_mnn.row, adj_mnn.col, adj_mnn.data

    print("Recursive Leiden sub-clustering...")
    final_labels = -1 * np.ones(n_samples, dtype=int)
    curr_cid = 0
    task_queue = [(np.arange(n_samples), 0.5)] 
    
    extracted_bins = 0
    pbar = tqdm(total=n_samples, desc="Binning Progress")

    while task_queue:
        active_nodes, current_res = task_queue.pop(0)
        
        if current_res > 50.0 or len(active_nodes) < 5:
            continue
            
        mask = np.isin(global_sources, active_nodes) & np.isin(global_targets, active_nodes)
        sub_s, sub_t, sub_w = global_sources[mask], global_targets[mask], global_weights[mask]
        
        if len(sub_s) == 0:
            continue
            
        id_map = {old: new for new, old in enumerate(active_nodes)}
        edges = [(id_map[s], id_map[t]) for s, t in zip(sub_s, sub_t)]
        
        g = ig.Graph(len(active_nodes), edges)
        g.es['weight'] = sub_w
        
        part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, 
                                        weights='weight', resolution_parameter=current_res)
        membership = np.array(part.membership)
        
        for lbl in np.unique(membership):
            cluster_nodes = active_nodes[membership == lbl]
            bin_size = contig_lens[cluster_nodes].sum()
            
            if bin_size < args.min_bin_size:
                continue
                
            conflict = check_scg_conflict(cluster_nodes, markers_dict, namelist)
            oversize = bin_size > args.max_bin_size
            
            if conflict or oversize:
                task_queue.append((cluster_nodes, current_res * 1.5))
            else:
                final_labels[cluster_nodes] = curr_cid
                curr_cid += 1
                extracted_bins += 1
                pbar.update(len(cluster_nodes))
                
    pbar.close()
    print(f"Recursive cutting finished, {extracted_bins} base bins generated.")
    
    print("Performing length-weighted safe merging...")
    bin_nodes = defaultdict(list)
    for i in range(n_samples):
        if final_labels[i] != -1:
            bin_nodes[final_labels[i]].append(i)
            
    active_bins = list(bin_nodes.keys())
    bin_centroids = {}
    bin_markers = {}
    
    for b_id in active_bins:
        nodes = bin_nodes[b_id]
        lens = contig_lens[nodes]
        centroid = np.average(embeddings[nodes], axis=0, weights=lens)
        bin_centroids[b_id] = centroid / (np.linalg.norm(centroid) + 1e-10)
        m_set = set()
        for n in nodes: 
            m_set.update(markers_dict.get(namelist[n], []))
        bin_markers[b_id] = m_set

    merged_count = 0
    sim_threshold = 0.92
    
    while len(active_bins) >= 2:
        c_matrix = np.array([bin_centroids[b] for b in active_bins])
        sim_matrix = cosine_similarity(c_matrix)
        np.fill_diagonal(sim_matrix, -1)
        
        found_merge = False
        while not found_merge:
            max_idx = np.argmax(sim_matrix)
            i, j = np.unravel_index(max_idx, sim_matrix.shape)
            max_sim = sim_matrix[i, j]
            
            if max_sim < sim_threshold:
                break
                
            bin_i = active_bins[i]
            bin_j = active_bins[j]
            
            if bin_markers[bin_i].isdisjoint(bin_markers[bin_j]):
                found_merge = True
            else:
                sim_matrix[i, j] = -1
                sim_matrix[j, i] = -1
                
        if not found_merge:
            break
            
        bin_nodes[bin_i].extend(bin_nodes[bin_j])
        bin_markers[bin_i].update(bin_markers[bin_j])
        
        new_lens = contig_lens[bin_nodes[bin_i]]
        new_c = np.average(embeddings[bin_nodes[bin_i]], axis=0, weights=new_lens)
        bin_centroids[bin_i] = new_c / (np.linalg.norm(new_c) + 1e-10)
        
        if i > j:
            active_bins.pop(i)
            active_bins.pop(j)
        else:
            active_bins.pop(j)
            active_bins.pop(i)
            
        del bin_nodes[bin_j]
        del bin_centroids[bin_j]
        del bin_markers[bin_j]
        
        active_bins.append(bin_i)
        merged_count += 1
        
    print(f"Merging finished, {merged_count} merges occurred.")

    print("Writing results...")
    contig_to_cluster = {}
    for b_id, nodes in bin_nodes.items():
        for n in nodes:
            contig_to_cluster[namelist[n]] = b_id
            
    for f in os.listdir(args.output_dir):
        if f.endswith(".fa"): 
            os.remove(os.path.join(args.output_dir, f))
        
    cluster_records = defaultdict(list)
    for record in SeqIO.parse(args.fasta_path, "fasta"):
        if record.id in contig_to_cluster: 
            cluster_records[contig_to_cluster[record.id]].append(record)
            
    vc = 0
    for cid, recs in cluster_records.items():
        if sum(len(r.seq) for r in recs) >= args.min_bin_size:
            with open(os.path.join(args.output_dir, f"leiden_bin_{vc}.fa"), "w") as f:
                SeqIO.write(recs, f, "fasta")
            vc += 1
            
    print(f"Generated {vc} Leiden candidate bins in {args.output_dir}")

if __name__ == "__main__":
    main()
