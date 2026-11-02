import os
import shutil
from Bio import SeqIO
import argparse

def parse_checkm(tsv_path, bins_dir, prefix):
    bins_info = []
    if not os.path.exists(tsv_path) or not os.path.exists(bins_dir):
        return bins_info
        
    with open(tsv_path, 'r') as f:
        lines = f.readlines()
        if len(lines) <= 1:
            return bins_info
            
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) >= 7:
                bin_id = parts[0]
                try:
                    comp = float(parts[5])
                    cont = float(parts[6])
                except ValueError:
                    continue
                
                if comp >= 90.0 and cont <= 5.0:
                    tier = 2
                    tier_score = comp - 2 * cont  
                elif comp >= 50.0 and cont <= 10.0:
                    tier = 1
                    tier_score = comp - 4 * cont
                else:
                    continue 
                    
                bin_path = os.path.join(bins_dir, f"{bin_id}.fa")
                if not os.path.exists(bin_path): 
                    bin_path = os.path.join(bins_dir, f"{bin_id}.fasta")
                if not os.path.exists(bin_path): 
                    bin_path = os.path.join(bins_dir, bin_id)
                    
                if not os.path.exists(bin_path):
                    continue
                    
                contigs_dict = {}
                try:
                    for rec in SeqIO.parse(bin_path, "fasta"):
                        contigs_dict[rec.id] = len(rec.seq)
                except Exception:
                    continue
                    
                total_bp = sum(contigs_dict.values())
                if total_bp > 0:
                    bins_info.append({
                        'new_id': f"{prefix}_{bin_id}",
                        'tier': tier,
                        'tier_score': tier_score,
                        'comp': comp,
                        'cont': cont,
                        'contigs': contigs_dict,
                        'total_bp': total_bp,
                        'path': bin_path
                    })
    return bins_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--checkm_v17', type=str, default=None)
    parser.add_argument('--dir_v17', type=str, default=None)
    parser.add_argument('--checkm_km', type=str, default=None)
    parser.add_argument('--dir_km', type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Gathering candidates...")
    all_bins = []
    if args.checkm_v17 and args.dir_v17:
        all_bins.extend(parse_checkm(args.checkm_v17, args.dir_v17, 'Leiden'))
    if args.checkm_km and args.dir_km:
        all_bins.extend(parse_checkm(args.checkm_km, args.dir_km, 'KMeans'))

    print(f"Total candidates: {len(all_bins)}")
    
    all_bins.sort(key=lambda x: (x['tier'], x['tier_score']), reverse=True)
    
    print("Performing BP-weighted greedy deduplication...")
    used_contigs = set()
    final_selected = []
    
    for b in all_bins:
        overlap_bp = sum(b['contigs'][c] for c in b['contigs'] if c in used_contigs)
        overlap_ratio = overlap_bp / b['total_bp']
        
        max_overlap = 0.50 if b['tier'] == 2 else 0.15
        
        if overlap_ratio <= max_overlap:
            final_selected.append(b)
            used_contigs.update(b['contigs'].keys())
            
    print(f"Selected {len(final_selected)} final bins.")
    
    for f in os.listdir(args.output_dir):
        if f.endswith(".fa") or f.endswith(".fasta"): 
            os.remove(os.path.join(args.output_dir, f))
        
    for i, b in enumerate(final_selected):
        new_name = f"ensemble_bin_{i}.fa"
        shutil.copy(b['path'], os.path.join(args.output_dir, new_name))
        
    print(f"Saved to {args.output_dir}")

if __name__ == "__main__":
    main()
