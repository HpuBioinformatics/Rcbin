import argparse
import itertools
import os
from Bio import SeqIO
import numpy as np
from sklearn.preprocessing import StandardScaler


def filter_and_deduplicate(input_fasta, output_fasta, min_len):
    records = []
    seen_seqs = set()

    for record in SeqIO.parse(input_fasta, "fasta"):
        if len(record.seq) >= min_len:
            seq_str = str(record.seq).upper()
            if seq_str not in seen_seqs:
                seen_seqs.add(seq_str)
                records.append(record)

    with open(output_fasta, "w") as f:
        SeqIO.write(records, f, "fasta")

    valid_names = [r.id for r in records]
    return valid_names


def get_rc(kmer):
    comp = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
    return "".join(comp.get(b, "N") for b in reversed(kmer))


def extract_tnf(fasta_path, valid_names):
    if not valid_names:
        return np.zeros((0, 136), dtype=np.float32)

    bases = ["A", "C", "G", "T"]
    kmers = ["".join(p) for p in itertools.product(bases, repeat=4)]

    mapping = {}
    seen = set()
    idx = 0
    for k in kmers:
        if k not in seen:
            mapping[k] = idx
            mapping[get_rc(k)] = idx
            seen.add(k)
            seen.add(get_rc(k))
            idx += 1

    name2idx = {name: i for i, name in enumerate(valid_names)}
    features = np.zeros((len(valid_names), 136), dtype=np.float32)

    for record in SeqIO.parse(fasta_path, "fasta"):
        if record.id not in name2idx:
            continue

        seq = str(record.seq).upper()
        counts = np.zeros(136, dtype=np.float32)

        for i in range(len(seq) - 3):
            kmer = seq[i : i + 4]
            if kmer in mapping:
                counts[mapping[kmer]] += 1

        features[name2idx[record.id]] = counts

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = np.divide(
        features,
        norms,
        out=np.zeros_like(features),
        where=norms > 0,
    )

    return features


def extract_abundance(depth_file, valid_names):
    if not valid_names:
        return np.zeros((0, 0), dtype=np.float32)

    name2idx = {name: i for i, name in enumerate(valid_names)}

    with open(depth_file, "r") as f:
        header = f.readline().strip().split("\t")
        depth_cols = list(range(3, len(header)))

    features = np.zeros((len(valid_names), len(depth_cols)), dtype=np.float32)

    with open(depth_file, "r") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue

            cname = parts[0]
            if cname not in name2idx:
                continue

            for j, col_idx in enumerate(depth_cols):
                if col_idx < len(parts):
                    try:
                        features[name2idx[cname], j] = float(parts[col_idx])
                    except ValueError:
                        features[name2idx[cname], j] = 0.0

    features = np.log1p(features)

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=str, required=True)
    parser.add_argument("--depth", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--min_len", type=int, default=1500)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    filtered_fasta = os.path.join(args.out_dir, "filtered_contigs.fasta")
    valid_names = filter_and_deduplicate(args.fasta, filtered_fasta, args.min_len)

    if not valid_names:
        print("[Warning] No contigs passed the length filter! Please check --min_len.")
        return

    with open(os.path.join(args.out_dir, "contig_names.txt"), "w") as f:
        for name in valid_names:
            f.write(f"{name}\n")

    view_0 = extract_tnf(filtered_fasta, valid_names)
    np.save(os.path.join(args.out_dir, "view_0.npy"), view_0)

    view_1 = extract_abundance(args.depth, valid_names)
    np.save(os.path.join(args.out_dir, "view_1.npy"), view_1)
    print(f"Features successfully extracted and saved to {args.out_dir}")


if __name__ == "__main__":
    main()
