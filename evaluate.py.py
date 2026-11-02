import argparse
from collections import defaultdict
import os
from Bio import SeqIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def calculate_n50(lengths):
    if not lengths:
        return 0
    lengths = sorted(lengths, reverse=True)
    total_len = sum(lengths)
    cum_len = 0
    for l in lengths:
        cum_len += l
        if cum_len >= total_len / 2.0:
            return l
    return 0


def parse_checkm(tsv_path):
    tiers = {">90%": 0, ">80%": 0, ">70%": 0, ">60%": 0, ">50%": 0}
    if not os.path.exists(tsv_path):
        return tiers

    with open(tsv_path, "r") as f:
        lines = f.readlines()
        if len(lines) <= 1:
            return tiers

        for line in lines[1:]:
            parts = line.strip().split("\t")
            if len(parts) >= 7:
                try:
                    comp = float(parts[5])
                    cont = float(parts[6])
                    if cont < 5.0:
                        if comp >= 90.0:
                            tiers[">90%"] += 1
                        if comp >= 80.0:
                            tiers[">80%"] += 1
                        if comp >= 70.0:
                            tiers[">70%"] += 1
                        if comp >= 60.0:
                            tiers[">60%"] += 1
                        if comp >= 50.0:
                            tiers[">50%"] += 1
                except ValueError:
                    continue
    return tiers


def parse_ground_truth(gt_path):
    gt_dict = {}
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                gt_dict[parts[0]] = parts[1]
    return gt_dict


def evaluate_bins(bins_dir, gt_dict):
    bin_stats = []
    total_correct_bp = 0
    total_bp = 0

    if not os.path.exists(bins_dir):
        return pd.DataFrame(), 0.0

    for bin_file in os.listdir(bins_dir):
        if not (bin_file.endswith(".fa") or bin_file.endswith(".fasta")):
            continue

        bin_path = os.path.join(bins_dir, bin_file)

        gt_counts = defaultdict(int)
        bin_lengths = []
        bin_total_len = 0

        for record in SeqIO.parse(bin_path, "fasta"):
            length = len(record.seq)
            bin_lengths.append(length)
            bin_total_len += length

            true_label = gt_dict.get(record.id, "Unknown")
            gt_counts[true_label] += length

        if bin_total_len == 0:
            continue

        n50 = calculate_n50(bin_lengths)

        dominant_gt = max(gt_counts.items(), key=lambda x: x[1])[0]
        dominant_bp = gt_counts[dominant_gt]

        purity = dominant_bp / bin_total_len if bin_total_len > 0 else 0

        if dominant_gt != "Unknown":
            total_correct_bp += dominant_bp

        total_bp += bin_total_len

        bin_stats.append(
            {
                "Bin_ID": bin_file,
                "Total_Length": bin_total_len,
                "N50": n50,
                "Dominant_GT": dominant_gt,
                "Purity": purity,
            }
        )

    global_accuracy = (
        total_correct_bp / total_bp if total_bp > 0 else 0.0
    )
    return pd.DataFrame(bin_stats), global_accuracy


def plot_results(checkm_tiers, bin_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. CheckM Tiers Barplot (Fixing seaborn palette warning)
    plt.figure(figsize=(8, 6))
    tiers = list(checkm_tiers.keys())
    counts = list(checkm_tiers.values())

    sns.barplot(x=tiers, y=counts, hue=tiers, palette="Blues_d", legend=False)
    plt.title(
        "Reconstructed Bins at Different Completeness Levels\n(Contamination < 5%)",
        fontsize=14,
    )
    plt.xlabel("Completeness Threshold", fontsize=12)
    plt.ylabel("Number of Bins", fontsize=12)

    for i, v in enumerate(counts):
        plt.text(
            i,
            v + max(counts) * 0.01 if max(counts) > 0 else 0.1,
            str(v),
            ha="center",
            va="bottom",
            fontsize=12,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "completeness_barplot.png"), dpi=300)
    plt.close()

    if not bin_df.empty:
        # 2. N50 Boxplot
        plt.figure(figsize=(8, 6))
        sns.boxplot(y=bin_df["N50"], color="skyblue", width=0.4)
        sns.stripplot(y=bin_df["N50"], color="black", alpha=0.5, jitter=True)
        plt.title("Distribution of N50 Across Reconstructed Bins", fontsize=14)
        plt.ylabel("N50 (bp)", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "n50_boxplot.png"), dpi=300)
        plt.close()

        # 3. Purity Histogram
        plt.figure(figsize=(8, 6))
        sns.histplot(bin_df["Purity"], bins=20, kde=True, color="green")
        plt.title("Distribution of Bin Purity", fontsize=14)
        plt.xlabel("Purity (Dominant GT Proportion)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "purity_histogram.png"), dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bins_dir", type=str, required=True)
    parser.add_argument("--checkm_tsv", type=str, required=True)
    parser.add_argument("--ground_truth", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    print("Parsing CheckM results...")
    checkm_tiers = parse_checkm(args.checkm_tsv)

    print("Parsing ground truth labels...")
    gt_dict = parse_ground_truth(args.ground_truth)

    print("Evaluating bins...")
    bin_df, global_accuracy = evaluate_bins(args.bins_dir, gt_dict)

    print("Generating visualizations...")
    plot_results(checkm_tiers, bin_df, args.output_dir)

    print("\n" + "=" * 40)
    print("Evaluation Summary")
    print("=" * 40)
    print(f"Total Bins Evaluated: {len(bin_df)}")
    print(f"Global Binning Accuracy (bp): {global_accuracy:.4f}")
    if not bin_df.empty:
        print(f"Median N50: {bin_df['N50'].median():.0f} bp")
        print(f"Mean Purity: {bin_df['Purity'].mean():.4f}")
    print("-" * 40)
    for tier, count in checkm_tiers.items():
        print(f"Bins with Completeness {tier}: {count}")
    print("=" * 40)
    print(f"Plots successfully saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
