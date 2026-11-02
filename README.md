# Rcbin: A Metagenomic Binning Framework via Cross-View Contrastive Learning and RL-Driven Dynamic Clustering

Rcbin is a deep learning-based metagenomic binning tool designed to recover high-quality metagenome-assembled genomes (MAGs) from complex microbial communities.

## Prerequisites and Dependencies

### External Bioinformatics Tools
Ensure the following tools are installed and added to your system PATH:
- BWA-MEM (for read mapping)
- MetaBAT2 (specifically `jgi_summarize_bam_contig_depths` for generating depth files)
- Prodigal (for ORF prediction)
- HMMER (for SCG marker identification)
- CheckM (for genome quality assessment)

## Quick Start and Pipeline

### Step 1: Preprocessing and Feature Extraction
Extract 136-dim k-mer (TNF) features and multi-sample depth features.
```bash
python preprocess.py \
    --fasta assembly.fasta \
    --depth depth.txt \
    --out_dir ./features \
    --min_len 1500
```
### Step 2: Model Training
Train the dual-branch encoders with contrastive learning and Q-Learning refinement.

```bash
python train_model.py \
    --data_path ./features \
    --save_path model.pth \
    --n_clusters 2000 \
    --epochs 100
```
### Step 3: Candidate Generation
### 3.1 Graph-based Generation (Recursive Leiden)
Note: You need to provide a markers.txt file containing contig_id and marker_gene_id mapping.

```bash
python generate_leiden_bins.py \
    --data_path ./features \
    --fasta_path assembly.fasta \
    --model_path model.pth \
    --marker_file markers.txt \
    --output_dir ./leiden_bins
```
### 3.2 K-Means Generation (Multi-resolution)
You can run this with different K values (e.g., 400, 600, 800, 1200) to enrich the candidate pool.

```bash
python generate_kmeans_bins.py \
    --data_path ./features \
    --fasta_path assembly.fasta \
    --model_path model.pth \
    --n_clusters 400 \
    --output_dir ./km_bins_400
```
### Step 4: Quality Assessment and Ensemble Integration
Run CheckM on the generated candidate bins first:

```bash
checkm lineage_wf -t 16 -x fa ./leiden_bins ./checkm_leiden_out
checkm qa ./checkm_leiden_out/lineage.ms ./checkm_leiden_out -f checkm_leiden.tsv -o 2 --tab_table

checkm lineage_wf -t 16 -x fa ./km_bins_400 ./checkm_km_out
checkm qa ./checkm_km_out/lineage.ms ./checkm_km_out -f checkm_km.tsv -o 2 --tab_table
```
Then run the ensemble script to get the final high-quality MAGs:

```bash
python ensemble_bins.py \
    --output_dir ./final_bins \
    --checkm_v17 checkm_leiden.tsv --dir_v17 ./leiden_bins \
    --checkm_km checkm_km.tsv --dir_km ./km_bins_400
```
### Step 5: Evaluation and Visualization
Compare the final bins against the ground truth to calculate Accuracy (bp) and generate distribution plots.

```bash
python evaluate.py \
    --bins_dir ./final_bins \
    --checkm_tsv checkm_ensemble.tsv \
    --ground_truth cami_ground_truth.tsv \
    --output_dir ./evaluation_plots
```
### Outputs
final_bins/: Contains the final refined .fa bins.

evaluation_plots/: Contains completeness_barplot.png, n50_boxplot.png, and purity_histogram.png.

Terminal prints global metrics including Accuracy (bp), Median N50, and Mean Purity.

### Citation
If you use Rcbin in your research, please cite our paper:

Rcbin: A Metagenomic Binning Framework via Cross-View Contrastive Learning and RL-Driven Dynamic Clustering. (Under Review)
