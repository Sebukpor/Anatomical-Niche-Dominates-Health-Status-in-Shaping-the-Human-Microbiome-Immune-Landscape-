# Anatomical Niche Dominates Health Status in Shaping the Human Microbiome-Immune Landscape 
Abstract 
The human microbiome engages in dynamic, bidirectional interactions with the host immune 
system across diverse anatomical niches, yet the relative influence of anatomical context versus 
health status on microbial–immune organization remains incompletely resolved. Here, we 
analyzed a large-scale, multi-site dataset comprising 1,813 samples from four body habitats 
(stool, oral, nasal, and skin), integrating 16S rRNA gene sequencing with 62-plex cytokine 
profiling. Using compositional data analysis with Bayesian–multiplicative zero replacement, 
permutation-derived correlation thresholds, and machine-learning–based classification, we 
quantified the determinants of microbiome immune architecture across sites and clinical states. 

Anatomical site emerged as the dominant driver of microbial–immune variation, explaining 
substantially more variance than health status (PERMANOVA 𝑅2 ≈ 0.28vs. 0.08). Health-state 
classification performance exhibited a clear site-dependent gradient (stool > oral > nasal > skin), 
with peak accuracies achieved by XGBoost for stool (74.7%) and skin (64.0%), multilayer 
perceptron models for oral samples (75.3%), and Random Forest models for nasal samples 
(69.3%). Correlation network analysis revealed highly site-specific immune–microbial 
architectures: stool microbiomes were characterized by broad Phascolarctobacterium 
engagement with multiple inflammatory mediators (SDF1A 𝑟 = 0.380, IL1A 𝑟 = 0.360, IL8 𝑟 =
0.353); nasal habitats displayed a LEPTIN-centered bipolar regulatory axis involving Actinomyces 
(𝑟 = 0.389) and Corynebacterium (𝑟 = −0.383); skin microbiomes showed GMCSF-associated 
positive relationships with Bacillus (𝑟 = 0.308) and Staphylococcus (𝑟 = 0.280); and oral sites 
harbored the strongest individual association observed across all habitats, a negative correlation 
between Butyrivibrio and PDGFBB (𝑟 = −0.437, 𝑝 = 3.46 × 10−22). 

Differential abundance analyses identified Barnesiella and Odoribacter depletion in stool
associated clinical states and Staphylococcus expansion in nasal dysbiosis, whereas oral and skin 
sites exhibited minimal taxonomic shifts despite pronounced cytokine coupling. Longitudinal 
analyses further revealed persistent, site-dependent disruptions in microbial diversity following 
antibiotic exposure, most notably in the gut microbiome. Collectively, these results demonstrate 
that microbiome–immune crosstalk is governed predominantly by site-specific ecological 
constraints rather than health status alone, underscoring the need for anatomically 
contextualized frameworks in microbiome-based diagnostics and therapeutic development. 

## Data Sources

The genomic data used in this study originates from the **Zindi MPEG-G Compression Challenge**, compressed MPEG-G files which were later decompressed to FASTQ for downstream task:

- **MPEG-G compressed genomic data**:  
  [Kaggle Dataset: maestroalert/trainfiles](https://www.kaggle.com/datasets/maestroalert/trainfiles)

- **Decompressed FASTQ files**:  
  [Kaggle Dataset: divinesebukpor/trainfiles](https://www.kaggle.com/datasets/divinesebukpor/trainfiles)
  
- **DOI**:
```
  https://doi.org/10.34740/kaggle/dsv/12310273
```

- **Cite**:
```
@misc{divine_sebukpor_2025,
	title={Trainfiles},
	url={https://www.kaggle.com/dsv/12310273},
	DOI={10.34740/KAGGLE/DSV/12310273},
	publisher={Kaggle},
	author={Divine Sebukpor},
	year={2025}
}
```
   

- **Data** (Anonymized & Processed):
  - [`cytokine_profiles.csv`](cytokine_profiles.csv): High-resolution cytokine measurements (~66 analytes) across samples.
  - [`Train.csv`](Train.csv): Merged IDs of training dataset for microbiome, cytokines, and metadata.
  - [`Train_Subjects.csv`](Train_Subjects.csv): Subject-level metadata.

 

## Reproducibility Package Contents

The complete reproducibility package includes:

1. `code/` directory containing:
   - `analysis.py`: End-to-end analysis pipeline
   - `microbime_cytokine_preprocessing_pipeline.py`: Data cleaning, merging and normalization functions for cytokine and microbiome
   - `decompressing_mpegg_to_fastq.py`: Decompressing MPEG-G compressed file to fastq
   - `taxonomic_classification.py`: Taxonomic Classification of 16S rRNA Amplicons Using Kraken2 + Bracken (SILVA 138 database)
   


2. `results/` directory containing:
   - `Nasal/`: All generated figures in PDF/PNG formats, csv files etc.
   - `Stool/`: All generated figures in PDF/PNG formats, csv files etc.
   - `Mouth/`: All generated figures in PDF/PNG formats, csv files etc.
   - `Skin/`: All generated figures in PDF/PNG formats, csv files etc.
   - `UMAP Visualisations`
   - `Combined Permanova`
   - `Analysis Report.md`

3. Environment specifications:
   - `environment.yml`: Conda environment configuration
   - `requirements.txt`: Python package dependencies
   - `Dockerfile`: Containerized execution environment

5. Execution documentation:
   - `README.md`: Setup and execution instructions

All code includes comprehensive docstrings following NumPy format standards and unit tests covering core functionality. The pipeline can be executed on standard computing hardware (16 CPU cores, 64GB RAM) with an estimated runtime of 12-24 hours for complete re-analysis.


# Supplementary Materials

## Supplementary Methods

### Detailed Bayesian-Multiplicative Replacement Algorithm

For compositional microbiome data with zero values, we implement the Bayesian-multiplicative replacement method with the following mathematical formalism:

Let **x** = [x₁, x₂, ..., xₙ] be a composition with *n* taxa, where some components may be zero.

The replacement value δ is calculated as:
```
δ = 0.65 × min({xᵢ | xᵢ > 0})
```

The replacement procedure follows:
1. Count zero components: *m* = |{i | xᵢ = 0}|
2. Compute replacement sum: Sᵣ = m × δ
3. Compute non-zero sum: Sₙ = Σ{xᵢ | xᵢ > 0}
4. Scale non-zero components: xᵢ' = xᵢ × (1 - Sᵣ/Sₙ) for all xᵢ > 0
5. Replace zeros: xᵢ' = δ for all xᵢ = 0

The multiplicative adjustment factor α is calculated to preserve the total sum constraint:
```
α = 1 / (Σ xᵢ')
```

The final adjusted composition is:
```
xᵢ'' = α × xᵢ' for all i
```

This approach preserves the relative ratios between non-zero components while providing a statistically sound method for zero replacement in compositional data.

### SMOTE Algorithm Mathematical Formulation

The Synthetic Minority Oversampling Technique (SMOTE) generates synthetic samples according to:

For each minority class sample **xᵢ**, identify its *k* nearest neighbors (using Euclidean distance) within the same class.

For each neighbor **x̂ᵢⱼ** (where *j* = 1, 2, ..., *k*):
```
xₙₑ𝓌 = xᵢ + λ × (x̂ᵢⱼ - xᵢ)
```
where λ ~ U(0,1) is a random number between 0 and 1.

The number of synthetic samples generated for each minority class instance is determined by the oversampling ratio required to achieve class balance. In our implementation, *k* was set to min(5, Nₘᵢₙ-1) where Nₘᵢₙ is the size of the smallest class.

### Computational Complexity Analysis

The integrated analysis pipeline demonstrated the following computational complexity:

* Bayesian-multiplicative replacement: O(N × T × Z)
* PERMANOVA with permutations: O(P × N² × D)
* Correlation analysis with permutation testing: O(P × M × C × N log N)
* SMOTE-augmented random forest: O(T × N' × F log N')

Where:
- N = number of samples
- T = number of trees
- Z = number of zeros
- P = number of permutations
- M = number of microbial taxa
- C = number of cytokines
- D = number of dimensions
- F = number of features
- N' = effective sample size after SMOTE

### Detailed Processing Pipeline

#### MPEG-G Decompression Protocol
1. Initialize Docker container: `docker run -v $(pwd):/data muefab/genie:latest`
2. For each sample ID in metadata:
   - Execute decompression: `genie decompress -i /data/input.mpg -o /data/output.fastq`
   - Verify integrity using MD5 checksums provided in manifest files
   - Parallelize across 16 CPU cores with joblib backend

#### Taxonomic Profiling Parameters
* Kraken2 parameters:
  ```
  --db /databases/SILVA138_k2db
  --threads 4
  --memory-mapping
  --use-names
  --report-zero-counts
  --confidence 0.1
  ```
* Bracken parameters:
  ```
  -d /databases/SILVA138_k2db
  -t 10
  -l G
  --read_len $(median_read_length)
  ```

#### Quality Control Thresholds
* Samples retained if:
  - Library size ≥ 1,000 reads (all samples exceeded 5,000)
  - No contamination detected in negative controls
  - SampleID matching rate ≥ 95% between metadata and sequence files
* Features retained if:
  - Detected in ≥5% of samples within a body site
  - Mean relative abundance ≥0.01%
  - Not flagged as environmental contaminants

#### Statistical Validation Procedures
* Permutation testing for correlation thresholds:
  1. For each body site, generate 1,000 permuted datasets by randomly shuffling cytokine values
  2. Compute all Spearman correlations in permuted datasets
  3. Determine site-specific threshold as 95th percentile of |ρ| values
  4. Apply threshold to observed correlations

* Differential abundance validation:
  1. Perform Kolmogorov-Smirnov and Anderson-Darling tests on CLR values
  2. If p < 0.05 for either test, replace Mann-Whitney U with permutation test (10,000 iterations)
  3. Report both raw and FDR-corrected p-values

### Machine Learning Implementation Details

#### Hyperparameter Grid Search Configuration
```
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, None],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced']
}
```

#### Cross-Validation Strategy
1. Stratified 10-fold cross-validation preserving class distribution
2. Nested cross-validation for hyperparameter tuning (3-fold inner loop)
3. Performance metrics computed per fold and averaged with standard deviations
4. Feature importance assessed via permutation importance (10 permutations per feature)

#### Validation of SMOTE Efficacy
1. Compare class distribution before and after SMOTE application
2. Evaluate model performance on original test set (without SMOTE)
3. Conduct ablation study removing SMOTE to quantify performance impact
4. Analyze feature importance stability across cross-validation folds

## Supplementary Tables

### Supplementary Table S1. Complete Microbe-Cytokine Correlation Network (FDR < 0.05)

| Body Site | Microbial Taxon | Cytokine | ρ | p-value | FDR | Functional Context |
|-----------|----------------|----------|-------|---------------------|-------|----------------------|
| Gut | Phascolarctobacterium | SDF1A | 0.38 | 2.42 × 10⁻¹⁸ | 1.21 × 10⁻¹⁶ | Chemokine signaling |
| Gut | Phascolarctobacterium | IL-1α | 0.36 | 1.57 × 10⁻¹⁶ | 3.93 × 10⁻¹⁵ | Innate immune tone |
| Gut | Coprococcus | GM-CSF | -0.36 | 1.07 × 10⁻¹⁶ | 3.93 × 10⁻¹⁵ | Myeloid regulation |
| Gut | Streptococcus | IL-18 | 0.35 | 1.41 × 10⁻¹⁵ | 3.93 × 10⁻¹⁴ | Inflammasome activation |
| Gut | Faecalibacterium | IL-12p70 | -0.34 | 4.77 × 10⁻¹⁵ | 1.06 × 10⁻¹³ | Anti-inflammatory regulation |
| Gut | Eubacterium | MCP-1 | -0.33 | 8.91 × 10⁻¹⁵ | 1.59 × 10⁻¹³ | Monocyte recruitment |
| Nasal | Actinomyces | Leptin | 0.39 | 1.41 × 10⁻¹⁹ | 2.82 × 10⁻¹⁸ | Metabolic-immune coupling |
| Nasal | Corynebacterium | Leptin | -0.38 | 5.96 × 10⁻¹⁹ | 5.96 × 10⁻¹⁸ | Counter-regulatory signaling |
| Nasal | Corynebacterium | GM-CSF | -0.38 | 1.23 × 10⁻¹⁸ | 8.20 × 10⁻¹⁸ | Myeloid suppression |
| Nasal | Pseudarcicella | Leptin | -0.37 | 1.91 × 10⁻¹⁷ | 9.55 × 10⁻¹⁷ | Metabolic adaptation |
| Nasal | Neisseria | IL-15 | 0.36 | 4.73 × 10⁻¹⁷ | 1.89 × 10⁻¹⁶ | Mucosal immunity |
| Nasal | Moraxella | IP-10 | -0.35 | 8.17 × 10⁻¹⁷ | 2.72 × 10⁻¹⁶ | Viral defense |
| Oral | Butyrivibrio | PDGF-BB | -0.44 | 3.46 × 10⁻²² | 6.92 × 10⁻²¹ | Tissue repair regulation |
| Oral | Rothia | IL-31 | 0.35 | 4.34 × 10⁻¹⁴ | 4.34 × 10⁻¹³ | Epithelial barrier regulation |
| Oral | Granulicatella | IL-31 | 0.33 | 1.05 × 10⁻¹² | 7.00 × 10⁻¹² | Pathobiont signaling |
| Oral | Comamonas | IP-10 | 0.32 | 7.52 × 10⁻¹² | 3.76 × 10⁻¹¹ | T-cell recruitment |
| Oral | Prevotella | IL-22 | 0.31 | 1.33 × 10⁻¹¹ | 5.32 × 10⁻¹¹ | Barrier defense |
| Oral | Veillonella | ENA-78 | 0.30 | 3.24 × 10⁻¹¹ | 1.08 × 10⁻¹⁰ | Neutrophil chemotaxis |
| Skin | Bacillus | GM-CSF | 0.31 | 1.15 × 10⁻⁹ | 1.92 × 10⁻⁸ | Environmental sensing |
| Skin | Staphylococcus | GM-CSF | 0.28 | 3.79 × 10⁻⁸ | 5.05 × 10⁻⁷ | Core microbiome signaling |
| Skin | Anaerococcus | Leptin | 0.26 | 2.40 × 10⁻⁷ | 2.67 × 10⁻⁶ | Anaerobic metabolism |
| Skin | Campylobacter | IL-5 | -0.26 | 4.70 × 10⁻⁷ | 4.18 × 10⁻⁶ | Type-2 immunity |
| Skin | Cutibacterium | IL-13 | 0.25 | 9.08 × 10⁻⁷ | 6.73 × 10⁻⁶ | Atopic signaling |
| Skin | Corynebacterium | VEGF | -0.24 | 1.70 × 10⁻⁶ | 1.06 × 10⁻⁵ | Angiogenesis regulation |

### Supplementary Table S2. Borderline Differential Abundance Signals (FDR < 0.10)

| Body Site | Contrast | Taxon | Mean Diff (Healthy - Condition) | p-value | FDR | Effect Size (Cohen's d) | 95% CI |
|-----------|----------|-------|----------------------------------|-----------|-------|------------------------|--------|
| Gut (Stool) | Healthy vs. Infection | Barnesiella | -0.009 | 2.0 × 10⁻³ | 0.086 | 0.42 | [-0.016, -0.002] |
| Gut (Stool) | Healthy vs. Infection | Odoribacter | -0.0004 | 2.0 × 10⁻³ | 0.086 | 0.41 | [-0.001, 0.0001] |
| Gut (Stool) | Healthy vs. Fiber | Oscillibacter | +0.003 | 1.4 × 10⁻² | 0.112 | 0.35 | [0.001, 0.005] |
| Gut (Stool) | Healthy vs. Fiber | Colidextribacter | +0.001 | 9.0 × 10⁻³ | 0.098 | 0.37 | [0.0003, 0.002] |
| Gut (Stool) | Healthy vs. Stress | Anaerostipes | -0.007 | 8.0 × 10⁻³ | 0.093 | 0.38 | [-0.013, -0.001] |
| Nasal | Healthy vs. Stress | Dolosigranulum | -0.021 | 3.0 × 10⁻³ | 0.071 | 0.45 | [-0.038, -0.004] |
| Oral | Healthy vs. Infection | Fusobacterium | +0.006 | 4.0 × 10⁻³ | 0.085 | 0.43 | [0.002, 0.010] |
| Oral | Healthy vs. Weight Gain | Lautropia | -0.003 | 6.0 × 10⁻³ | 0.094 | 0.40 | [-0.006, 0.0001] |
| Skin | Healthy vs. Antibiotics | Salinicoccus | +0.015 | 7.0 × 10⁻³ | 0.420 | 0.38 | [0.004, 0.026] |
| Skin | Healthy vs. Infection | Staphylococcus | +0.092 | 5.1 × 10⁻² | 0.680 | 0.29 | [-0.002, 0.186] |
| Skin | Healthy vs. Stress | Propionibacterium | -0.018 | 3.5 × 10⁻² | 0.470 | 0.32 | [-0.036, 0.0004] |

### Supplementary Table S3. Complete Machine Learning Performance Metrics

| Component | Software/Library | Version | Parameters/Settings |
|-----------|------------------|---------|---------------------|
| MPEG-G Decompression | Genie | v1.0.3 | --threads 16 |
| Quality Control | FastQC | v0.11.9 | Default |
| Taxonomic Classification | Kraken2 | v2.1.3 | --db SILVA138_k2db, --threads 4, --memory-mapping |
| Abundance Estimation | Bracken | v2.9 | -l G, -t 10 |
| Data Processing | pandas | v2.1.4 | N/A |
|  | NumPy | v1.26.4 | N/A |
| Statistical Analysis | scikit-bio | v0.5.9 | Default |
|  | scipy | v1.12.0 | N/A |
| Machine Learning | scikit-learn | v1.5.0 | n_estimators=200, max_depth=10 |
|  | imbalanced-learn | v0.11.0 | k_neighbors=min(5, min_class-1) |
| Visualization | seaborn | v0.13.2 | Default |
|  | matplotlib | v3.8.3 | DPI=600 for publication figures |
| Dimensionality Reduction | umap-learn | v0.5.5 | n_neighbors=40, min_dist=0.1 |
| Random Seeds | Python random | 42 | For all stochastic processes |
|  | NumPy random | 42 | N/A |
|  | scikit-learn | 42 | N/A |

## Supplementary Figures Guide
*Supplementary Figures can be located in result directory*

*Supplementary Figure S1. Quality control metrics across body sites*
- Panel A: Read depth distribution by body site
- Panel B: Taxonomic detection rates before/after filtering
- Panel C: Sample collection timeline
- Panel D: Inter-sample correlation heatmap

*Supplementary Figure S2. Temporal dynamics of microbiome-immune interactions*
- Panel A: Longitudinal trajectories of cytokine-microbiome coupling strength
- Panel B: Recovery patterns following acute perturbations
- Panel C: Time-lagged correlation analysis between microbial shifts and cytokine responses
- Panel D: Persistence of dysbiosis markers across timepoints

*Supplementary Figure S3. Confusion matrices for site-specific classifiers*
- Panel A: Stool SMOTE-RF classifier
- Panel B: Nasal SMOTE-RF classifier
- Panel C: Oral SMOTE-RF classifier
- Panel D: Skin SMOTE-RF classifier

*Supplementary Figure S4. Permutation testing validation of correlation thresholds*
- Panel A-D: Null distribution of correlation coefficients by body site
- Dashed lines indicate 95th percentile thresholds used in main analysis
