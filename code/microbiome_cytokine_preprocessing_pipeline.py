"""
Microbiome + Cytokine Preprocessing Pipeline
===========================================

This script performs publication-grade preprocessing of
Kraken2 + Bracken (SILVA 16S) genus-level output and merges it with
sample metadata and cytokine profiles.

Key features:
- Relative abundance normalization
- Depth-aware prevalence filtering
- Conservative taxon validation
- Environmental genera flagging
- Final merged dataset export

Author: <YOUR NAME>
Repository: <GITHUB REPO URL>
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List


# ======================================================
# Microbiome Preprocessor
# ======================================================

class MicrobiomePreprocessor:
    """
    Publication-grade preprocessing for Kraken2 + Bracken (SILVA 16S) output.

    Design principles:
    - Assumes Bracken was run at genus level (-l G)
    - Treats SILVA genus / group / clade labels as FEATURES
    - Relative abundance normalization only (no CLR)
    - Depth-aware prevalence filtering
    - Conservative taxon filtering
    """

    def __init__(
        self,
        min_rel_abundance: float = 1e-4,
        min_prevalence: float = 0.05,
        min_total_reads: int = 5000,
        verbose: bool = True,
    ):
        self.min_rel_abundance = min_rel_abundance
        self.min_prevalence = min_prevalence
        self.min_total_reads = min_total_reads
        self.verbose = verbose

        # Truly non-informative / malformed labels
        self.invalid_taxon_patterns = [
            "uncultured",
            "unclassified",
            "unknown",
            "metagenome",
            "[",
            "]",
        ]

        # Optional flag-only environmental/extreme genera
        self.environmental_genera = {
            "Thermus",
            "Xylella",
            "Zobellia",
            "Thermosediminibacter",
            "Nitratiruptor",
            "Halococcus",
        }

    # --------------------------------------------------
    # Taxon validation
    # --------------------------------------------------

    def _is_valid_taxon(self, taxon: str) -> bool:
        """
        Conservative check for biologically interpretable taxa.
        SILVA groups/clades are retained.
        """
        taxon_lower = taxon.lower()
        return not any(pat in taxon_lower for pat in self.invalid_taxon_patterns)

    # --------------------------------------------------
    # Core preprocessing
    # --------------------------------------------------

    def preprocess(
        self, file_path: str
    ) -> Tuple[pd.DataFrame, Dict[str, object]]:
        """
        Run preprocessing pipeline.

        Returns
        -------
        microbiome_df : pd.DataFrame
            Relative abundance table (SampleID + taxa)
        qc_metrics : dict
            Quality-control summary statistics
        """

        # -------------------------
        # Load data
        # -------------------------
        df = pd.read_csv(file_path)

        if "SampleID" not in df.columns:
            raise ValueError("Input CSV must contain a 'SampleID' column.")

        df = df.set_index("SampleID")
        raw_counts = df.copy()

        n_samples_raw = raw_counts.shape[0]
        n_taxa_raw = raw_counts.shape[1]

        # -------------------------
        # Remove empty & low-depth samples
        # -------------------------
        library_sizes = raw_counts.sum(axis=1)
        raw_counts = raw_counts.loc[library_sizes >= self.min_total_reads]

        n_samples_filtered = raw_counts.shape[0]

        if raw_counts.empty:
            raise ValueError("All samples removed due to low read depth.")

        # -------------------------
        # Relative abundance normalization
        # -------------------------
        rel_abund = raw_counts.div(raw_counts.sum(axis=1), axis=0)

        # -------------------------
        # Depth-aware prevalence filtering
        # -------------------------
        prevalence = (rel_abund >= self.min_rel_abundance).mean(axis=0)
        keep_taxa = prevalence[prevalence >= self.min_prevalence].index
        rel_abund = rel_abund[keep_taxa]

        n_taxa_prevalence = rel_abund.shape[1]

        # -------------------------
        # Remove malformed / uninformative taxa
        # -------------------------
        valid_taxa = [t for t in rel_abund.columns if self._is_valid_taxon(t)]
        rel_abund = rel_abund[valid_taxa]

        n_taxa_final = rel_abund.shape[1]

        # -------------------------
        # Flag environmental genera
        # -------------------------
        detected_environmental = sorted(
            set(rel_abund.columns).intersection(self.environmental_genera)
        )

        # -------------------------
        # QC metrics
        # -------------------------
        qc_metrics = {
            "n_samples_raw": n_samples_raw,
            "n_samples_retained": n_samples_filtered,
            "n_taxa_raw": n_taxa_raw,
            "n_taxa_after_prevalence": n_taxa_prevalence,
            "n_taxa_final": n_taxa_final,
            "mean_library_size": library_sizes.loc[raw_counts.index].mean(),
            "median_library_size": library_sizes.loc[raw_counts.index].median(),
            "min_library_size": library_sizes.loc[raw_counts.index].min(),
            "mean_prevalence_retained": prevalence.loc[keep_taxa].mean(),
            "min_prevalence_retained": prevalence.loc[keep_taxa].min(),
            "environmental_genera_detected": detected_environmental,
        }

        # -------------------------
        # Logging
        # -------------------------
        if self.verbose:
            print("\n🔬 Microbiome Preprocessing QC Summary")
            print("=" * 50)
            for k, v in qc_metrics.items():
                if isinstance(v, float):
                    print(f"{k:35s}: {v:.4f}")
                else:
                    print(f"{k:35s}: {v}")
            print("=" * 50)

            if detected_environmental:
                print(
                    "\n⚠️ Environmental / extreme genera detected "
                    "(flagged, not removed):"
                )
                for g in detected_environmental:
                    print(f"  - {g}")

        # Restore SampleID
        rel_abund = rel_abund.reset_index()

        return rel_abund, qc_metrics


# ======================================================
# Main Pipeline
# ======================================================

if __name__ == "__main__":

    # -------- File paths (REPLACE WITH YOUR PATHS) --------
    BRACKEN_GENUS_PATH = "PATH/TO/bracken_genus_abundances.csv"
    TRAIN_METADATA_PATH = "PATH/TO/Train.csv"
    CYTOKINE_PATH = "PATH/TO/cytokine_profiles.csv"
    OUTPUT_PATH = "PATH/TO/merged_microbiome_cytokines.csv"

    # -------- Run microbiome preprocessing --------
    preprocessor = MicrobiomePreprocessor(
        min_rel_abundance=1e-4,
        min_prevalence=0.05,
        min_total_reads=5000,
        verbose=True,
    )

    microbiome_rel, qc_metrics = preprocessor.preprocess(
        BRACKEN_GENUS_PATH
    )

    # -------- Load metadata and cytokines --------
    train = pd.read_csv(TRAIN_METADATA_PATH)
    cytokines = pd.read_csv(CYTOKINE_PATH)

    # -------- Map filenames to SampleID --------
    train["filename_noext"] = train["filename"].str.replace(
        ".mgb", "", regex=False
    )

    # -------- Merge microbiome with metadata --------
    microbiome_processed = microbiome_rel.rename(
        columns={"SampleID": "FileID"}
    )

    microbiome_merged = pd.merge(
        microbiome_processed,
        train[["SampleID", "SampleType", "filename_noext"]],
        left_on="FileID",
        right_on="filename_noext",
        how="inner",
    )

    cols = (
        ["SampleID", "SampleType"]
        + [
            c
            for c in microbiome_merged.columns
            if c
            not in [
                "SampleID",
                "SampleType",
                "FileID",
                "filename_noext",
            ]
        ]
    )

    microbiome_merged = microbiome_merged[cols]

    # -------- Merge with cytokine profiles --------
    final_merged = pd.merge(
        microbiome_merged,
        cytokines,
        on="SampleID",
        how="inner",
    )

    # -------- Save final dataset --------
    final_merged.to_csv(OUTPUT_PATH, index=False)

    print("\n✅ Final merged dataset saved")
    print("Shape:", final_merged.shape)
    print(final_merged.head())
